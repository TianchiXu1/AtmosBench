"""Model architectures used in the AtmosBench evaluation."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn


def _ensure_path(source_dir: str | Path) -> str:
    source = str(Path(source_dir).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    return source


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PromptIRAdapter(nn.Module):
    """NeurIPS 2023 PromptIR (decoder/prompt path enabled) for ATB -> BC."""

    def __init__(self, source_dir: str | Path, *, in_channels: int = 6) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        source = _ensure_path(source_dir)
        # unique module name: PromptIR and AdaIR both ship a `net.model` package
        module = _load_module_from_file("atmosbench_promptir_model", Path(source) / "net" / "model.py")
        self.backbone = module.PromptIR(inp_channels=self.in_channels,
                                       out_channels=self.in_channels, decoder=True)

    def forward(self, conditioning: torch.Tensor) -> dict[str, torch.Tensor]:
        if conditioning.ndim != 4 or conditioning.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, H, W) conditioning, got {tuple(conditioning.shape)}")
        restored = self.backbone(conditioning)
        return {"mean_norm": torch.sigmoid(restored[:, 0:1])}


class DATPRLIRAdapter(nn.Module):
    """ICLR 2026 DATPRL-IR dual-prompt-pool Restormer for ATB -> BC.

    The official all-in-one trainers (``train_3task.py``/``train_5task.py``)
    both instantiate ``DualPromptRestormerv3_only_task_prompt`` with its
    default configuration (dim=48, degradation pool 15x1024, top-k 3, two
    value tokens, cross-attention fusion); that registered official variant
    is preserved here.  Only the image-facing stems change through the documented
    ``img_channel`` constructor argument, so the official forward
    (including its ``+ x_in`` residual) runs unmodified; channel 0 -- whose
    residual base is the normalized log-ATB channel -- passes through sigmoid
    to produce ``mean_norm``.  The arch file uses package-relative imports,
    so it is loaded inside a synthetic ``atmosbench_datprl_net`` package.
    """

    def __init__(self, source_dir: str | Path, *, in_channels: int = 6) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        net_dir = Path(source_dir).resolve() / "All_in_One" / "net"
        if not (net_dir / "promptrestormerv3_arch.py").exists():
            raise FileNotFoundError(f"official DATPRL-IR arch not found under {net_dir}")
        package_name = "atmosbench_datprl_net"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(net_dir)]
            sys.modules[package_name] = package
        module = importlib.import_module(f"{package_name}.promptrestormerv3_arch")
        self.backbone = module.DualPromptRestormerv3_only_task_prompt(img_channel=self.in_channels)

    def forward(self, conditioning: torch.Tensor) -> dict[str, torch.Tensor]:
        if conditioning.ndim != 4 or conditioning.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, H, W) conditioning, got {tuple(conditioning.shape)}")
        restored = self.backbone(conditioning)
        return {"mean_norm": torch.sigmoid(restored[:, 0:1])}


class AdaIRAdapter(nn.Module):
    """ICLR 2025 AdaIR (frequency-mining decoder enabled) for ATB -> BC."""

    def __init__(self, source_dir: str | Path, *, in_channels: int = 6) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        source = _ensure_path(source_dir)
        module = _load_module_from_file("atmosbench_adair_model", Path(source) / "net" / "model.py")
        self.backbone = module.AdaIR(
            inp_channels=self.in_channels,
            out_channels=self.in_channels,
            decoder=True,
        )
        # AdaIR's FreModule hard-codes 3-channel convs that consume the raw
        # input image; widen exactly those image-facing stems to the
        # ATB-only channel count (fresh init, trained from scratch).  Channel 0
        # stays the normalized log-ATB residual base for the official + inp_img
        # residual and the adapter's channel-0 selection.
        replaced = 0
        for parent in self.backbone.modules():
            for name, child in parent.named_children():
                if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                    setattr(parent, name, nn.Conv2d(
                        self.in_channels, child.out_channels, kernel_size=child.kernel_size,
                        stride=child.stride, padding=child.padding,
                        dilation=child.dilation, groups=1, bias=child.bias is not None,
                    ))
                    replaced += 1
        if replaced == 0:
            raise RuntimeError("expected to widen AdaIR image-facing 3-channel stems, found none")

    def forward(self, conditioning: torch.Tensor) -> dict[str, torch.Tensor]:
        if conditioning.ndim != 4 or conditioning.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (B, {self.in_channels}, H, W) conditioning, got {tuple(conditioning.shape)}"
            )
        restored = self.backbone(conditioning)
        return {"mean_norm": torch.sigmoid(restored[:, 0:1])}
