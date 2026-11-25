"""Load the selected retrieval checkpoint with its registered adaptation."""
from pathlib import Path

import torch

from backbone_adapters import AdaIRAdapter, DATPRLIRAdapter, PromptIRAdapter
from logunet import LogUNetBaseline


SOURCE_NAMES = {"adair": "AdaIR", "promptir": "PromptIR", "datprl": "DATPRL-IR"}
TRAINING_SCHEMA = "atmosbench-fp32-v1"
DEFAULT_MODEL_ID = "adair_physicalsdc_run1"


def build_model(row, third_party: Path):
    architecture = row["architecture"]
    channels = 7 if row["representation"] in ("physical_sdc", "physical_null") else 6
    if architecture == "logunet":
        return LogUNetBaseline(in_channels=channels, base_channels=128, depth=3,
                              out_channels=2 if row.get("head") == "gaussian" else 1)
    source = third_party / SOURCE_NAMES[architecture]
    if not source.is_dir():
        raise FileNotFoundError(f"Missing {architecture} source checkout at {source}; see THIRD_PARTY.md.")
    if architecture == "adair":
        return AdaIRAdapter(source, in_channels=channels)
    if architecture == "promptir":
        return PromptIRAdapter(source, in_channels=channels)
    return DATPRLIRAdapter(source, in_channels=channels)


def load_model(row, checkpoint: Path, third_party: Path, device: torch.device):
    if not checkpoint.is_file():
        if row.get("released", False):
            raise FileNotFoundError(f"Missing released checkpoint {checkpoint}; see assets/checkpoints.csv.")
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}; this evaluation requires a locally obtained weight.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    representation = state.get("atb_representation", "default")
    if representation != row["representation"]:
        raise ValueError(f"Checkpoint representation {representation!r} differs from {row['representation']!r}.")
    epoch = state.get("epoch")
    if row.get("epoch") is not None and epoch != row["epoch"]:
        raise ValueError(f"Selected epoch {row['epoch']} required, got {epoch} in {checkpoint.name}.")
    model = build_model(row, third_party)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), epoch


def load_fresh_model(checkpoint: Path, third_party: Path, device: torch.device):
    """Load a checkpoint created by the release's FP32 training entry point."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("training_schema") != TRAINING_SCHEMA:
        raise ValueError("Pass a best.pt produced by scripts/train.py as the fresh checkpoint.")
    row = {"architecture": state["architecture"], "representation": state["atb_representation"],
           "head": state["head"], "encoding_floor": state["encoding_floor"],
           "numerical_guard": state["numerical_guard"], "training_condition": state["training_condition"],
           "smoke": state["smoke"], "amp": False}
    model = build_model(row, third_party)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), state["epoch"], row
