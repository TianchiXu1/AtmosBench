# Third-party restoration architectures

| Architecture | Source revision | License and use |
|---|---|---|
| AdaIR | [c-yn/AdaIR](https://github.com/c-yn/AdaIR/tree/ccb8b98e49614e07badd0641e5163fa7635c2f02) | MIT. The selected architecture and its license are included. |
| PromptIR | [va1shn9v/PromptIR at `106159a`](https://github.com/va1shn9v/PromptIR/tree/106159a) | Academic Public License. Obtain the original source under its noncommercial terms. |
| DATPRL-IR | [GuangluDong0728/DATPRL-IR at `5de5764`](https://github.com/GuangluDong0728/DATPRL-IR/tree/5de5764) | Obtain the original source and keep its license with it. Verify the architecture files against their retained SHA-256 values. |

To obtain the PromptIR and DATPRL-IR source for training or evaluation, run these commands from `code/`.

```bash
git clone https://github.com/va1shn9v/PromptIR.git third_party/PromptIR
git -C third_party/PromptIR checkout --detach 106159ab809101f2e25b6714195cd6fa9a938d36
(cd third_party/PromptIR && shasum -a 256 -c ../../assets/promptir_source.sha256)

git clone https://github.com/GuangluDong0728/DATPRL-IR.git third_party/DATPRL-IR
git -C third_party/DATPRL-IR checkout --detach 5de5764
(cd third_party/DATPRL-IR && shasum -a 256 -c ../../assets/datprl_source.sha256)
```

The AtmosBench checkpoints use task-specific random initialization and training. Upstream image-restoration checkpoints are for a different task. The `third_party` notices apply to their respective source files.
