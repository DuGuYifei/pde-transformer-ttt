# Thesis evaluation

`test_pretrained_mc_server.py` supports both thesis Lightning checkpoints and
official PDE-Transformer safetensors models. It performs autoregressive
rollout without gradients and writes CSV/JSON provenance and nRMSE results.

Use `--dataset-profile full_paper --strict-test-split` for the 16-family,
850-trajectory Full-256 test. Use `--id-ood-test` with a generated
`manifest.json` directory for the 17-family physical-parameter matrix.

Convenience launchers provide the recorded P, PDE-TTT, EMA three-seed, and
ID/OOD command patterns. Their checkpoint and data paths are environment
variables so the scripts can be reused without editing source files.

The evaluator reports both macro and micro nRMSE. Macro weights each PDE
family equally; micro weights each trajectory equally. They are different on
the unbalanced strict test and identical on the balanced generated matrix.
