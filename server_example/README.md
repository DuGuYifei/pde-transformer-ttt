# Full-map ViT3 configuration

The paper-facing Full-map ViT3 experiment uses the internal compatibility
name `global_h_vittt` in saved configurations and checkpoints.

```bash
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-h-vittt_128_60sims.yaml \
  --check-config

python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-h-vittt_128_60sims.yaml
```
