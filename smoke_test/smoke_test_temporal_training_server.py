from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as L
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from server_example.train_ttt_ape_xxl_server import (
    build_data_module,
    build_strategy,
    initialize_matching_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-data DDP smoke test for temporal TTT.")
    parser.add_argument("--data-dir", type=Path, default=Path("~/working/datasets"))
    parser.add_argument("--init-checkpoint-path", type=Path, default=None)
    parser.add_argument("--devices", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    data_module = build_data_module(
        data_dir=data_dir,
        dataset_names=["burgers"],
        batch_size=1,
        num_workers=1,
        downsample_factor=2,
        train_unrolling_steps=4,
        train_step_size=4,
        test_unrolling_steps=4,
        max_channels=2,
    )
    strategy = build_strategy(
        seed=42,
        model_type="PDE-S",
        sample_size=128,
        in_channels=2,
        out_channels=2,
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        use_ttt_window_attention=False,
        token_mixer_type="attention",
        use_ttt_state_cache_train=False,
        training_mode="temporal_rollout",
        train_unrolling_steps=4,
        tbptt_chunk_size=2,
        gradient_accumulation_batches=1,
        temporal_ttt_enabled=True,
        temporal_ttt_layer_type="mlp",
        temporal_ttt_mini_batch_size=64,
        temporal_ttt_base_lr=1.0,
        temporal_ttt_gate_init=0.1,
        temporal_ttt_learning_rate=1.0e-4,
        learning_rate=1.0e-5,
    )
    if args.init_checkpoint_path is not None:
        initialize_matching_weights(strategy, args.init_checkpoint_path)

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.is_available() and args.devices > 1
            else "auto"
        ),
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=2,
        limit_val_batches=1,
        log_every_n_steps=1,
    )
    trainer.fit(strategy, datamodule=data_module)
    if trainer.is_global_zero:
        gate = strategy.model.model.temporal_ttt.gate.detach()
        print(
            "Temporal TTT real-data DDP smoke test passed: "
            f"global_step={trainer.global_step} gate_mean_abs={gate.abs().mean().item():.6g}"
        )


if __name__ == "__main__":
    main()
