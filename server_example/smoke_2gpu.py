"""Small Lightning DDP smoke test for visible GPUs.

Examples:
    CUDA_VISIBLE_DEVICES=0 python smoke_2gpu.py --devices 1
    CUDA_VISIBLE_DEVICES=0,1 python smoke_2gpu.py
    CUDA_VISIBLE_DEVICES=0,1,2 python smoke_2gpu.py --devices 3

This does not use PDE data or the TTT model. It only checks whether PyTorch,
Lightning, CUDA, and the requested distributed strategy can complete a tiny
training loop on the visible CUDA devices.
"""

from __future__ import annotations

import argparse
import os

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x)
        loss = self.loss(pred, y)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Lightning on visible CUDA devices.")
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of visible CUDA devices to use. Default: all visible devices.",
    )
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = args.devices if args.devices is not None else visible_devices

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("visible cuda device count:", visible_devices)
    print("requested devices:", devices)
    print("strategy:", args.strategy)
    print("precision:", args.precision)

    if not torch.cuda.is_available() or visible_devices < 1:
        raise SystemExit("Need at least 1 visible CUDA device for this smoke test.")
    if devices < 1:
        raise SystemExit("--devices must be >= 1.")
    if devices > visible_devices:
        raise SystemExit(f"Requested {devices} devices, but only {visible_devices} are visible.")

    torch.manual_seed(42)
    x = torch.randn(64, 16)
    y = torch.randn(64, 4)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, num_workers=0)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=args.strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(TinyModule(), train_dataloaders=loader)
    print(f"GPU smoke test finished successfully with devices={devices}, strategy={args.strategy}.")


if __name__ == "__main__":
    main()
