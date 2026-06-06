"""Training loop, EMA, checkpointing, curriculum schedule."""
from pathlib import Path
from typing import Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .diffusion import cosine_schedule, sample_xt
from .model import DiffQEC


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def step(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v.detach().cpu(), 1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state: dict):
        self.shadow = {k: v.clone() for k, v in state.items()}


def train_epoch(
    model: DiffQEC,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional,
    alpha_bar: torch.Tensor,
    beta: torch.Tensor,
    T: int,
    device: torch.device,
    max_rounds: Optional[int] = None,
) -> dict:
    """Run one training epoch.

    Returns metrics dict with 'loss'.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        syndrome = batch["syndrome"].to(device)   # (batch, rounds, D)
        logical = batch["logical"].to(device)     # (batch, L)

        if max_rounds is not None and syndrome.size(1) > max_rounds:
            syndrome = syndrome[:, :max_rounds, :]

        b = logical.size(0)
        L = logical.size(1)

        # Sample random timestep per shot
        t = torch.randint(1, T + 1, (b,), device=device)
        # Corrupt x0 → xt
        xt = sample_xt(logical, t, alpha_bar.cpu().numpy())
        xt = xt.to(device)

        logits = model(xt, t, syndrome)  # (batch, L, 2)
        # Bitwise cross-entropy
        loss = nn.functional.cross_entropy(
            logits.view(b * L, 2),
            logical.long().view(b * L),
            reduction="mean",
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def train_diffqec(
    model: DiffQEC,
    train_loader: DataLoader,
    T: int = 100,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: Optional[Path] = None,
    ema_decay: float = 0.999,
    curriculum_schedule: Optional[dict] = None,
) -> DiffQEC:
    """Full training loop with EMA and checkpointing.

    curriculum_schedule: dict epoch -> max_rounds (e.g. {0:3, 3:5, 6:7, 9:9})
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    alpha_bar, beta, _ = cosine_schedule(T)
    alpha_bar_t = torch.from_numpy(alpha_bar).to(device)
    beta_t = torch.from_numpy(beta).to(device)

    ema = EMA(model, decay=ema_decay)
    model.to(device)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        max_rounds = curriculum_schedule.get(epoch, None) if curriculum_schedule else None
        # If curriculum says max_rounds is less than full, slice syndromes
        if max_rounds is not None:
            # dataset/dataloader may need to know; for simplicity we slice in train_epoch
            pass

        metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            alpha_bar_t, beta_t, T, device, max_rounds=max_rounds,
        )
        ema.step(model)
        print(f"Epoch {epoch+1}/{epochs}  loss={metrics['loss']:.4f}")

        if checkpoint_dir is not None:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt")

    # Return EMA-averaged model
    ema.apply_shadow(model)
    if checkpoint_dir is not None:
        torch.save({
            "model": model.state_dict(),
            "ema": ema.state_dict(),
        }, checkpoint_dir / "final.pt")
    return model
