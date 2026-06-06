"""Public decoder API."""
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from torch import nn

from .diffusion import cosine_schedule, sample_x_prev, get_x0_from_logits
from .model import DiffQEC
from .data import reshape_syndrome


class DiffQECDecoder:
    """High-level decoder interface.

    Wraps a trained DiffQEC model and runs the reverse diffusion sampling.
    """

    def __init__(
        self,
        model: DiffQEC,
        num_steps: int = 100,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.num_steps = num_steps
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.alpha_bar, self.beta, _ = cosine_schedule(num_steps)

    @torch.no_grad()
    def decode_syndromes(
        self,
        syndromes_dict: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a syndromes dict from extract_syndromes().

        Parameters
        ----------
        syndromes_dict : dict
            Must contain 'det_events' (shots, num_detectors) bool.

        Returns
        -------
        predicted_logical : (shots, L) bool
        confidence        : (shots,) float  — mean per-bit confidence
        """
        det_events = syndromes_dict["det_events"].astype(np.float32)
        shots, total_d = det_events.shape
        # Reshape to (shots, rounds, D)
        # Heuristic: use sqrt(total_d) as rounds estimate if not known
        rounds_guess = int(round(total_d ** 0.5)) or 1
        syndrome = reshape_syndrome(det_events, circuit=None, target_rounds=rounds_guess)
        syndrome_t = torch.from_numpy(syndrome).to(self.device)

        L = self.model.L
        # Start from noise
        xt = (torch.rand(shots, L, device=self.device) > 0.5).float()

        for t_idx in range(self.num_steps, 0, -1):
            t = torch.full((shots,), t_idx, device=self.device, dtype=torch.float32)
            logits = self.model(xt, t, syndrome_t)  # (shots, L, 2)
            x0_prob = torch.softmax(logits, dim=-1)[:, :, 1]  # P(x0=1)

            if t_idx > 1:
                beta_t = float(self.beta[t_idx - 1])
                ab_t = float(self.alpha_bar[t_idx])
                ab_tm1 = float(self.alpha_bar[t_idx - 1])
                xt = sample_x_prev(xt, x0_prob, beta_t, ab_t, ab_tm1)
            else:
                xt = (x0_prob > 0.5).float()

        predicted = xt.cpu().numpy().astype(bool)
        # Confidence = mean per-bit max probability
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.mean(dim=-1).cpu().numpy()
        return predicted, confidence

    def train(self, *args, **kwargs):
        """Train the underlying model. Delegates to training.train_diffqec."""
        from .training import train_diffqec
        return train_diffqec(self.model, *args, **kwargs)


def load_decoder(checkpoint_path: str, syndrome_shape: tuple, L: int, **model_kwargs) -> DiffQECDecoder:
    """Load a trained decoder from a checkpoint file."""
    model = DiffQEC(L=L, syndrome_shape=syndrome_shape, **model_kwargs)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    return DiffQECDecoder(model)


__all__ = ["DiffQECDecoder", "load_decoder"]
