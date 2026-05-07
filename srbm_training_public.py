"""
Public SRBM training core.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SRBM(nn.Module):
    """

    Energy:
        E(v, h) = -v @ b_v - h @ b_h - v @ W.T @ h - 0.5 * h @ J @ h
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        init_scale: float,
        coupling_strength: float,
        coupling_sparsity: float,
    ) -> None:
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.W = nn.Parameter(init_scale * torch.randn(n_hidden, n_visible))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

        j_init = coupling_strength * torch.randn(n_hidden, n_hidden)
        keep_mask = torch.rand(n_hidden, n_hidden) > coupling_sparsity
        j_init = j_init * keep_mask.float()
        j_init = (j_init + j_init.t()) / 2
        j_init.fill_diagonal_(0)
        self.J = nn.Parameter(j_init)

    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        v_term = torch.matmul(v, self.v_bias)
        h_term = torch.matmul(h, self.h_bias)
        vh_term = torch.sum(v * F.linear(h, self.W.t()), dim=1)
        hh_term = torch.sum(h * torch.matmul(h, self.J), dim=1) / 2
        return -v_term - h_term - vh_term - hh_term

    def free_energy(
        self,
        v: torch.Tensor,
        mean_field_iterations: int,
    ) -> torch.Tensor:
        mu = self._mean_field_hidden(v, mean_field_iterations)

        v_term = torch.matmul(v, self.v_bias)
        h_term = torch.matmul(mu, self.h_bias)
        vh_term = torch.sum(v * F.linear(mu, self.W.t()), dim=1)
        hh_term = torch.sum(mu * torch.matmul(mu, self.J), dim=1) / 2
        expected_energy = -v_term - h_term - vh_term - hh_term

        eps = torch.finfo(mu.dtype).eps
        entropy = -torch.sum(
            mu * torch.log(mu + eps) + (1 - mu) * torch.log(1 - mu + eps),
            dim=1,
        )
        return expected_energy - entropy

    def sample_h_given_v(
        self,
        v: torch.Tensor,
        sampling_method: str,
        mean_field_iterations: int,
        gibbs_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sampling_method == "mean_field":
            mu = self._mean_field_hidden(v, mean_field_iterations)
            return mu, torch.bernoulli(mu)
        if sampling_method == "gibbs":
            return self._gibbs_hidden(v, gibbs_steps)
        raise ValueError("sampling_method must be 'mean_field' or 'gibbs'.")

    def sample_v_given_h(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, torch.bernoulli(p_v)

    def contrastive_divergence(
        self,
        v0: torch.Tensor,
        cd_steps: int,
        sampling_method: str,
        mean_field_iterations: int,
        gibbs_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ph0, _ = self.sample_h_given_v(
            v0,
            sampling_method,
            mean_field_iterations,
            gibbs_steps,
        )

        vk = v0
        for _ in range(cd_steps):
            _, hk = self.sample_h_given_v(
                vk,
                sampling_method,
                mean_field_iterations,
                gibbs_steps,
            )
            _, vk = self.sample_v_given_h(hk)

        phk, _ = self.sample_h_given_v(
            vk,
            sampling_method,
            mean_field_iterations,
            gibbs_steps,
        )
        return v0, ph0, vk, phk

    def _mean_field_hidden(
        self,
        v: torch.Tensor,
        iterations: int,
    ) -> torch.Tensor:
        mu = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        for _ in range(iterations):
            field = F.linear(v, self.W, self.h_bias) + torch.matmul(mu, self.J)
            mu = torch.sigmoid(field)
        return mu

    def _gibbs_hidden(
        self,
        v: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.bernoulli(torch.sigmoid(F.linear(v, self.W, self.h_bias)))

        for _ in range(steps):
            for j in torch.randperm(self.n_hidden, device=v.device):
                field_j = (
                    self.h_bias[j]
                    + torch.matmul(v, self.W[j])
                    + torch.matmul(h, self.J[j])
                )
                h[:, j] = torch.bernoulli(torch.sigmoid(field_j))

        field = F.linear(v, self.W, self.h_bias) + torch.matmul(h, self.J)
        return torch.sigmoid(field), h


def train_class_conditional_srbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_classes: int,
    n_hidden: int,
    init_scale: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    cd_steps: int,
    sampling_method: str,
    mean_field_iterations: int,
    gibbs_steps: int,
    device: str | torch.device,
) -> list[SRBM]:
    """Train one SRBM per class with contrastive divergence."""
    if sampling_method not in {"mean_field", "gibbs"}:
        raise ValueError("sampling_method must be 'mean_field' or 'gibbs'.")

    device = torch.device(device)
    n_visible = x_train.shape[1]

    models = [
        SRBM(
            n_visible=n_visible,
            n_hidden=n_hidden,
            init_scale=init_scale,
        ).to(device)
        for _ in range(n_classes)
    ]

    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)

    for class_idx, model in enumerate(models):
        x_class = x_train_t[y_train_t == class_idx]
        if x_class.numel() == 0:
            continue

        loader = DataLoader(
            TensorDataset(x_class),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(epochs):
            total_loss = 0.0

            for (v0,) in loader:
                v0, ph0, vk, phk = model.contrastive_divergence(
                    v0=v0,
                    cd_steps=cd_steps,
                    sampling_method=sampling_method,
                    mean_field_iterations=mean_field_iterations,
                    gibbs_steps=gibbs_steps,
                )

                pos_grad_w = torch.matmul(ph0.t(), v0) / v0.size(0)
                neg_grad_w = torch.matmul(phk.t(), vk) / vk.size(0)
                pos_grad_j = torch.matmul(ph0.t(), ph0) / v0.size(0)
                neg_grad_j = torch.matmul(phk.t(), phk) / vk.size(0)

                with torch.no_grad():
                    model.W.data += learning_rate * (pos_grad_w - neg_grad_w)

                    j_grad = pos_grad_j - neg_grad_j
                    j_grad = (j_grad + j_grad.t()) / 2
                    j_grad.fill_diagonal_(0)
                    model.J.data += learning_rate * j_grad

                    model.v_bias.data += learning_rate * (v0 - vk).mean(0)
                    model.h_bias.data += learning_rate * (ph0 - phk).mean(0)

                loss = (
                    model.free_energy(v0, mean_field_iterations).mean()
                    - model.free_energy(vk.detach(), mean_field_iterations).mean()
                )
                total_loss += loss.item()

            print(
                f"class={class_idx} epoch={epoch + 1} "
                f"loss={total_loss / len(loader):.6f}"
            )

    return models
