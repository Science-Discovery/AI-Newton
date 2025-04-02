import torch
import torch.nn as nn


class ConceptGroupPolicy(nn.Module):
    def __init__(
        self,
        num_in: int = 4,
        hidden_layers: list[int] = [64, 64],
        activation: str = 'GELU',
        num_out: int = 1,
        dtype=torch.float32,
        device="cpu",  # "cuda" is now not confirmed to work 
    ):
        super().__init__()

        self.num_in = num_in
        self.num_out = num_out
        layer_sizes = [num_in] + hidden_layers
        self.network = nn.Sequential(
            *[
                f(x)
                for x in zip(layer_sizes[:-1], layer_sizes[1:])
                for f in (
                    lambda x: nn.Linear(*x, dtype=dtype, device=device),
                    lambda _: getattr(nn, activation)(),
                )
            ],
            nn.Linear(
                layer_sizes[-1], self.num_out, dtype=dtype, device=device
            ),
            nn.Sigmoid(),
        )

        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
