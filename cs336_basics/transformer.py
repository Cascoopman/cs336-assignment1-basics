import math
import torch
import einops


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Construct a linear transformation module.

        Args:
            in_features: Final dimension of the input
            out_features: Final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        tensor = torch.Tensor(self.out_features, self.in_features, device=device)
        stdev = math.sqrt(2 / (self.in_features + self.out_features))

        self.weights = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(
                tensor=tensor,
                mean=0,
                std=stdev,
                a=-3 * abs(stdev),
                b=3 * abs(stdev),
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return einops.einsum(token_ids, self.weights, "... d_in, d_out d_in -> ... d_out")
