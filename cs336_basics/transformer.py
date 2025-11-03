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


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct an embedding module.

        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors, i.e., d_model
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim

        self.embedding = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(
                tensor=torch.Tensor(num_embeddings, embedding_dim, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.embedding[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        """
        Construct the RMSNorm module.

        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = torch.nn.Parameter(data=torch.nn.init.ones_(tensor=torch.Tensor(self.d_model, device=device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a
        tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(einops.reduce(x**2, "... d_in -> ... 1", "mean") + self.eps)
        normalized = x / rms

        result = normalized * self.gamma

        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, device: torch.device = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = torch.round((8 * self.d_model / 3) / 64) * 64

        tensor = torch.Tensor(self.d_ff, self.d_model, device=device)
        stdev = math.sqrt(2 / (self.d_model + self.d_ff))

        self.weights_1 = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(
                tensor=tensor,
                mean=0,
                std=stdev,
                a=-3 * abs(stdev),
                b=3 * abs(stdev),
            )
        )
        self.weights_3 = self.weights_1.copy_()

        # TODO

    def forward(self, x):
        x = x * torch.sigmoid(x)
