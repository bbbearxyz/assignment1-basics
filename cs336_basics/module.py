import torch
import math
from einops import einsum, rearrange


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embeddings = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings[x]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gains = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            einsum(torch.square(x), "... d_model -> ...") / self.d_model + self.eps
        )
        return (x / rms.unsqueeze(-1) * self.gains).to(in_dtype)


class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_model, d_ff, device, dtype)
        self.activation = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.W1(x)) * self.W2(x)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        raw_dff = int(8 / 3 * d_model)
        d_ff = ((raw_dff + 64 - 1) // 64) * 64
        self.glu = GLU(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.glu(x))

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        theta = theta ** (-1 * torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k).unsqueeze(0)
        pos = torch.arange(0, max_seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        
        cos_val = torch.zeros((max_seq_len, d_k), device=device)
        sin_val = torch.zeros((max_seq_len, d_k), device=device)
        
        sin_val[:, 0::2] = torch.sin(pos * theta)
        sin_val[:, 1::2] = torch.sin(pos * theta)
        cos_val[:, 0::2] = torch.cos(pos * theta)
        cos_val[:, 1::2] = torch.cos(pos * theta)
        self.register_buffer("rope_cos_buffer", cos_val)
        self.register_buffer("rope_sin_buffer", sin_val)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_sin = self.rope_sin_buffer[token_positions]
        x_cos = self.rope_cos_buffer[token_positions]
        return x * x_cos + rotate_half(x) * x_sin
    
class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, dim: int):
        max_val, _ = torch.max(x, dim = dim, keepdim=True)
        exp = torch.exp(x - max_val)
        sum_val = torch.sum(exp, dim = dim, keepdim=True)
        return exp / sum_val
    
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Softmax = Softmax()
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        value = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(Q.shape[-1])
        if mask == None:
            normal_value = self.Softmax.forward(value, -1)
            return einsum(normal_value, V, "... queries values, ... values d_v -> ... queries d_v")
        normal_value = self.Softmax.forward(value.masked_fill(mask == 0, float("-inf")), -1)
        return einsum(normal_value, V, "... queries values, ... values d _v -> ... queries d_v")

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.Softmax = Softmax()
        self.num_heads = num_heads
        self.d_model = d_model

        self.Wo = Linear(d_model, d_model, device, dtype)
        self.Wq = Linear(d_model, d_model, device, dtype)
        self.Wk = Linear(d_model, d_model, device, dtype)
        self.Wv = Linear(d_model, d_model, device, dtype)

    def forward(self, in_features: torch.Tensor):
        len = in_features.shape[-2]
        causal_mask = torch.triu(torch.ones(len, len, dtype=in_features.dtype, device=in_features.device), diagonal=1)
        # rearrange(self.Wq, "head_num d_k d_in")
        Q = rearrange(self.Wq(in_features), "... sequence_length (num_heads d_head) -> ... sequence_length num_heads d_head", num_heads = self.num_heads, d_head = self.d_model // self.num_heads)
        K = rearrange(self.Wk(in_features), "... sequence_length (num_heads d_head) -> ... sequence_length num_heads d_head", num_heads = self.num_heads, d_head = self.d_model // self.num_heads)

        value = einsum(Q, K, "... sequence_length1 num_heads d_head, ... sequence_length2 num_heads d_head -> ... num_heads sequence_length1 sequence_length2") / math.sqrt(Q.shape[-1])

        normal_value = self.Softmax(value.masked_fill(causal_mask == 1, float("-inf")), -1)
        
        V = rearrange(self.Wv(in_features), "... sequence_length (num_heads d_head) -> ... sequence_length num_heads d_head", num_heads = self.num_heads, d_head = self.d_model // self.num_heads)

        val = einsum(normal_value, V, "... num_heads sequence_length sequence_length2, ... sequence_length2 num_heads d_head -> ... sequence_length num_heads d_head")
        val_head = rearrange(val, "... sequence_length num_heads d_head -> ... sequence_length (num_heads d_head)")
        return self.Wo.forward(val_head)
        