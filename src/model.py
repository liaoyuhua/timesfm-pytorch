from typing import List, Tuple
import einops
import torch
from torch import nn
import torch.nn.functional as F
from .utils import masked_mean_std, shift_padded_seq
from .constants import *


class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-6):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

        self.scale = nn.Parameter(torch.full((self.dim,), 1.0))

    def forward(self, inputs):
        var = torch.mean(torch.square(inputs), dim=[-1], keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs *= self.scale
        return normed_inputs


NORM_MAP = {
    "ln": nn.LayerNorm,
    "rms": RMSNorm,
}


class FeedForward(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        dropout: float = 0.0,
        bias: bool = True,
        activation: nn.Module = nn.Identity,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x: torch.Tensor):
        return self.dropout(self.activation(self.linear(x)))


class PositionalEmbedding(nn.Module):
    """
    Generates position embedding for a given 1-d sequence.
    """

    def __init__(
        self,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
        embedding_dims: int = 0,
    ):
        """
        Args:
        min_timescale: Start of the geometric index. Determines the periodicity of
            the added signal.
        max_timescale: End of the geometric index. Determines the frequency of the
            added signal.
        embedding_dims: Dimension of the embedding to be generated.
        """
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.embedding_dims = embedding_dims

    def forward(
        self, seq_length: int | None = None, position: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Generates a Tensor of sinusoids with different frequencies.

        Args:
            seq_length: an optional Python int definiing the output sequence length.
            if the `position` argument is specified.
            position: [B, seq_length], optional position for each token in the
            sequence, only required when the sequence is packed.

        Returns:
            [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
        """
        if position is None:
            assert seq_length is not None
            # [1, seqlen]
            position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
        else:
            assert position.ndim == 2, position.shape

        num_timescales = self.embedding_dims // 2
        log_timescale_increment = torch.log(
            torch.tensor(float(self.max_timescale) / float(self.min_timescale))
        ) / torch.max(torch.tensor(num_timescales - 1), torch.tensor(1))
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        scaled_time = position[:, :, None] * inv_timescales[None, None, :]
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        # Force usage of `np` rather than `jnp` to compute static values at trace
        # time.
        signal = F.pad(signal, (0, self.embedding_dims % 2))
        return signal


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dims: int = 0,
        hidden_dims: int = 0,
        output_dims: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        norm: nn.Module = nn.LayerNorm,
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        self.ln = norm(output_dims) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.hidden_layer = FeedForward(
            input_dims, hidden_dims, 0, activation=activation
        )
        self.output_layer = FeedForward(hidden_dims, output_dims, 0)
        self.residual_layer = FeedForward(input_dims, output_dims, 0)

    def forward(self, x: torch.Tensor):
        hidden = self.hidden_layer(x)
        output = self.output_layer(hidden)
        output = self.dropout(output)
        residual = self.residual_layer(x)

        return self.ln(output + residual)


class PerDimScale(nn.Module):
    def __init__(self, model_dims: int, num_heads: int):
        super().__init__()
        self.model_dims = model_dims
        self.num_heads = num_heads
        self.head_dim = model_dims // num_heads

        if self.head_dim * num_heads != model_dims:
            raise ValueError(
                f"Model dimensions {model_dims} must be divisible by the number of heads {num_heads}"
            )

        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def forward(self, x: torch.Tensor):
        inputs_shape = x.shape
        assert inputs_shape[-1] == self.head_dim

        scale = 1.442695041 / torch.sqrt(torch.tensor(self.head_dim))
        scale = scale * F.softplus(self.per_dim_scale)
        return x * scale


class AttentionProjection(nn.Module):
    def __init__(
        self,
        input_dim: int = 0,
        num_heads: int = 0,
        dim_per_head: int = 0,
        is_output_projection: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.is_output_projection = is_output_projection

        if is_output_projection:
            self.w = nn.Parameter(torch.randn(input_dim, num_heads, dim_per_head))

            if use_bias:
                self.b = nn.Parameter(
                    torch.zeros(
                        input_dim,
                    )
                )
            else:
                self.b = None

        else:
            self.w = nn.Parameter(torch.randn(input_dim, num_heads, dim_per_head))

            if use_bias:
                self.b = nn.Parameter(torch.zeros(num_heads, dim_per_head))
            else:
                self.b = None

    def forward(self, x: torch.Tensor):
        if self.is_output_projection:
            x = torch.einsum("btnh,dnh->btd", x, self.w)
            if self.b is not None:
                x = x + self.b
        else:
            x = torch.einsum("btd,dnh->btnh", x, self.w)
            if self.b is not None:
                x = x + self.b

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dims: int = MODEL_DIMS,
        num_heads: int = NUM_HEADS,
        atttn_pdrop: float = 0.0,
    ):
        super().__init__()
        self.model_dims = model_dims
        self.num_heads = num_heads
        self.head_dim = model_dims // num_heads

        if self.head_dim * num_heads != model_dims:
            raise ValueError(
                f"Model dimensions {model_dims} must be divisible by the number of heads {num_heads}"
            )

        self.per_dim_scale = PerDimScale(model_dims, num_heads)

        self.atttn_dropout = nn.Dropout(atttn_pdrop)

        self.query = AttentionProjection(model_dims, num_heads, self.head_dim, False)
        self.key = AttentionProjection(model_dims, num_heads, self.head_dim, False)
        self.value = AttentionProjection(model_dims, num_heads, self.head_dim, False)

        self.post = AttentionProjection(model_dims, num_heads, self.head_dim, True)

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        q: [B, T, N, H]
        k: [B, S, N, H]
        v: [B, S, N, H]
        attention_mask: [B, T, S]
        """
        # scale q
        q = self.per_dim_scale(q)

        # Compute the dot product between the query and key tensors.
        # [B, T, S]
        dot_product = torch.einsum("btnh,bsnh->bnts", q, k)

        # Apply the attention mask.
        dot_product = torch.where(
            attention_mask == 0, torch.full_like(dot_product, -1e9), dot_product
        )

        # Compute the attention weights.
        attention_weights = F.softmax(dot_product, dim=-1)

        # Apply dropout to the attention weights.
        attention_weights = self.atttn_dropout(attention_weights)

        # Compute the weighted sum of the values.
        # [B, T, N, H]
        attention_output = torch.einsum("bnts,bsnh->btnh", attention_weights, v)

        return attention_output, attention_weights

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Queries, keys, and values are all the same tensor.

        input_tensor: [B, T, D]
        attention_mask: [B, T, S]
        """
        B, T, D = input_tensor.shape

        q_proj = self.query(input_tensor)
        k_proj = self.key(input_tensor)
        v_proj = self.value(input_tensor)

        attention_output, _ = self._attn(q_proj, k_proj, v_proj, attention_mask)

        attention_output = self.post(attention_output)

        return attention_output


class TransformerFeedForward(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        output_dims: int,
        bias: bool = True,
        residual_dropout_prob: float = 0.0,
        pre_norm: bool = True,
        layer_norm: str = "ln",
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.pre_norm = pre_norm
        self.layer_norm = NORM_MAP[layer_norm](output_dims)

        self.ffn_layer1 = FeedForward(
            input_dims, hidden_dims, dropout=0.0, bias=bias, activation=activation
        )
        self.ffn_layer2 = FeedForward(
            hidden_dims, output_dims, dropout=0.0, bias=bias, activation=nn.Identity
        )
        self.residual_dropout = nn.Dropout(residual_dropout_prob)

    def forward(self, x: torch.Tensor):
        res = x

        if self.pre_norm:
            x = self.layer_norm(x)

        x = self.ffn_layer1(x)
        x = self.ffn_layer2(x)

        if not self.pre_norm:
            x = self.layer_norm(x)

        return res + self.residual_dropout(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        hidden_dims: int,
        dropout: float = 0.0,
        trf_layer_norm: str = "rms",
        trf_pre_norm: bool = True,
        ffn_layer_norm: str = "ln",
        ffn_pre_norm: bool = True,
        ffn_activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.trf_pre_norm = trf_pre_norm

        self.self_attention = MultiHeadAttention(model_dims, num_heads)

        self.ff_layer = TransformerFeedForward(
            model_dims,
            hidden_dims,
            model_dims,
            residual_dropout_prob=dropout,
            pre_norm=ffn_pre_norm,
            layer_norm=ffn_layer_norm,
            activation=ffn_activation,
        )

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = NORM_MAP[trf_layer_norm](model_dims)

    def _padding_to_attention_causal_mask(
        self, input_tensor: torch.Tensor, padding: torch.Tensor
    ):
        """
        Create a causal mask from padding tensor.

        Args:
            input_tensor: [B, T, D]
            padding: [B, T]

        Returns:
            [B, T, T]
        """
        B, T = input_tensor.shape[:2]

        causal_mask = torch.tril(torch.ones(T, T, device=input_tensor.device))
        causal_mask = causal_mask.unsqueeze(0).expand(B, T, T)

        padding_mask = padding.unsqueeze(1) * padding.unsqueeze(2)
        attention_mask = causal_mask * padding_mask

        return attention_mask

    def forward(self, x: torch.Tensor, padding: torch.Tensor):
        res = x

        if self.trf_pre_norm:
            x = self.layer_norm(x)

        mask = self._padding_to_attention_causal_mask(x, padding)

        attn = self.self_attention(x, mask)

        if not self.trf_pre_norm:
            attn = self.layer_norm(attn)
        attn = self.dropout(attn)
        attn = res + attn
        output = self.ff_layer(attn)

        return output


class TimesFM(nn.Module):
    def __init__(
        self,
        context_len: int,
        horizon_len: int,
        model_dims: int = MODEL_DIMS,
        input_patch_len: int = INPUT_PATCH_LEN,
        output_patch_len: int = OUTPUT_PATCH_LEN,
        max_len: int = MAX_LEN,
        hidden_dims: int = MODEL_DIMS,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        quantiles: List[float] = DEFAULT_QUANTILES,
        use_freq: bool = USE_FREQ,
    ):
        super().__init__()

        self.num_outputs = len(quantiles) + 1
        self.use_freq = use_freq
        self.patch_len = input_patch_len
        self.context_len = context_len
        self.horizon_len = horizon_len

        self.stacked_transformer_layer = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    model_dims,
                    num_heads,
                    hidden_dims,
                    dropout=0.0,
                    trf_layer_norm="rms",
                    trf_pre_norm=True,
                    ffn_layer_norm="ln",
                    ffn_pre_norm=True,
                    ffn_activation=nn.ReLU,
                )
                for _ in range(num_layers)
            ]
        )

        self.model_dims = model_dims
        self.hidden_dims = hidden_dims
        if output_patch_len is None:
            self.output_patch_len = self.horizon_len
        else:
            self.output_patch_len = output_patch_len

        self.max_len = max_len

        self.num_decode_patches = (
            self.horizon_len + self.output_patch_len - 1
        ) // self.output_patch_len

        ff_in_dims = 2 * input_patch_len
        self.input_ff_layer = ResidualBlock(ff_in_dims, hidden_dims, model_dims)
        self.horizon_ff_layer = ResidualBlock(
            model_dims, hidden_dims, output_patch_len * self.num_outputs
        )

        self.freq_emb = nn.Embedding(3, model_dims)

        self.position_emb = PositionalEmbedding(embedding_dims=model_dims)

    def _forward_transform(self, inputs: torch.Tensor, patched_pads: torch.Tensor):
        """Input is of shape [B, N, P]."""
        mu, sigma = masked_mean_std(inputs, patched_pads)
        sigma = torch.where(sigma < _TOLERANCE, torch.tensor(1.0), sigma)
        # Normalize each patch.
        outputs = (inputs - mu.unsqueeze(1).unsqueeze(1)) / sigma.unsqueeze(
            1
        ).unsqueeze(1)
        outputs = torch.where(
            torch.abs(inputs - PAD_VAL) < _TOLERANCE, PAD_VAL, outputs
        )
        return outputs, (mu, sigma)

    def _preprocess_input(self, x, paddings):
        patched_inputs = einops.rearrange(x, "b (c p) -> b c p", p=self.patch_len)
        input_padding = torch.where(
            torch.abs(x - PAD_VAL) < _TOLERANCE, torch.tensor(1), paddings
        )
        patched_pads = einops.rearrange(
            input_padding, "b (c p) -> b c p", p=self.patch_len
        )
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        model_input = self.input_ff_layer(concat_inputs)
        # A patch should not be padded even if there is at least one zero.
        patched_padding = torch.min(patched_pads, dim=-1).values

        position_emb = self.position_emb(seq_length=model_input.shape[1])

        if position_emb.shape[0] != model_input.shape[0]:
            position_emb = torch.repeat_interleave(
                position_emb, model_input.shape[0], dim=0
            )
        position_emb = shift_padded_seq(patched_padding, position_emb)
        model_input += position_emb

        return model_input, patched_padding, stats, patched_inputs

    def _reverse_transform(
        self, outputs: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Output is of shape [B, N, P, Q]."""
        mu, sigma = stats
        return outputs * sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1) + mu.unsqueeze(
            1
        ).unsqueeze(1).unsqueeze(1)

    def _postprocess_output(
        self,
        y_hat: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Postprocess output of stacked transformer.
        """
        # B x N x (H.Q)
        output_ts = self.horizon_ff_layer(y_hat)
        output_ts = einops.rearrange(
            output_ts,
            " b n (h q) -> b n h q",
            q=self.num_outputs,
            h=self.output_patch_len,
        )

        return self._reverse_transform(output_ts, stats)

    def forward(
        self,
        x: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.Tensor,
        **kwargs,
    ):
        model_input, patched_padding, stats, _ = self._preprocess_input(
            x=x,
            paddings=paddings,
        )

        if self.use_freq:
            freq = freq.int()
            f_emb = self.freq_emb(freq)  # B x 1 x D
            f_emb = torch.repeat_interleave(f_emb, model_input.shape[1], dim=1)
            model_input += f_emb

        for layer in self.stacked_transformer_layer:
            model_input = layer(model_input, patched_padding)

        output_ts = self._postprocess_output(model_input, stats)
        return output_ts

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.Tensor = None,
        **kwargs,
    ):
        B, C = x.shape

        if not self.use_freq:
            freq = torch.zeros((B, 1), dtype=torch.int32, device=x.device)

        final_out = x
        full_outputs = []
        if paddings.size(1) != C + self.horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {paddings.shape[1]} != {C} + {self.horizon_len}"
            )

        for _ in range(self.num_decode_patches):
            current_padding = paddings[:, 0:C]
            input_ts = x[:, -self.max_len :]
            input_padding = current_padding[:, -self.max_len :]

            fprop_outputs = self(input_ts, input_padding, freq)

            new_ts = fprop_outputs[:, -1, : self.output_patch_len, 0]

            full_outputs.append(fprop_outputs[:, -1, : self.output_patch_len, :])
            final_out = torch.concatenate([final_out, new_ts], axis=-1)

        return (
            final_out[:, C : C + self.horizon_len],
            torch.concatenate(full_outputs, axis=1)[:, 0 : self.horizon_len, :],
        )

    @classmethod
    def from_pretrained(cls, **kwargs):
        from huggingface_hub import hf_hub_download

        REPO_ID = "yuhua/timesfm-1.0-200m-pytorch"
        FILENAME = "timesfm_pytorch.pth"

        weights = torch.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))

        model = cls(**kwargs).load_state_dict(weights)

        return model
