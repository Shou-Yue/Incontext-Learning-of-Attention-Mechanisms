import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd = 256, n_layer = 12, n_head = 8):
        super().__init__()
        config = GPT2Config(
            n_positions = 2 * n_positions,
            n_embd = n_embd,
            n_layer = n_layer,
            n_head = n_head,
            resid_pdrop = 0.0,
            embd_pdrop = 0.0,
            attn_pdrop = 0.0,
            use_cache = False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims

        self.read_in = nn.Linear(n_dims, n_embd)
        self.backbone = GPT2Model(config)
        self.read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleaves x's and y's into a single sequence

        Args:
            xs_b: [B, T, D] input vectors
            ys_b: [B, T] corresponding scalar outputs

        Returns:
            zs: [B, 2T, D]  where positions are x_1, y_1, x_2, y_2, ...
        """
        bsize, points, dim = xs_b.shape

        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device = ys_b.device),
            ),
            dim = 2,
        )

        zs = torch.stack((xs_b, ys_b_wide), dim = 2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys):
        """
        Returns predictions for all y positions

        Args:
            xs: [B, T, D] input vectors
            ys: [B, T] ground-truth outputs

        Returns:
            pred: [B, T]
        """
        zs = self._combine(xs, ys)
        embeds = self.read_in(zs)
        output = self.backbone(inputs_embeds = embeds).last_hidden_state
        prediction = self.read_out(output)
        prediction = prediction[:, ::2, 0]
        return prediction
