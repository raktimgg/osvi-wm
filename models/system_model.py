import math
import inspect
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.utils.modules import Block


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, window_size, n_layer, n_head, n_embed, output_dim, dropout, bias, is_causal=False):
        super().__init__()
        assert input_dim is not None
        assert window_size is not None
        self.window_size = window_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(input_dim, n_embed),
                wpe=nn.Embedding(window_size, n_embed),
                # wpe=TemporalPositionalEncoding(n_embed, 0.2),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([Block(dim=n_embed, num_heads=n_head, drop=0., attn_drop=0., is_causal=is_causal) 
                                 for _ in range(n_layer)]),
                ln_f=LayerNorm(n_embed, bias=bias),
            )
        )
        self.output_head = nn.Linear(n_embed, output_dim, bias=True)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer)
                )

        # report number of parameters
        logging.info("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, target=None):
        device = x.device
        b, t, d = x.size()
        assert (
            t <= self.window_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.window_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(x)  # token embeddings of shape (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embed)
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = self.transformer.wpe(tok_emb.transpose(1,2)).transpose(1,2)  # position embeddings of shape (t, n_embed)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        output = self.output_head(x)
        loss = None if target is None else F.mse_loss(output, target)
        if target is None:
            return output
        else:
            return output, loss

    def configure_optimizers(self, weight_decay, lr, betas, device_type=None):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logging.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logging.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        logging.info(f"using fused AdamW: {use_fused}")

        return optimizer