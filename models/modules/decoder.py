import copy
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
        self.norm = norm if norm is not None else nn.Identity()  # a placeholder; may break backward compatibility
        # If possible use nn.Identity() instead of None as this way is more readable

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        out = self.norm(out)
        return out  # shape: (batch_size, tgt_seq_len, d_embed)


class DecoderPlaceholder(nn.Module):

    def __init__(self, decoder_block=None, n_layer=None, norm=None, *args, **kwargs):
        super(DecoderPlaceholder, self).__init__()
        # We don't store or use the provided arguments,
        # but we accept them to ensure interface compatibility.

    def forward(self, tgt, encoder_out=None, tgt_mask=None, src_tgt_mask=None, *args, **kwargs):
        return tgt  # Just returning the input tgt as is

