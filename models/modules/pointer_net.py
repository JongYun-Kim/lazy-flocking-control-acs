import copy
import math
import torch
import torch.nn as nn


class RawAttentionScoreGenerator(nn.Module):
    """
    Single-head attention and outputs raw attention scores
    """
    def __init__(self, d_model, q_fc, k_fc, dr_rate=0):
        super(RawAttentionScoreGenerator, self).__init__()
        self.d_model = d_model
        # W^Q, W^K; learnable params
        self.q_fc = copy.deepcopy(q_fc)  # (d_embed_query, d_model)
        self.k_fc = copy.deepcopy(k_fc)  # (d_embed_key,   d_model)
        assert self.q_fc.out_features == self.k_fc.out_features, "query and key must be transformed to a same dimension"
        assert self.q_fc.out_features == self.d_model, "query and key must be transformed to d_model dimension"
        self.dropout = nn.Dropout(p=dr_rate)  # Dropout layer; not used here

    def calculate_attention(self, query, key, mask):
        # query:  (n_batch, seq_len_query, d_model) - Batch of query vectors
        # key:    (n_batch, seq_len_key,   d_model) - Batch of key vectors
        # mask:   (n_batch, seq_len_query, seq_len_key) - Mask tensor

        d_k = key.shape[-1]  # Get the last dimension of the key
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Calculate the dot product: (Q x K^T)
        attention_score = attention_score / math.sqrt(d_k)  # Scale the attention scores
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)  # Apply the mask to the attention scores
        return attention_score  # (n_batch, seq_len_query, seq_len_key)

    def forward(self, input_query, input_key, mask=None):
        # input_query: (n_batch, seq_len_query, d_embed_query) - Batch of query vectors
        # input_key:   (n_batch, seq_len_key,   d_embed_key) - Batch of key vectors
        # mask:        (n_batch, seq_len_query, seq_len_key) - Mask tensor

        n_batch = input_query.size(0)  # Get the batch size

        # Apply the linear transformations to the query and key with no head splitting (single-head)
        query = self.q_fc(input_query)  # (n_batch, seq_len_query, d_model)
        key = self.k_fc(input_key)      # (n_batch, seq_len_key,   d_model)

        raw_attention_score = self.calculate_attention(query, key, mask)  # (n_batch, seq_len_query, seq_len_key)

        return raw_attention_score  # (n_batch, seq_len_query, seq_len_key) - The attention probabilities


class RawAttentionScoreGeneratorPlaceholder(nn.Module):
    def __init__(self):
        super(RawAttentionScoreGeneratorPlaceholder, self).__init__()

    def forward(self, input_query, input_key, mask=None):
        # Assuming that the desired output shape is the same as that of the original module
        batch_size = input_query.size(0)
        seq_len_query = input_query.size(1)
        seq_len_key = input_key.size(1)

        # Create a tensor of zeros with the same shape as the original output
        out = torch.zeros(batch_size, seq_len_query, seq_len_key, device=input_query.device)

        return out
