import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
            Also see https://arxiv.org/abs/1606.08415
        """
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MLP(nn.Module):
    """
    Multi-layer perceptron
    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, bidirectional, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        '''
        src: [batch_size, slen, input_size]
        mask: [batch_size, slen]
        '''
        if mask is None:
            mask = torch.ones([src.size(0), src.size(1)]).to(src.device)
        src_lengths = torch.sum(mask, dim=1)
        zero_index = None
        if torch.any(src_lengths == 0):
            # zero_index = torch.nonzero(src_lengths == 0).squeeze(-1)
            zero_index = torch.nonzero(src_lengths.eq(0), as_tuple=False).squeeze(-1)
            src = src.index_fill_(0, zero_index, 0)
            src_lengths = src_lengths.index_fill_(0, zero_index, 1)
        # self.lstm.flatten_parameters()
        bsz, slen, input_size = src.size()
        new_src_lengths, sort_index = torch.sort(src_lengths, dim=-1, descending=True)
        new_src = torch.index_select(src, dim=0, index=sort_index)

        packed_src = nn.utils.rnn.pack_padded_sequence(new_src, new_src_lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)
        # packed_outputs, src_h_t = self.lstm(packed_src)
        # src_c_t = src_h_t

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, padding_value=0)

        unsort_index = torch.argsort(sort_index)
        outputs = torch.index_select(outputs, dim=0, index=unsort_index)
        if self.bidirectional:
            src_h_t = src_h_t.view(self.num_layers, 2, bsz, self.hidden_dim)
            src_c_t = src_c_t.view(self.num_layers, 2, bsz, self.hidden_dim)
            output_h_t = torch.cat((src_h_t[-1, 0], src_h_t[-1, 1]), dim=-1)
            output_c_t = torch.cat((src_c_t[-1, 0], src_c_t[-1, 1]), dim=-1)
        else:
            src_h_t = src_h_t.view(self.num_layers, 1, bsz, self.hidden_dim)
            src_c_t = src_c_t.view(self.num_layers, 1, bsz, self.hidden_dim)
            output_h_t = src_h_t[-1, 0]
            output_c_t = src_c_t[-1, 0]
        output_h_t = torch.index_select(output_h_t, dim=0, index=unsort_index)
        output_c_t = torch.index_select(output_c_t, dim=0, index=unsort_index)

        outputs = self.out_dropout(outputs)
        output_h_t = self.out_dropout(output_h_t)
        output_c_t = self.out_dropout(output_c_t)
        if zero_index is not None:
            outputs = outputs.index_fill_(0, zero_index, 0)
            output_h_t = output_h_t.index_fill_(0, zero_index, 0)
            output_c_t = output_c_t.index_fill_(0, zero_index, 0)
        return outputs, (output_h_t, output_c_t)


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, in_feat, hidden_feat, label_size, dropout=0):
        super(SelfAttentiveEncoder, self).__init__()
        self.in_feat = in_feat
        self.hidden_feat = hidden_feat
        self.label_size = label_size
        # self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(in_feat, hidden_feat)
        self.ws2 = nn.Linear(hidden_feat, label_size)

    def forward(self, inp, mask=None, predictors=None):

        alphas = self.ws2(torch.tanh(self.ws1(inp)))
        alphas = alphas.permute(0, 2, 1)
        if mask is not None:
            alphas = alphas.masked_fill_(mask.eq(0).unsqueeze(-1).permute(0, 2, 1), -1e8)
        alphas = torch.softmax(alphas, dim=2)
        temp_out = torch.bmm(alphas, inp)
        # temp_out = self.drop(temp_out)

        if predictors is None:
            return temp_out, alphas
        outs = []
        for l_index in range(self.label_size):
            out = temp_out[:, l_index]
            out = predictors[l_index](out)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        return outs, alphas


class MatrixVectorScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)
        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask.eq(0), -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class AttPool(nn.Module):
    def __init__(self,  in_feat, hidden_feat, label_size, dropout=0.1):
        super().__init__()
        self.label_size = label_size
        # self.w_qs = nn.Linear(in_feat, in_feat * label_size)
        self.w_q1 = nn.Linear(in_feat, hidden_feat)
        self.w_q2 = nn.Linear(hidden_feat, in_feat * label_size)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (in_feat + in_feat * label_size)))
        nn.init.normal_(self.w_q1.weight, mean=0, std=np.sqrt(2.0 / (in_feat + in_feat * label_size)))
        nn.init.normal_(self.w_q2.weight, mean=0, std=np.sqrt(2.0 / (in_feat + in_feat * label_size)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (in_feat + in_feat * label_size)))
        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(in_feat * label_size, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, predictors=None, mask=None):
        """
        q: tensor of shape (b, d_q)
        k: tensor of shape (b, l, d_k)
        returns: tensor of shape (b, d_k)
        """
        batch_size, d_q = q.size()
        _, l, d_k = k.size()
        # qs = self.w_qs(q).reshape(batch_size, self.label_size, d_k)  # (b, d_k)
        qs = self.w_q2(self.w_q1(q)).reshape(batch_size, self.label_size, d_k)  # (b, d_k)

        outs = []
        attns = []
        for l_index in range(self.label_size):
            out, attn = self.attention(qs[:, l_index, :], k, k, mask=mask)
            out = self.dropout(out)
            if predictors is not None:
                out = predictors[l_index](out)
            outs.append(out)
            attns.append(attn)
        outs = torch.stack(outs, dim=1)
        attns = torch.stack(attns, dim=1)
        return outs, attns


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttentioGCN(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentioGCN, self).__init__()
        num_heads = 1
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.gcn = GraphConvLayerAGGCN(embed_dim, embed_dim, dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout).squeeze(1)
        outputs = self.gcn(attn, value)
        return outputs.permute(1, 0, 2), attn


class GraphConvLayerAGGCN(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, in_dim, out_dim, dropout=0.):
        super(GraphConvLayerAGGCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gcn_drop = nn.Dropout(dropout)

        # gcn layer
        self.Linear = nn.Linear(in_dim, out_dim)
        self.weight = nn.Linear(in_dim, in_dim)

    def forward(self, adj, gcn_inputs):
        denom = adj.sum(2).unsqueeze(2) + 1
        Ax = adj.bmm(gcn_inputs)
        AxW = self.weight(Ax)
        AxW = AxW + self.weight(gcn_inputs)  # self loop
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        gcn_ouputs = self.gcn_drop(gAxW)
        gcn_ouputs = gcn_ouputs + gcn_inputs
        out = self.Linear(gcn_ouputs)
        return out


class SmoothL1Loss(nn.Module):
    def __init__(self, reduction="none", beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction="none")
        self.l2 = nn.MSELoss(reduction="none")

    def forward(self, pred, gold):
        l1_loss = self.l1(pred, gold)
        l2_loss = self.l2(pred, gold)
        if self.beta < 1e-5:
            loss = l1_loss
        else:
            loss = torch.where(l1_loss < self.beta, 0.5 * l2_loss / self.beta, l1_loss - 0.5 * self.beta)
        return loss
