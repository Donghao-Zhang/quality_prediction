import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np

from src.layers import MLP, BiLSTM, SelfAttentiveEncoder, AttPool, MultiHeadAttentioGCN, SmoothL1Loss
from src.utils.loss import QuantileLoss


class Model(nn.Module):
    def __init__(self, config, node_position, stage_position):
        super(Model, self).__init__()
        self.node_position = node_position
        assert len(stage_position) == 3
        self.stage_position = stage_position
        self.model_type = config.model_type
        self.dropout = nn.Dropout(config.dropout)
        self.label_size_s1 = config.label_size_s1
        self.label_size_s2 = config.label_size_s2
        self.config = config

        if config.use_path:
            encoder_output = config.gcn_hidden // 2
        else:
            encoder_output = config.gcn_hidden
        # encoder_output = config.gcn_hidden
        self.encoders = nn.ModuleList([
            nn.Sequential(
                MLP(node_position[n_idx] - node_position[n_idx - 1], encoder_output // 4, encoder_output,
                    config.encoder_layers - 1, config.dropout, activation="relu"),
                nn.ReLU(),
                # GELU(),
                nn.Dropout(config.dropout),
            ) for n_idx in range(1, len(node_position))])
        if config.use_path:
            self.path_encoder = BiLSTM(
                input_dim=config.gcn_hidden // 2,
                hidden_dim=(config.gcn_hidden // 4 if config.bidirectional else config.gcn_hidden // 2),
                num_layers=config.path_layer, dropout=config.dropout, bidirectional=config.bidirectional,
                batch_first=True)
            self.path_attention = SelfAttentiveEncoder(config.gcn_hidden // 2, config.path_attention_hidden, 1,
                                                        dropout=config.dropout)

        self.gcn_layers = config.gcn_layers
        self.rel_name_lists = ["forward_edge", "backward_edge"]
        self.node_extractor = nn.ModuleList()
        for i in range(self.gcn_layers):
            if i == 0:
                self.node_extractor.append(
                    RelGraphConvLayer(config.gcn_hidden, config.gcn_hidden, self.rel_name_lists,
                                      num_bases=len(self.rel_name_lists), activation=nn.ReLU(),
                                      self_loop=True, dropout=config.dropout,
                                      model_type=self.model_type, num_head=config.num_head))
            else:
                self.node_extractor.append(
                    RelGraphConvLayer(config.gcn_hidden, config.gcn_hidden,
                                      self.rel_name_lists, num_bases=len(self.rel_name_lists), activation=nn.ReLU(),
                                      self_loop=True, dropout=config.dropout,
                                      model_type=self.model_type, num_head=config.num_head))
        self.attention_layer = config.attention_layer
        self.structure_infer_layers = nn.ModuleList([
            MultiHeadAttentioGCN(embed_dim=config.gcn_hidden, num_heads=config.num_head, dropout=config.dropout)
            for _ in range(config.attention_layer)
        ])
        self.node_hidden = config.gcn_hidden
        self.node_pool_s1 = AttPool(self.node_hidden, config.agg_attention_hidden, self.label_size_s1, dropout=config.dropout)
        self.node_pool_s2 = AttPool(self.node_hidden, config.agg_attention_hidden, self.label_size_s2, dropout=config.dropout)
        if config.loss_type == "quantile":
            quantiles = config.quantiles.split(",")
            quantiles = [float(q.strip()) for q in quantiles]
            self.output_dim = len(quantiles)
        else:
            self.output_dim = 1

        self.predictors_s1 = nn.ModuleList([
            MLP(self.node_hidden, self.node_hidden // 4, self.output_dim, config.predictor_layers - 1, dropout=config.dropout,
                activation="relu") for _ in range(self.label_size_s1)
        ])
        self.predictors_s2 = nn.ModuleList([
            MLP(self.node_hidden, self.node_hidden // 4, self.output_dim, config.predictor_layers - 1, dropout=config.dropout,
                activation="relu") for _ in range(self.label_size_s2)
        ])

        if config.loss_type == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        elif config.loss_type == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif config.loss_type == "smooth_l1":
            self.criterion = SmoothL1Loss(reduction="none", beta=config.smooth_loss_beta)
        elif config.loss_type == "quantile":
            self.criterion = QuantileLoss(quantiles)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def getNodeFeature(self, encoder_outputs, node_id, node_start_index, node_end_index):
        batch_size = encoder_outputs.size(0)
        return encoder_outputs[(node_start_index <= node_id) & (node_id < node_end_index)].reshape(batch_size, -1)

    def encode(self, feature, node_mask):
        node_rep = []
        for node_idx in range(1, len(self.node_position)):
            node_rep.append(
                self.encoders[node_idx - 1](feature[:, self.node_position[node_idx - 1]: self.node_position[node_idx]]))
        node_rep = torch.stack(node_rep, dim=1)
        return node_rep

    def path_encode(self, node_rep):
        batch_size, num_node, hidden_dim = node_rep.size()
        path_index, path_mask = self.get_path()
        num_node, max_num_paths, max_path_length = path_index.size()
        # batch_size * num_node * max_num_paths * max_path_length
        path_index = path_index.to(node_rep.device).unsqueeze(0).expand(batch_size, -1, -1, -1)
        path_index = path_index.reshape(batch_size, -1).unsqueeze(-1).expand(-1, -1, hidden_dim)
        path_mask = path_mask.to(node_rep.device).unsqueeze(0).expand(batch_size, -1, -1, -1)
        path_rep = torch.gather(node_rep, 1, path_index).masked_fill_(path_mask.reshape(batch_size, -1).unsqueeze(-1).eq(0), 0)
        path_rep = path_rep.reshape(batch_size, num_node, max_num_paths, max_path_length, hidden_dim).reshape(-1, max_path_length, hidden_dim)
        outputs, (output_h_t, output_c_t) = self.path_encoder(path_rep, path_mask.reshape(-1, max_path_length))
        output_h_t = output_h_t.reshape(batch_size * num_node, max_num_paths, -1)
        output_h_t, output_alpha = self.path_attention(output_h_t, mask=path_mask.sum(-1).gt(0).reshape(batch_size * num_node, max_num_paths), predictors=None)
        output_h_t = self.dropout(output_h_t.squeeze(1).reshape(batch_size, num_node, -1))
        output_h_t = torch.cat([node_rep, output_h_t], dim=-1)
        return output_h_t

    @staticmethod
    def get_path():
        path = [[[0]],
                [[1]],
                [[2]],
                [[0, 3], [1, 3], [2, 3]],
                [[0, 3, 4], [1, 3, 4], [2, 3, 4]],
                [[0, 3, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]]]
        num_node = len(path)
        max_num_paths = 0
        max_path_length = 0
        for node in path:
            max_num_paths = max(max_num_paths, len(node))
            for node_p in node:
                max_path_length = max(max_path_length, len(node_p))
        path_index = torch.zeros([num_node, max_num_paths, max_path_length], dtype=torch.long)
        path_mask = torch.zeros([num_node, max_num_paths, max_path_length], dtype=torch.float)
        for n_id in range(len(path)):
            for p_id in range(len(path[n_id])):
                path_index[n_id, p_id][:len(path[n_id][p_id])] = torch.from_numpy(np.array(path[n_id][p_id]))
                path_mask[n_id, p_id][:len(path[n_id][p_id])] = 1
        return path_index, path_mask

    def node_interaction(self, node_emb, graphs):
        batch_size = node_emb.size(0)
        num_node = node_emb.size(1)
        graph_big = dgl.batch(graphs).to(node_emb.device)
        features = node_emb.reshape(-1, node_emb.size(-1))
        for layer in self.node_extractor:
            features = layer(graph_big, {"node": features})['node']
        return features.reshape(batch_size, num_node, -1), graph_big

    def structure_infer(self, node_emb):
        output = node_emb.permute(1, 0, 2)
        for l_index in range(self.attention_layer):
            output, o_att = self.structure_infer_layers[l_index](output, output, output)
        return output.permute(1, 0, 2)

    def get_predict(self, feature, node_mask, graph):
        node_emb = self.encode(feature, node_mask)
        if self.config.use_path:
            node_emb = self.path_encode(node_emb)

        node_emb, graph_big = self.node_interaction(node_emb, graph)
        node_emb = self.structure_infer(node_emb)

        stage1_rep = node_emb[:, self.stage_position[0]: self.stage_position[-2]]
        stage1_rep, stage1_alpha = self.node_pool_s1(stage1_rep[:, -1], stage1_rep, predictors=self.predictors_s1)
        stage1_label = stage1_rep.squeeze(-1)

        stage2_rep = node_emb[:, :self.stage_position[-1]]
        stage2_rep, stage2_alpha = self.node_pool_s2(stage2_rep[:, -1], stage2_rep, predictors=self.predictors_s2)
        stage2_label = stage2_rep.squeeze(-1)

        stage1_alpha = F.pad(stage1_alpha, (0, self.stage_position[-1]-self.stage_position[-2]-self.stage_position[0]), "constant", 0)
        stage_alpha = torch.cat([stage1_alpha, stage2_alpha], dim=1)
        return stage1_label, stage2_label, stage_alpha

    def compute_loss(self, feature, node_mask, graph, stage1_label, stage1_label_mask, stage2_label, stage2_label_mask):
        stage1_pre, stage2_pre, _ = self.get_predict(feature, node_mask, graph)
        stage1_loss = self.criterion(stage1_pre, stage1_label)
        if torch.sum(stage1_label_mask) == 0:
            stage1_loss = 0
        else:
            stage1_loss = torch.sum(stage1_loss.masked_fill_(stage1_label_mask.eq(0), 0)) / torch.sum(stage1_label_mask)
        stage2_loss = self.criterion(stage2_pre, stage2_label)
        if torch.sum(stage2_label_mask) == 0:
            stage2_loss = 0
        else:
            stage2_loss = torch.sum(stage2_loss.masked_fill_(stage2_label_mask.eq(0), 0)) / torch.sum(stage2_label_mask)
        return stage1_loss, stage2_loss

    def forward(self, feature, node_mask, graph):
        stage1_pre, stage2_pre, stage_alpha = self.get_predict(feature, node_mask, graph)
        if self.output_dim != 1:
            mean_position = int(self.output_dim // 2)
            stage1_pre = stage1_pre[:, :, mean_position]
            stage2_pre = stage2_pre[:, :, mean_position]
        return stage1_pre, stage2_pre, stage_alpha


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0,
                 model_type="gcn",
                 num_head=2):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.model_type = model_type

        if model_type == "gat":
            self.conv = dglnn.HeteroGraphConv({
                rel: dglnn.GATConv(in_feat, out_feat // num_head, num_heads=num_head)
                for rel in rel_names
            })
        else:
            self.conv = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight and self.model_type == "gcn":
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        # graph.edata
        hs = self.conv(g, inputs, mod_kwargs=wdict)
        if self.model_type == "gat":
            for node_name in hs.keys():
                hs[node_name] = hs[node_name].reshape(hs[node_name].size(0), -1)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
