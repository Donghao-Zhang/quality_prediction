import torch
import torch.nn as nn
from src.layers import MLP


class WeightGenerator(nn.Module):
    def __init__(self, feature_hidden, few_shot_module_hidden, task_label_hidden, stage_label_hidden, weight_dim,
                 dropout):
        super(WeightGenerator, self).__init__()
        self.pre_project = MLP(feature_hidden + 1 + stage_label_hidden + task_label_hidden,
                               few_shot_module_hidden, few_shot_module_hidden, 0, dropout, activation="relu")
        self.self_att = nn.MultiheadAttention(few_shot_module_hidden, 4, dropout=dropout)
        self.feature_project = MLP(feature_hidden, few_shot_module_hidden, few_shot_module_hidden, 1, dropout,
                                   activation="relu")
        self.query_att = nn.MultiheadAttention(few_shot_module_hidden, 4, dropout=dropout,
                                               kdim=few_shot_module_hidden, vdim=few_shot_module_hidden)
        self.post_project = MLP(few_shot_module_hidden, weight_dim+1, weight_dim+1, 0, dropout, activation="relu")

    def forward(self, query_feature, query_mask,
                support_feature, support_label, support_label_mask, support_mask, task_label_feature, stage_label_feature):
        kshot = support_feature.size(1)
        hidden = torch.cat([support_feature, support_label.unsqueeze(-1),
                            task_label_feature.unsqueeze(1).expand(-1, kshot, -1),
                            stage_label_feature.unsqueeze(1).expand(-1, kshot, -1)], dim=-1)
        hidden = self.pre_project(hidden)
        hidden, alpha = self.self_att(query=hidden.permute(1, 0, 2), key=hidden.permute(1, 0, 2),
                                      value=hidden.permute(1, 0, 2), key_padding_mask=support_label_mask.eq(0))
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.masked_fill_(support_label_mask.unsqueeze(-1).eq(0), 0)

        hidden_att, alpha = self.query_att(self.feature_project(query_feature).permute(1, 0, 2),
                                           self.feature_project(support_feature).permute(1, 0, 2),
                                           hidden.permute(1, 0, 2), key_padding_mask=support_label_mask.eq(0))
        hidden_att = hidden_att.permute(1, 0, 2)

        hidden = self.post_project(hidden_att)
        hidden = hidden.masked_fill_(query_mask.unsqueeze(-1).eq(0), 0)
        return hidden
