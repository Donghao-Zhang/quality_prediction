import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import MLP


class TaskLabelGenerator(nn.Module):
    def __init__(self, config, feature_hidden, few_shot_module_hidden, task_label_hidden, dropout):
        super(TaskLabelGenerator, self).__init__()
        self.config = config
        self.pre_project = MLP(feature_hidden+1, few_shot_module_hidden, few_shot_module_hidden, 0, dropout, activation="relu")
        self.att = nn.MultiheadAttention(few_shot_module_hidden, 4, dropout=dropout)
        self.post_project = MLP(few_shot_module_hidden, task_label_hidden, task_label_hidden, 0, dropout, activation="relu")

    def forward(self, support_feature, support_label, support_label_mask):
        task_hidden = torch.cat([support_feature, support_label.unsqueeze(-1)], dim=-1)
        task_hidden = self.pre_project(task_hidden)
        task_hidden, alpha = self.att(query=task_hidden.permute(1, 0, 2), key=task_hidden.permute(1, 0, 2),
                                      value=task_hidden.permute(1, 0, 2), key_padding_mask=support_label_mask.eq(0))
        task_hidden = self.post_project(task_hidden.permute(1, 0, 2))
        task_hidden = task_hidden.masked_fill_(support_label_mask.unsqueeze(-1).eq(0), 0)
        task_hidden = torch.sum(task_hidden, dim=1) / (torch.sum(support_label_mask.unsqueeze(-1), dim=1) + 1e-12)
        return F.normalize(task_hidden, dim=1)

