import copy
from tqdm import tqdm
import numpy as np
import torch


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, support_feature, query_feature, task_ids, batch_tasks, batch_samples, nway, kshot,
                 dtype="train", all_features=None):
        self.batch_tasks = batch_tasks
        self.batch_samples = batch_samples
        self.task_ids = task_ids
        self.dtype = dtype
        if nway is None:
            nway = len(task_ids[0]) * 2
        nway_for_one_stage = nway // 2
        self.nway = nway
        self.kshot = kshot
        self.nway_for_one_stage = nway_for_one_stage
        assert self.nway_for_one_stage * 2 == self.nway
        self.num_batch = int(np.ceil(len(query_feature) / batch_samples)) * int(
            np.ceil(nway_for_one_stage / batch_tasks)) * 2

        if all_features is None:
            self.support_data = self.gen_tasks(support_feature, True)
            self.query_data = self.gen_tasks(query_feature, False)
        else:
            if dtype == "train":
                self.support_data = self.gen_tasks(all_features, True)
                self.query_data = self.gen_tasks(all_features, False)
            else:
                self.support_data, self.query_data = self.gen_tasks_by_all_features(all_features)

        self.features = None
        if dtype != "train":
            self.init_test()

    def gen_tasks_by_all_features(self, features):
        # [stage1, stage2]; stage1: [task1, ...]; task1: [feature1, ...]
        support_data = [[[] for _ in self.task_ids[0]], [[] for _ in self.task_ids[1]]]
        query_data = [[[] for _ in self.task_ids[0]], [[] for _ in self.task_ids[1]]]
        min_support_number = 0
        for feature in tqdm(features):
            labels = feature.label
            label_mask = feature.label_mask
            num_stage = len(labels)
            for s_id in range(num_stage):
                measurement_ids = self.task_ids[s_id]
                for task_id, m_id in enumerate(measurement_ids):
                    if label_mask[s_id][m_id] == 1:
                        temp_feat = copy.deepcopy(feature)
                        temp_feat.label = labels[s_id][m_id]
                        temp_feat.stage_id = s_id
                        temp_feat.m_id = m_id
                        if len(support_data[s_id][task_id]) < self.kshot:
                            support_data[s_id][task_id].append(temp_feat)
                        else:
                            query_data[s_id][task_id].append(temp_feat)
        return support_data, query_data

    def gen_tasks(self, features, is_support):
        # [stage1, stage2]; stage1: [task1, ...]; task1: [feature1, ...]
        data = [[[] for _ in self.task_ids[0]], [[] for _ in self.task_ids[1]]]
        # max([len(d) for d_s in data for d in d_s ])
        # data = [[[] for _ in range(15)], [[] for _ in range(15)]]
        min_support_number = 0
        for feature in tqdm(features):
            labels = feature.label
            label_mask = feature.label_mask
            num_stage = len(labels)
            for s_id in range(num_stage):
                measurement_ids = self.task_ids[s_id]
                # measurement_ids = range(15)
                for task_id, m_id in enumerate(measurement_ids):
                    if label_mask[s_id][m_id] == 1:
                        temp_feat = copy.deepcopy(feature)
                        temp_feat.label = labels[s_id][m_id]
                        temp_feat.stage_id = s_id
                        temp_feat.m_id = m_id
                        if self.dtype != "train" and is_support:
                            if len(data[s_id][task_id]) < self.kshot:
                                data[s_id][task_id].append(temp_feat)
                            min_support_number = min([len(task_data) for s_data in data for task_data in s_data])
                            if min_support_number >= self.kshot:
                                break
                        else:
                            data[s_id][task_id].append(temp_feat)
                if min_support_number >= self.kshot:
                    break
            if min_support_number >= self.kshot:
                break
        return data

    def __len__(self):
        return self.num_batch

    def get_train_batch(self):
        batch_features = []
        for stage_id in range(2):
            support_data = self.support_data[stage_id]
            query_data = self.query_data[stage_id]
            task_ids = np.arange(len(support_data))
            choosen_task_ids = np.random.choice(task_ids, self.nway_for_one_stage)
            num_sample_per_task = self.batch_tasks // 2 // len(choosen_task_ids)
            choosen_task_id_samples = []
            for t_id in choosen_task_ids:
                cur_number = min(self.batch_tasks // 2-len(choosen_task_id_samples), num_sample_per_task)
                choosen_task_id_samples += [t_id for _ in range(cur_number)]
            if self.batch_tasks // 2 > len(choosen_task_id_samples):
                choosen_task_id_samples += [np.random.choice(choosen_task_ids, 1).tolist()[0] for _ in range(self.batch_tasks // 2 - len(choosen_task_id_samples))]
            else:
                choosen_task_id_samples = choosen_task_id_samples[:self.batch_tasks // 2]

            gen_one_train_sample_func = self.gen_one_train_sample_map(choosen_task_id_samples, support_data, query_data)
            batch_features.extend(list(map(gen_one_train_sample_func, range(self.batch_tasks // 2))))
        return batch_features

    def gen_one_train_sample_map(self, choosen_task_id_samples, support_data, query_data):
        def gen_one_train_sample(sample_id):
            cur_task_id = choosen_task_id_samples[sample_id]
            cur_support_data = support_data[cur_task_id]
            cur_query_data = query_data[cur_task_id]
            sample_ids = np.arange(len(cur_support_data))
            support_sample_ids = np.random.choice(sample_ids, self.kshot)
            sample_ids = np.array([s_id for s_id in sample_ids if s_id not in support_sample_ids])
            query_sample_ids = np.random.choice(sample_ids, self.batch_samples)
            cur_support_data = [cur_support_data[s_id] for s_id in support_sample_ids]
            cur_query_data = [cur_query_data[s_id] for s_id in query_sample_ids]
            return [cur_support_data, cur_query_data]
        return gen_one_train_sample

    def init_test(self):
        test_features = []
        for stage_id in range(2):
            support_data = self.support_data[stage_id]
            query_data = self.query_data[stage_id]
            for task_id in range(len(support_data)):
                cur_support_data = support_data[task_id]
                cur_query_data = query_data[task_id]
                for i in range(0, len(cur_query_data), self.batch_samples):
                    test_features.append([cur_support_data, cur_query_data[i:i + self.batch_samples]])
        self.features = []
        for i in range(0, len(test_features), self.batch_tasks):
            self.features.append(test_features[i:i + self.batch_tasks])
        self.num_batch = len(self.features)

    @staticmethod
    def batch_feature2tensor(batch):
        batch_tasks = len(batch)
        kshot = max([len(batch[t_id][0]) for t_id in range(len(batch))])
        batch_samples = max([len(batch[t_id][1]) for t_id in range(len(batch))])
        feature_size = batch[0][0][0].feature.size
        support_feature = torch.zeros(batch_tasks, kshot, feature_size)
        support_mask = torch.zeros(batch_tasks, kshot)
        support_label = torch.zeros(batch_tasks, kshot)
        support_label_mask = torch.zeros(batch_tasks, kshot)
        support_graph = []
        support_node_mask = torch.zeros(batch_tasks, kshot, feature_size)

        query_feature = torch.zeros(batch_tasks, batch_samples, feature_size)
        query_mask = torch.zeros(batch_tasks, batch_samples)
        query_label = torch.zeros(batch_tasks, batch_samples)
        query_label_mask = torch.zeros(batch_tasks, batch_samples)
        query_graph = []
        query_node_mask = torch.zeros(batch_tasks, batch_samples, feature_size)

        stage_mark = torch.zeros(batch_tasks, dtype=torch.long)
        measurement_mark = torch.zeros(batch_tasks, dtype=torch.long)

        for batch_task_id in range(len(batch)):
            cur_task = batch[batch_task_id]
            # comment
            cur_support_features = [f.feature for f in cur_task[0]]
            support_feature[batch_task_id, :len(cur_support_features)] = torch.tensor(cur_support_features, dtype=torch.float)
            support_mask[batch_task_id, :len(cur_support_features)] = 1
            cur_support_label = [f.label for f in cur_task[0]]
            support_label[batch_task_id, :len(cur_support_features)] = torch.tensor(cur_support_label, dtype=torch.float)
            support_label_mask[batch_task_id, :len(cur_support_features)] = 1
            cur_support_node_mask = [f.node_mask for f in cur_task[0]]
            support_node_mask[batch_task_id, :len(cur_support_features)] = torch.tensor(cur_support_node_mask, dtype=torch.float)
            support_graph += [f.graph for f in cur_task[0]]

            cur_query_features = [f.feature for f in cur_task[1]]
            query_feature[batch_task_id, :len(cur_query_features)] = torch.tensor(cur_query_features, dtype=torch.float)
            query_mask[batch_task_id, :len(cur_query_features)] = 1
            cur_query_label = [f.label for f in cur_task[1]]
            query_label[batch_task_id, :len(cur_query_features)] = torch.tensor(cur_query_label, dtype=torch.float)
            query_label_mask[batch_task_id, :len(cur_query_features)] = 1
            cur_query_node_mask = [f.node_mask for f in cur_task[1]]
            query_node_mask[batch_task_id, :len(cur_query_features)] = torch.tensor(cur_query_node_mask, dtype=torch.float)
            query_graph += [f.graph for f in cur_task[1]]

            stage_mark[batch_task_id] = cur_task[0][0].stage_id
            measurement_mark[batch_task_id] = cur_task[0][0].m_id
        return {
            "support_feature": support_feature,
            "support_mask": support_mask,
            "support_label": support_label,
            "support_label_mask": support_label_mask,
            "support_graph": support_graph,
            "support_node_mask": support_node_mask,

            "query_feature": query_feature,
            "query_mask": query_mask,
            "query_label": query_label,
            "query_label_mask": query_label_mask,
            "query_graph": query_graph,
            "query_node_mask": query_node_mask,

            "stage_mark": stage_mark,
            "measurement_mark": measurement_mark
        }

    def __iter__(self):
        if self.dtype == "train":
            for i in range(self.__len__()):
                batch_features = self.get_train_batch()
                yield self.batch_feature2tensor(batch_features)
        else:
            for i in range(self.__len__()):
                yield self.batch_feature2tensor(self.features[i])
