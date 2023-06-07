import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import defaultdict
import dgl
import torch


class Example(object):
    def __init__(self, feature, node_position, label):
        self.feature = feature
        self.node_position = node_position
        self.label = label


class ExampleProcessor(object):
    def __init__(self, test_ratio=0.3, dev_ratio=0.3, train_ratio=1):
        self.test_ratio = test_ratio
        self.dev_ratio = dev_ratio
        self.train_ratio = train_ratio
        self.scaler = None

    def read_samples_from_file(self, data_path):
        df1 = pd.read_csv(data_path, delimiter=',', nrows=None)
        nRow, nCol = df1.shape
        print(f'There are {nRow} rows and {nCol} columns')
        samples = df1.values
        train_samples, dev_samples, test_samples = self.split_dataset(samples)
        return train_samples, dev_samples, test_samples

    def convert_sample_to_example(self, samples, normalization="min_max", model_type="gat"):
        examples, node_position, stage_position = self.preprocess_data(samples, normalization=normalization, model_type=model_type)
        return examples, node_position, stage_position

    def split_dataset(self, samples):
        num_samples = len(samples)
        num_test = int(num_samples * self.test_ratio)
        num_dev = int(num_samples * self.dev_ratio)
        random.shuffle(samples)
        test_samples = samples[:num_test]
        dev_samples = samples[num_test:num_test+num_dev]
        train_samples = samples[num_test+num_dev:]
        num_train = int(len(train_samples) * self.train_ratio)
        return train_samples[: num_train], dev_samples, test_samples

    def normalization(self, data, method="min_max"):
        if self.scaler is None:
            if method == "min_max":
                scaler = MinMaxScaler((-1, 1))
            else:
                scaler = StandardScaler()
            data = scaler.fit_transform(data)
        else:
            scaler = self.scaler
            data = scaler.transform(data)
        return scaler, data

    def preprocess_data(self, samples, normalization="min_max", model_type="gat"):
        node_position = [0]
        stage_position = [0]
        if model_type == "linear":
            # split into input (X) and output (Y) variables
            # [common_node, m1_node, m2_node, m3_node, s1_common_node, m4_node, m5_node]
            x_for_common = samples[:, 1:3].astype('float64')
            node_position.append(node_position[-1] + x_for_common.shape[1])
            x_for_machine1 = samples[:, 3:15].astype('float64')
            node_position.append(node_position[-1] + x_for_machine1.shape[1])
            x_for_machine2 = samples[:, 15:27].astype('float64')
            node_position.append(node_position[-1] + x_for_machine2.shape[1])
            x_for_machine3 = samples[:, 27:39].astype('float64')
            node_position.append(node_position[-1] + x_for_machine3.shape[1])
            x_for_stage1_common = samples[:, 39:42].astype('float64')
            node_position.append(node_position[-1] + x_for_stage1_common.shape[1])
            stage_position.append(len(node_position) - 1)
            x_for_machine4 = samples[:, 72:79].astype('float64')
            node_position.append(node_position[-1] + x_for_machine4.shape[1])
            x_for_machine5 = samples[:, 79:86].astype('float64')
            node_position.append(node_position[-1] + x_for_machine5.shape[1])
            stage_position.append(len(node_position) - 1)
            x = np.concatenate([x_for_common, x_for_machine1, x_for_machine2, x_for_machine3,
                                x_for_stage1_common, x_for_machine4, x_for_machine5], axis=-1)

        else:
            # [m1_node, m2_node, m3_node, s1_combiner, m4_node, m5_node]
            x_for_common = samples[:, 1:3].astype('float64')

            x_for_machine1 = np.concatenate([x_for_common, samples[:, 3:15].astype('float64')],
                                            axis=-1)
            node_position.append(node_position[-1] + x_for_machine1.shape[1])
            x_for_machine2 = np.concatenate([x_for_common, samples[:, 15:27].astype('float64')],
                                            axis=-1)
            node_position.append(node_position[-1] + x_for_machine2.shape[1])
            x_for_machine3 = np.concatenate([x_for_common, samples[:, 27:39].astype('float64')],
                                            axis=-1)
            node_position.append(node_position[-1] + x_for_machine3.shape[1])
            x_for_stage1_combiner = np.concatenate([x_for_common, samples[:, 39:42].astype('float64')],
                                                   axis=-1)
            node_position.append(node_position[-1] + x_for_stage1_combiner.shape[1])
            stage_position.append(len(node_position) - 1)

            x_for_machine4 = np.concatenate([x_for_common, samples[:, 72:79].astype('float64')], axis=-1)
            node_position.append(node_position[-1] + x_for_machine4.shape[1])
            x_for_machine5 = np.concatenate([x_for_common, samples[:, 79:86].astype('float64')], axis=-1)
            node_position.append(node_position[-1] + x_for_machine5.shape[1])
            stage_position.append(len(node_position) - 1)
            assert len(node_position) - 1 == 6
            x = np.concatenate([x_for_machine1, x_for_machine2, x_for_machine3, x_for_stage1_combiner, x_for_machine4, x_for_machine5],
                               axis=-1)
        y_for_stage1 = samples[:, 42:71:2].astype('float64')

        y_for_stage2 = samples[:, 86::2].astype('float64')

        scaler, x = self.normalization(x, normalization)
        if self.scaler is None:
            self.scaler = scaler
        num_samples = x.shape[0]
        examples = []
        for i in range(num_samples):
            examples.append(Example(x[i], node_position, [y_for_stage1[i], y_for_stage2[i]]))
        return examples, node_position, stage_position


class Feature(object):
    def __init__(self, feature, node_mask, graph, label=None, label_mask=None):
        self.feature = feature
        self.node_mask = node_mask
        self.graph = graph
        self.label = label
        self.label_mask = label_mask


class FeatureProcessor(object):
    def __init__(self, station_id=None, measurement_id=None):
        self.station_id = station_id
        self.measurement_id = measurement_id
        self.graph = None

    def convert_example_to_feature(self, examples):
        features = []
        num_samples_measurements = None
        for example in examples:
            feature = example.feature
            label = example.label
            label_np = np.array(label)
            label_mask = np.ones_like(label_np)
            if num_samples_measurements is None:
                num_samples_measurements = np.zeros_like(label_np)
            label_mask[np.where(label_np < 1e-4)] = 0
            label_mask[np.where(label_np > 1e4)] = 0
            if self.station_id is not None and self.measurement_id is not None:
                label_mask[self.station_id][:self.station_id] = 0
                label_mask[self.station_id][self.station_id+1:] = 0
                label_mask[:self.station_id] = 0
                label_mask[self.station_id+1:] = 0

            num_samples_measurements += label_mask
            label_mask = label_mask.tolist()
            node_mask = np.ones(feature.shape[0], dtype=np.float) * -1
            # [common_node, m1_node, m2_node, m3_node, s1_common_node, m4_node, m5_node]
            node_position = example.node_position
            for n_i in range(1, len(node_position)):
                node_mask[node_position[n_i-1]:node_position[n_i]] = n_i - 1
            if self.graph is None:
                graph = self.create_heterograph()
                self.graph = graph
            else:
                graph = self.graph
            features.append(Feature(feature, node_mask, graph, label, label_mask))
        return features

    @staticmethod
    def create_heterograph():
        """
        node from index 1, [m1_node, m2_node, m3_node, s1_combiner, m4_node, m5_node]
        :return:
        """
        d = defaultdict(list)
        # forward
        d[('node', 'forward_edge', 'node')].append((0, 3))
        d[('node', 'forward_edge', 'node')].append((1, 3))
        d[('node', 'forward_edge', 'node')].append((2, 3))
        d[('node', 'forward_edge', 'node')].append((3, 4))
        d[('node', 'forward_edge', 'node')].append((4, 5))

        # backward
        d[('node', 'backward_edge', 'node')].append((3, 0))
        d[('node', 'backward_edge', 'node')].append((3, 1))
        d[('node', 'backward_edge', 'node')].append((3, 2))
        d[('node', 'backward_edge', 'node')].append((4, 3))
        d[('node', 'backward_edge', 'node')].append((5, 4))
        graph = dgl.heterograph(d)
        return graph


def collate_fn(batch):
    feature = [f.feature for f in batch]
    node_mask = [f.node_mask for f in batch]
    graph = [f.graph for f in batch]
    stage1_label = [f.label[0] for f in batch]
    stage2_label = [f.label[1] for f in batch]
    stage1_label_mask = [f.label_mask[0] for f in batch]
    stage2_label_mask = [f.label_mask[1] for f in batch]

    feature = torch.tensor(feature, dtype=torch.float)
    node_mask = torch.tensor(node_mask, dtype=torch.float)
    stage1_label = torch.tensor(stage1_label, dtype=torch.float)
    stage2_label = torch.tensor(stage2_label, dtype=torch.float)
    stage1_label_mask = torch.tensor(stage1_label_mask, dtype=torch.float)
    stage2_label_mask = torch.tensor(stage2_label_mask, dtype=torch.float)
    output = {
        "feature": feature,
        "node_mask": node_mask,
        "graph": graph,
        "stage1_label": stage1_label,
        "stage2_label": stage2_label,
        "stage1_label_mask": stage1_label_mask,
        "stage2_label_mask": stage2_label_mask,
    }
    return output

