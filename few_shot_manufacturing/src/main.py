import torch
import json
import time
import sys
import os
import re
import numpy as np
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")
from src.config import get_config
from src.utils.utils import set_seed, print_params, lr_decay
from src.preprocess import ExampleProcessor, FeatureProcessor, collate_fn
from src.model import Model
from src.utils.optimization_utils import OPTIMIZER_CLASSES
from src.utils.utils import regression_metrics
from src.data_loader import DataLoader


def gen_feature(config, data_path):
    example_processer = ExampleProcessor(test_ratio=config.test_ratio, dev_ratio=config.dev_ratio)
    train_samples, dev_samples, test_samples = example_processer.read_samples_from_file(data_path)
    if 0 < config.train_ratio < 1:
        num_train = int(len(train_samples) * config.train_ratio)
        print("sample {:.2f}% train samples".format(config.train_ratio*100))
        train_samples = train_samples[: num_train]
    train_examples, node_position, stage_position = example_processer.convert_sample_to_example(train_samples, config.normalization)
    dev_examples, _, _ = example_processer.convert_sample_to_example(dev_samples, config.normalization)
    test_examples, _, _ = example_processer.convert_sample_to_example(test_samples, config.normalization)
    print("number of train sample: ", len(train_examples))
    print("number of dev sample: ", len(dev_examples))
    print("number of test sample: ", len(test_examples))

    feature_processer = FeatureProcessor(config.station_id, config.measurement_id)
    train_feature = feature_processer.convert_example_to_feature(train_examples)
    dev_feature = feature_processer.convert_example_to_feature(dev_examples)
    test_feature = feature_processer.convert_example_to_feature(test_examples)
    return train_feature, dev_feature, test_feature, node_position, stage_position


def train(config, model, dataloader, dev_dataloader=None, test_dataloader=None):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OPTIMIZER_CLASSES[config.optim](parameters, lr=config.lr)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    step = 0
    sample_loss = 0
    best_epoch = 0
    # rmse mse mae
    best_dev = [1e9, 1e9, 1e9]
    best_s1_dev = [1e9, 1e9, 1e9]
    best_s2_dev = [1e9, 1e9, 1e9]
    start_time = time.time()
    num_total_step = len(dataloader) * config.epoch
    for epoch_idx in range(config.epoch):
        optimizer = lr_decay(optimizer, epoch_idx, config.decay_rate, config.lr)
        for batch_idx, batch in enumerate(dataloader):
            model.train()
            step += 1
            inputs = {
                "support_feature": batch["support_feature"].to(config.device),
                "support_mask": batch["support_mask"].to(config.device),
                "support_label": batch["support_label"].to(config.device),
                "support_label_mask": batch["support_label_mask"].to(config.device),
                "support_graph": batch["support_graph"],
                "support_node_mask": batch["support_node_mask"].to(config.device),

                "query_feature": batch["query_feature"].to(config.device),
                "query_mask": batch["query_mask"].to(config.device),
                "query_label": batch["query_label"].to(config.device),
                "query_label_mask": batch["query_label_mask"].to(config.device),
                "query_graph": batch["query_graph"],
                "query_node_mask": batch["query_node_mask"].to(config.device),

                "stage_mark": batch["stage_mark"].to(config.device),
                "measurement_mark": batch['measurement_mark'].to(config.device)
            }
            loss = model.compute_loss(**inputs)
            sample_loss += loss.item()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            if step % config.log_step == 0:
                temp_cost = time.time() - start_time
                start_time = time.time()
                print(("     Epoch: %s; Step: %s; Time: %.2fs; loss: %.4f" % (epoch_idx+1, step, temp_cost, sample_loss)))
                sys.stdout.flush()
                sample_loss = 0
            if (step % config.eval_step == 0 or step == num_total_step) and dev_dataloader is not None:
                speed, rmse, mse, mae, s1_rmse, s1_mse, s1_mae, s2_rmse, s2_mse, s2_mae = evaluate(config, model, dev_dataloader)
                print("*"*30)
                print("learning rate: ", optimizer.param_groups[0]["lr"])
                print(("Dev: speed: %.2fst/s; rmse: %.4f, mse: %.4f, mae: %.4f" % (speed, rmse, mse, mae)))
                print(("stage 1: rmse: %.4f, mse: %.4f, mae: %.4f" % (s1_rmse, s1_mse, s1_mae)))
                print(("stage 2: rmse: %.4f, mse: %.4f, mae: %.4f" % (s2_rmse, s2_mse, s2_mae)))

                if test_dataloader is not None:
                    test_speed, test_rmse, test_mse, test_mae, test_s1_rmse, test_s1_mse, test_s1_mae, \
                    test_s2_rmse, test_s2_mse, test_s2_mae = evaluate(config, model, test_dataloader)
                    print("*"*30)
                    print(("TEST: speed: %.2fst/s; rmse: %.4f, mse: %.4f, mae: %.4f" %
                           (test_speed, test_rmse, test_mse, test_mae)))
                    print(("stage 1: rmse: %.4f, mse: %.4f, mae: %.4f" % (test_s1_rmse, test_s1_mse, test_s1_mae)))
                    print(("stage 2: rmse: %.4f, mse: %.4f, mae: %.4f" % (test_s2_rmse, test_s2_mse, test_s2_mae)))
                if best_dev[0] > rmse:
                    print("Exceed previous best rmse score:", best_dev[0])
                    best_epoch = epoch_idx + 1
                    model_save_path = os.path.join(config.model_save_dir, config.model_save_name+".pt")
                    torch.save(model.state_dict(), model_save_path)
                    best_dev = [rmse, mse, mae]
                    best_s1_dev = [s1_rmse, s1_mse, s1_mse]
                    best_s2_dev = [s2_rmse, s2_mse, s2_mse]
                print("*"*30)
    print("Best dev epoch: %.4f, score: rmse: %.4f, mse: %.4f, mae: %.4f" % (best_epoch, best_dev[0], best_dev[1], best_dev[2]))
    print(("stage 1: rmse: %.4f, mse: %.4f, mae: %.4f" % (best_s1_dev[0], best_s1_dev[1], best_s1_dev[2])))
    print(("stage 2: rmse: %.4f, mse: %.4f, mae: %.4f" % (best_s2_dev[0], best_s2_dev[1], best_s2_dev[2])))
    print("*" * 30)


def evaluate(config, model, dataloader, return_results=False):
    model.eval()
    pred_results_np = None
    gold_results_np = None
    result_mask_np = None
    stage_mark_np = None

    start_time = time.time()
    num_instances = 0
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            inputs = {
                "support_feature": batch["support_feature"].to(config.device),
                "support_mask": batch["support_mask"].to(config.device),
                "support_label": batch["support_label"].to(config.device),
                "support_label_mask": batch["support_label_mask"].to(config.device),
                "support_graph": batch["support_graph"],
                "support_node_mask": batch["support_node_mask"].to(config.device),

                "query_feature": batch["query_feature"].to(config.device),
                "query_mask": batch["query_mask"].to(config.device),
                "query_graph": batch["query_graph"],
                "query_node_mask": batch["query_node_mask"].to(config.device),

                "stage_mark": batch["stage_mark"].to(config.device),
                "cache": None
            }
            num_instances += inputs["query_feature"].size(0)
            predictions, cache = model(**inputs)
            if pred_results_np is None:
                pred_results_np = predictions.detach().cpu().numpy()
                gold_results_np = batch['query_label'].detach().cpu().numpy()
                result_mask_np = batch['query_label_mask'].detach().cpu().numpy()
                stage_mark_np = batch["stage_mark"].detach().cpu().numpy()
            else:
                pred_results_np = np.concatenate([pred_results_np, predictions.detach().cpu().numpy()], axis=0)
                gold_results_np = np.concatenate([gold_results_np, batch['query_label'].detach().cpu().numpy()], axis=0)
                result_mask_np = np.concatenate([result_mask_np, batch['query_label_mask'].detach().cpu().numpy()], axis=0)
                stage_mark_np = np.concatenate([stage_mark_np, batch["stage_mark"].detach().cpu().numpy()], axis=0)

    pred_results, gold_results = [], []
    stage_pred_results, stage_gold_results = [[], []], [[], []]
    for task_id in range(pred_results_np.shape[0]):
        for sample_id in range(pred_results_np[task_id].shape[0]):
            if result_mask_np[task_id][sample_id] != 0:
                cur_stage = stage_mark_np[task_id]
                cur_pred = pred_results_np[task_id][sample_id]
                cur_gold = gold_results_np[task_id][sample_id]
                pred_results.append(cur_pred)
                gold_results.append(cur_gold)
                stage_pred_results[cur_stage].append(cur_pred)
                stage_gold_results[cur_stage].append(cur_gold)

    decode_time = time.time() - start_time
    speed = num_instances / decode_time
    rmse, mse, mae = regression_metrics(gold_results, pred_results)
    s1_rmse, s1_mse, s1_mae = regression_metrics(stage_gold_results[0], stage_pred_results[0])
    s2_rmse, s2_mse, s2_mae = regression_metrics(stage_gold_results[1], stage_pred_results[1])

    if return_results:
        return speed, rmse, mse, mae, s1_rmse, s1_mse, s1_mae, s2_rmse, s2_mse, s2_mae, gold_results, pred_results
    else:
        return speed, rmse, mse, mae, s1_rmse, s1_mse, s1_mae, s2_rmse, s2_mse, s2_mae


def get_n_way_targets(n_way_train, n_way_dev, n_way_test):
    n_way_train = n_way_train // 2
    n_way_dev = n_way_dev // 2
    n_way_test = n_way_test // 2
    assert n_way_train + n_way_dev + n_way_test == 15
    stage1 = np.arange(15)
    stage2 = np.arange(15)
    return [stage1[:n_way_train], stage2[:n_way_train]], \
           [stage1[n_way_train:n_way_train+n_way_dev], stage2[n_way_train:n_way_train+n_way_dev]], \
           [stage1[n_way_train+n_way_dev:], stage2[n_way_train+n_way_dev:]]


def main():
    config = get_config()
    if config.seed >= 0:
        set_seed(config.seed)
    station_measurement = re.findall(r"s(\d+)m(\d+)", config.target)
    if len(station_measurement) == 1:
        config.station_id = int(station_measurement[0][0])-1
        config.measurement_id = int(station_measurement[0][1])
        if config.station_id == 0:
            config.loss_weight = 1
        else:
            config.loss_weight = 0
    else:
        config.station_id = None
        config.measurement_id = None

    train_feature, dev_feature, test_feature, node_position, stage_position = gen_feature(config, config.data_path)
    config.label_size_s1 = train_feature[0].label[0].shape[0]
    config.label_size_s2 = train_feature[0].label[1].shape[0]

    train_task_ids, dev_task_ids, test_task_ids = get_n_way_targets((15-config.nway*2)*2, config.nway*2, config.nway*2)
    all_features = train_feature + dev_feature + test_feature

    label_max = [np.ones([15]) * -np.inf, np.ones([15]) * -np.inf]
    label_min = [np.ones([15]) * np.inf, np.ones([15]) * np.inf]
    for f in all_features:
        cur_label = f.label
        for stage in range(2):
            mask = np.array(f.label_mask[stage])
            label_max[stage] = np.max(np.stack([label_max[stage], np.where(mask == 1, cur_label[stage], -np.inf)], axis=0), axis=0)
            label_min[stage] = np.min(np.stack([label_min[stage], np.where(mask == 1, cur_label[stage], np.inf)], axis=0), axis=0)

    train_dataloader = DataLoader(support_feature=train_feature, query_feature=train_feature,
                                  task_ids=train_task_ids, batch_tasks=config.batch_tasks, batch_samples=config.batch_samples,
                                  nway=len(dev_task_ids[0])*2, kshot=config.kshot, dtype="train", all_features=all_features)
    dev_dataloader = DataLoader(support_feature=train_feature, query_feature=dev_feature,
                                task_ids=dev_task_ids, batch_tasks=config.batch_tasks, batch_samples=config.batch_samples,
                                nway=None, kshot=config.kshot, dtype="test", all_features=all_features)
    test_dataloader = DataLoader(support_feature=train_feature, query_feature=test_feature,
                                 task_ids=test_task_ids, batch_tasks=config.batch_tasks, batch_samples=config.batch_samples,
                                 nway=None, kshot=config.kshot, dtype="test", all_features=all_features)
    print("=" * 15 + "comment" + "=" * 15)
    print("COMMENT: " + config.comment)
    print("=" * 30)
    print("=" * 15 + "config" + "=" * 15)
    config_json = json.dumps(vars(config), indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
    print(config_json)

    config.device = torch.device("cuda:{}".format(config.gpu)
                                 if torch.cuda.is_available() and not config.no_cuda else "cpu")

    model = Model(config, node_position, stage_position)
    model.to(config.device)
    print("="*30)
    print("="*15 + "parameters" + "="*15)
    print(model.parameters)
    print_params(model)
    print("="*30)
    if not config.no_train:
        train(config, model, train_dataloader, dev_dataloader, test_dataloader)

    load_param = True
    if load_param:
        model_load_path = os.path.join(config.model_save_dir, config.model_save_name + ".pt")
        model.load_state_dict(torch.load(model_load_path))
        print("load model: ", model_load_path)

    print("test model")
    speed, rmse, mse, mae, s1_rmse, s1_mse, s1_mae, \
    s2_rmse, s2_mse, s2_mae, gold_results, pred_results = evaluate(config, model, test_dataloader, return_results=True)
    print("test speed: %.4f, score: rmse: %.4f, mse: %.4f, mae: %.4f" % (speed, rmse, mse, mae))
    print(("stage 1: rmse: %.4f, mse: %.4f, mae: %.4f" % (s1_rmse, s1_mse, s1_mae)))
    print(("stage 2: rmse: %.4f, mse: %.4f, mae: %.4f" % (s2_rmse, s2_mse, s2_mae)))


if __name__ == "__main__":
    # plot_test()
    main()
