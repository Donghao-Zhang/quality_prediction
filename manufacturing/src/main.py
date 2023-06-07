import torch
from torch.utils.data import DataLoader
import time
import sys
import os
import re
import pickle
import numpy as np
import pandas as pd
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")
from src.config import get_config
from src.utils.utils import set_seed, print_params, lr_decay
from src.preprocess import ExampleProcessor, FeatureProcessor, collate_fn
from src.model import Model
from src.utils.optimization_utils import OPTIMIZER_CLASSES
from src.utils.utils import regression_metrics


def gen_feature(config, data_path):
    if os.path.exists(config.train_save_path) and os.path.exists(config.dev_save_path) and \
            os.path.exists(config.test_save_path) and os.path.exists(config.processer_save_path):
        print("reading samples from file")
        train_samples = pickle.load(open(config.train_save_path, "rb"))
        dev_samples = pickle.load(open(config.dev_save_path, "rb"))
        test_samples = pickle.load(open(config.test_save_path, "rb"))
        example_processer = pickle.load(open(config.processer_save_path, "rb"))
    else:
        print("processing samples")
        example_processer = ExampleProcessor(test_ratio=config.test_ratio, dev_ratio=config.dev_ratio)
        train_samples, dev_samples, test_samples = example_processer.read_samples_from_file(data_path)
        pickle.dump(train_samples, open(config.train_save_path, "wb"))
        pickle.dump(dev_samples, open(config.dev_save_path, "wb"))
        pickle.dump(test_samples, open(config.test_save_path, "wb"))
        pickle.dump(example_processer, open(config.processer_save_path, "wb"))
    if 0 < config.train_ratio < 1:
        num_train = int(len(train_samples) * config.train_ratio)
        print("sample {:.2f}% train samples".format(config.train_ratio*100))
        train_samples = train_samples[: num_train]
    train_examples, node_position, stage_position = example_processer.convert_sample_to_example(train_samples, config.normalization, config.model_type)
    dev_examples, _, _ = example_processer.convert_sample_to_example(dev_samples, config.normalization, config.model_type)
    test_examples, _, _ = example_processer.convert_sample_to_example(test_samples, config.normalization, config.model_type)
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
    best_dev_s1 = [1e9, 1e9, 1e9]
    best_dev_s2 = [1e9, 1e9, 1e9]
    start_time = time.time()
    num_total_step = len(dataloader) * config.epoch
    for epoch_idx in range(config.epoch):
        optimizer = lr_decay(optimizer, epoch_idx, config.decay_rate, config.lr)
        for batch_idx, batch in enumerate(dataloader):
            model.train()
            step += 1
            inputs = {
                "feature": batch["feature"].to(config.device),
                "node_mask": batch["node_mask"].to(config.device),
                "graph": batch["graph"],
                "stage1_label": batch["stage1_label"].to(config.device),
                "stage1_label_mask": batch["stage1_label_mask"].to(config.device),
                "stage2_label": batch["stage2_label"].to(config.device),
                "stage2_label_mask": batch["stage2_label_mask"].to(config.device),
            }
            stage1_loss, stage2_loss = model.compute_loss(**inputs)
            loss = config.loss_weight * stage1_loss + (1 - config.loss_weight) * stage2_loss
            if isinstance(loss, torch.Tensor):
                sample_loss += loss.item()
                loss.backward()
            else:
                sample_loss += loss
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
                    print("*" * 30)
                    print(("TEST: speed: %.2fst/s; rmse: %.4f, mse: %.4f, mae: %.4f" % (test_speed, test_rmse, test_mse, test_mae)))
                    print(("stage 1: rmse: %.4f, mse: %.4f, mae: %.4f" % (test_s1_rmse, test_s1_mse, test_s1_mae)))
                    print(("stage 2: rmse: %.4f, mse: %.4f, mae: %.4f" % (test_s2_rmse, test_s2_mse, test_s2_mae)))
                if best_dev[0] > rmse:
                    print("Exceed previous best rmse score:", best_dev[0])
                    best_epoch = epoch_idx + 1
                    model_save_path = os.path.join(config.model_save_dir, config.model_save_name+".pt")
                    torch.save(model.state_dict(), model_save_path)
                    best_dev = [rmse, mse, mae]
                    best_dev_s1 = [s1_rmse, s1_mse, s1_mae]
                    best_dev_s2 = [s2_rmse, s2_rmse, s2_rmse]
                print("*"*30)
    print("Best dev epoch: %.4f, score: rmse: %.4f, mse: %.4f, mae: %.4f" % (best_epoch, best_dev[0], best_dev[1], best_dev[2]))
    print(("stage 1: rmse: %.4f, mse: %.4f, mae: %.4f" % (best_dev_s1[0], best_dev_s1[1], best_dev_s1[2])))
    print(("stage 2: rmse: %.4f, mse: %.4f, mae: %.4f" % (best_dev_s2[0], best_dev_s2[1], best_dev_s2[2])))
    print("*" * 30)


def evaluate(config, model, dataloader, show_detail=False):
    model.eval()
    pred_results = []
    gold_results = []

    stage1_pred_results = []
    stage1_gold_results = []
    stage2_pred_results = []
    stage2_gold_results = []
    stage1_pred_results_np = None
    stage1_gold_results_np = None
    stage2_pred_results_np = None
    stage2_gold_results_np = None
    stage1_mask_np = None
    stage2_mask_np = None

    start_time = time.time()
    num_instances = 0
    alpha = None
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            inputs = {
                "feature": batch["feature"].to(config.device),
                "node_mask": batch["node_mask"].to(config.device),
                "graph": batch["graph"],
            }
            batch_label = torch.cat([batch["stage1_label"], batch["stage2_label"]], dim=-1)
            label_pos = np.where(torch.cat([batch["stage1_label_mask"], batch["stage2_label_mask"]], dim=-1).numpy() == 1)
            stage1_label_pos = np.where(batch["stage1_label_mask"].numpy() == 1)
            stage2_label_pos = np.where(batch["stage2_label_mask"].numpy() == 1)

            stage1_pre, stage2_pre, stage_alpha = model(**inputs)
            if stage_alpha is not None:
                if alpha is None:
                    alpha = stage_alpha.cpu().numpy()
                else:
                    alpha = np.concatenate([alpha, stage_alpha.cpu().numpy()], axis=0)
            num_instances += batch_label.size(0)

            pred_results += torch.cat([stage1_pre, stage2_pre], dim=-1).cpu().numpy()[label_pos].tolist()
            gold_results += batch_label.cpu().numpy()[label_pos].tolist()
            # pred_results += torch.cat([stage1_pre, stage2_pre], dim=-1).tolist()
            # gold_results += batch_label.tolist()

            if stage1_pred_results_np is None:
                stage1_pred_results_np = stage1_pre.cpu().numpy()
                stage1_gold_results_np = batch["stage1_label"].numpy()
                stage2_pred_results_np = stage2_pre.cpu().numpy()
                stage2_gold_results_np = batch["stage2_label"].numpy()
                stage1_mask_np = batch["stage1_label_mask"].numpy()
                stage2_mask_np = batch["stage2_label_mask"].numpy()
            else:
                stage1_pred_results_np = np.concatenate([stage1_pred_results_np, stage1_pre.cpu().numpy()], axis=0)
                stage1_gold_results_np = np.concatenate([stage1_gold_results_np, batch["stage1_label"].numpy()], axis=0)
                stage2_pred_results_np = np.concatenate([stage2_pred_results_np, stage2_pre.cpu().numpy()], axis=0)
                stage2_gold_results_np = np.concatenate([stage2_gold_results_np, batch["stage2_label"].numpy()], axis=0)
                stage1_mask_np = np.concatenate([stage1_mask_np, batch["stage1_label_mask"].numpy()], axis=0)
                stage2_mask_np = np.concatenate([stage2_mask_np, batch["stage2_label_mask"].numpy()], axis=0)

            stage1_pred_results += stage1_pre.cpu().numpy()[stage1_label_pos].tolist()
            stage1_gold_results += batch["stage1_label"].numpy()[stage1_label_pos].tolist()
            stage2_pred_results += stage2_pre.cpu().numpy()[stage2_label_pos].tolist()
            stage2_gold_results += batch["stage2_label"].numpy()[stage2_label_pos].tolist()
    decode_time = time.time() - start_time
    speed = num_instances / decode_time
    rmse, mse, mae = regression_metrics(gold_results, pred_results)

    s1_rmse, s1_mse, s1_mae = regression_metrics(stage1_gold_results, stage1_pred_results)

    s2_rmse, s2_mse, s2_mae = regression_metrics(stage2_gold_results, stage2_pred_results)
    if show_detail:
        print("=" * 30)
        print("=" * 10 + "detailed information" + "=" * 10)
        pred_results_np_list = [stage1_pred_results_np, stage2_pred_results_np]
        gold_results_np_list = [stage1_gold_results_np, stage2_gold_results_np]
        mask_np_list = [stage1_mask_np, stage2_mask_np]
        for stage_index in range(2):
            num_label = gold_results_np_list[stage_index].shape[-1]
            for label_index in range(num_label):
                label_pos = np.where(mask_np_list[stage_index][:, label_index] == 1)
                cur_gold = gold_results_np_list[stage_index][:, label_index][label_pos]
                cur_pred = pred_results_np_list[stage_index][:, label_index][label_pos]
                if config.save_results:
                    df = pd.DataFrame(np.concatenate([cur_gold[:, np.newaxis], cur_pred[:, np.newaxis]], axis=-1))
                    df.to_csv("../data/results/s{}m{}.csv".format(stage_index+1, label_index))
                cur_gold = cur_gold.tolist()
                cur_pred = cur_pred.tolist()
                cur_rmse, cur_mse, cur_mae = regression_metrics(cur_gold, cur_pred)
                print("station %d measurement %d: rmse: %.4f, mse: %.4f, mae: %.4f" %
                      (stage_index+1, label_index, cur_rmse, cur_mse, cur_mae))
        print("=" * 30)
    return speed, rmse, mse, mae, \
           s1_rmse, s1_mse, s1_mae, \
           s2_rmse, s2_mse, s2_mae


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

    train_dataloader = DataLoader(train_feature, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=False)
    dev_dataloader = DataLoader(dev_feature, batch_size=config.test_batch_size, shuffle=False, collate_fn=collate_fn,
                                drop_last=False)
    test_dataloader = DataLoader(test_feature, batch_size=config.test_batch_size, shuffle=False, collate_fn=collate_fn,
                                 drop_last=False)

    config.device = torch.device("cuda:{}".format(config.gpu)
                                 if torch.cuda.is_available() and not config.no_cuda else "cpu")

    model = Model(config, node_position, stage_position)
    model.to(config.device)
    print(model.parameters)
    print_params(model)
    if not config.no_train:
        train(config, model, train_dataloader, dev_dataloader)

    model_load_path = os.path.join(config.model_save_dir, config.model_save_name + ".pt")
    model.load_state_dict(torch.load(model_load_path))

    speed, rmse, mse, mae, s1_rmse, s1_mse, s1_mae, s2_rmse, s2_mse, s2_mae = evaluate(config, model, test_dataloader, show_detail=True)
    print(("test: speed: %.2fst/s; rmse: %.4f, mse: %.4f, mae: %.4f" %
         (speed, rmse, mse, mae)))
    print(("stage 1: rmse: %.4f, mse: %.4f, mae: %.4f" % (s1_rmse, s1_mse, s1_mae)))
    print(("stage 2: rmse: %.4f, mse: %.4f, mae: %.4f" % (s2_rmse, s2_mse, s2_mae)))


if __name__ == "__main__":
    main()
