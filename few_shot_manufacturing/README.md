# CDG

This repository contains code and models for the paper: [Contrastive Decoder Generator for Few-shot Learning in Product Quality Prediction](https://ieeexplore.ieee.org/abstract/document/9829252/).

## Dataset

The [MCMP dataset](https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process?select=continuous_factory_process.csv) is released by [Liveline Technologies](https://www.liveline.tech/) on [Kaggle](https://www.kaggle.com/). This dataset was collected from an actual production line near Detroit, MI, USA. The production line is a high-speed, continuous manufacturing process with parallel and series stages so that the dataset is a proper one to demonstrate the validity of PGAT. The goal of this dataset is to predict certain properties of the line’s output from the various input data. 



We sample the instances from the last $N/2$ measurements of each stage as the test set. Instances from the first $15-N$ measurements of each stage are selected as the training set. The remaining instances from the middle $N/2$ measurements of each stage are selected as the validation set. In addition, the first $K$ instances for each measurement from the validation set and test set are chosen as the corresponding support set, and the remaining instances are chosen as the corresponding query set, which forms an $N$-way $K$-shot quality prediction task.

## Requirements

```
python==3.7
pytorch==1.6
dgl-cu101==0.5.3
```

## Train and Test

**train script**  for 8-way 100-shot quality prediction task

```shell
cd src
python main.py --use_path --bidirectional --gcn_layers 2 --model_save_name test_k100 --gcn_hidden 256 --few_shot_module_hidden 128 --task_label_hidden 32 --stage_label_hidden 16 --weight_penalty_l1 1e-5 --weight_penalty_l2 1e-4 --contrastive_loss_alpha 1e-2 --lr 0.001 --epoch 100 --comment '' --seed 0 --kshot 100 --nway 4 --dropout 0.2 --loss_type smooth_l1
```

**train script**  for 8-way 50-shot quality prediction task

```shell
cd src
python main.py --use_path --bidirectional --gcn_layers 2 --model_save_name test_k50 --gcn_hidden 256 --few_shot_module_hidden 128 --task_label_hidden 32 --stage_label_hidden 16 --weight_penalty_l1 1e-5 --weight_penalty_l2 1e-4 --contrastive_loss_alpha 1e-2 --lr 0.001 --epoch 100 --comment '' --seed 0 --kshot 50 --nway 4 --dropout 0.2 --loss_type smooth_l1
```

**train script**  for 8-way 100-shot quality prediction task

```shell
cd src
python main.py --use_path --bidirectional --gcn_layers 2 --model_save_name test_k20 --gcn_hidden 256 --few_shot_module_hidden 128 --task_label_hidden 32 --stage_label_hidden 16 --weight_penalty_l1 1e-5 --weight_penalty_l2 1e-4 --contrastive_loss_alpha 1e-2 --lr 0.001 --epoch 100 --comment '' --seed 0 --kshot 20 --nway 4 --dropout 0.2 --loss_type smooth_l1
```

**test script** for 8-way 100-shot quality prediction task

```shell
cd src
python main.py --use_path --bidirectional --gcn_layers 2 --model_save_name best_k100 --gcn_hidden 256 --few_shot_module_hidden 128 --task_label_hidden 32 --stage_label_hidden 16 --weight_penalty_l1 1e-5 --weight_penalty_l2 1e-4 --contrastive_loss_alpha 1e-2 --lr 0.001 --epoch 100 --comment '' --seed 0 --kshot 100 --nway 4 --dropout 0.2 --loss_type smooth_l1 --no_train
```

**test script** for 8-way 50-shot quality prediction task

```shell
cd src
python main.py --use_path --bidirectional --gcn_layers 2 --model_save_name best_k50 --gcn_hidden 256 --few_shot_module_hidden 128 --task_label_hidden 32 --stage_label_hidden 16 --weight_penalty_l1 1e-5 --weight_penalty_l2 1e-4 --contrastive_loss_alpha 1e-2 --lr 0.001 --epoch 100 --comment '' --seed 0 --kshot 50 --nway 4 --dropout 0.2 --loss_type smooth_l1 --no_train
```

**test script** for 8-way 20-shot quality prediction task

```shell
cd src
python main.py --use_path --bidirectional --gcn_layers 2 --model_save_name best_k20 --gcn_hidden 256 --few_shot_module_hidden 128 --task_label_hidden 32 --stage_label_hidden 16 --weight_penalty_l1 1e-5 --weight_penalty_l2 1e-4 --contrastive_loss_alpha 1e-2 --lr 0.001 --epoch 100 --comment '' --seed 0 --kshot 20 --nway 4 --dropout 0.2 --loss_type smooth_l1 --no_train
```

## Citation

If you found this code useful, please consider citing this paper:

```
@article{zhang2022contrastive,
  title={Contrastive Decoder Generator for Few-shot Learning in Product Quality Prediction},
  author={Zhang, Donghao and Liu, Zhenyu and Jia, Weiqiang and Liu, Hui and Tan, Jianrong},
  journal={IEEE Transactions on Industrial Informatics},
  year={2022},
  publisher={IEEE}
}
```

