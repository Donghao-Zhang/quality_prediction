# PGAT

This repository contains code and models for the paper: [Path enhanced bidirectional graph attention network for quality prediction in multistage manufacturing process](https://ieeexplore.ieee.org/abstract/document/9420277).

## Dataset

The [MCMP dataset](https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process?select=continuous_factory_process.csv) is released by [Liveline Technologies](https://www.liveline.tech/) on [Kaggle](https://www.kaggle.com/). This dataset was collected from an
actual production line near Detroit, MI, USA. The production line is a high-speed, continuous manufacturing process with parallel and series stages so that the dataset is a proper one to demonstrate the validity of PGAT. The goal of this dataset is to predict certain properties of the line’s output from the various input data. 

## Requirements

```
python==3.7
pytorch==1.6
dgl-cu101==0.5.3
```

## Train and Test

train script

```shell
cd src
python main.py --use_path --bidirectional --path_layer 1 --path_attention_hidden 32 --gcn_layers 2 --lr 0.0015 --epoch 500 --seed 0 --gcn_hidden 512 --loss_type quantile --model_save_name 0
```

test script

```shell
cd src
python main.py --use_path --bidirectional --path_layer 1 --path_attention_hidden 32 --gcn_layers 2 --lr 0.0015 --epoch 500 --seed 0 --gcn_hidden 512 --loss_type quantile --model_save_name best --no_train
```

## Citation

If you found this list useful, please consider citing this paper:

```
@article{zhang2021path,
  title={Path enhanced bidirectional graph attention network for quality prediction in multistage manufacturing process},
  author={Zhang, Donghao and Liu, Zhenyu and Jia, Weiqiang and Liu, Hui and Tan, Jianrong},
  journal={IEEE Transactions on Industrial Informatics},
  volume={18},
  number={2},
  pages={1018--1027},
  year={2021},
  publisher={IEEE}
}
```

