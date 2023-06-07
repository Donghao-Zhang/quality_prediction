import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="linear model 0.001, gat 0.001")
    parser.add_argument("--optim", type=str, default="radam", choices=["adamw", "sgd", "adam", "radam", "pradam"], help="linear model adamw or radam, gat radam")
    parser.add_argument("--decay_rate", type=float, default=0)

    parser.add_argument("--linear_hidden", type=int, default=512, help="for linear model")
    parser.add_argument("--linear_layer", type=int, default=4, help="for linear model, 3 or 4")

    parser.add_argument("--cnn_layer", type=int, default=4, help="for linear model, 3 or 4")

    parser.add_argument("--encoder_layers", type=int, default=2, help="for gat and gcn")
    parser.add_argument("--predictor_layers", type=int, default=2, help="for gat and gcn")
    parser.add_argument("--gcn_layers", type=int, default=3, help="for gat and gcn")
    parser.add_argument("--gcn_hidden", type=int, default=256, help="for gat and gcn")
    parser.add_argument("--num_head", type=int, default=8, help="for gat and node aggregation")
    parser.add_argument("--node_hidden", type=int, default=128, help="for gat and node aggregation")
    parser.add_argument("--agg_attention_hidden", type=int, default=32, help="for gat and node aggregation")
    parser.add_argument("--attention_layer", type=int, default=0, help="for gat and gcn")

    parser.add_argument("--use_path", action='store_true', default=False)
    parser.add_argument("--bidirectional", action='store_true', default=False)
    parser.add_argument("--path_layer", type=int, default=2, help="for gat and gcn")
    parser.add_argument("--path_attention_hidden", type=int, default=32, help="for gat and node aggregation")

    parser.add_argument("--model_type", type=str, default="gat", choices=["gat", "gcn"])
    parser.add_argument("--dropout", type=float, default=0.1, help="linear model 0.2, gat 0.1")
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "l1", "smooth_l1", "quantile"])
    parser.add_argument("--smooth_loss_beta", type=float, default=1.)
    parser.add_argument("--quantiles", type=str, default="0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98", help="0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98 // .05,.5,.95")
    parser.add_argument("--loss_weight", type=float, default=0.3)

    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--normalization", type=str, default="norm", choices=["min_max", "norm"])

    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--train_ratio", type=float, default=1)

    parser.add_argument("--no_train", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--save_results", action='store_true', default=False)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--eval_step", type=int, default=80)

    parser.add_argument("--data_path", type=str, default="../data/continuous_factory_process.csv")
    parser.add_argument("--model_save_dir", type=str, default="../data/checkpoint")
    parser.add_argument("--model_save_name", type=str, default="manufacturing_model")
    parser.add_argument("--train_save_path", type=str, default="../data/train.pkl")
    parser.add_argument("--dev_save_path", type=str, default="../data/dev.pkl")
    parser.add_argument("--test_save_path", type=str, default="../data/test.pkl")
    parser.add_argument("--processer_save_path", type=str, default="../data/processer.pkl")
    parser.add_argument("--target", type=str, default="all")

    parser.add_argument("--no_cuda", action='store_true', default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args
