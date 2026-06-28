"""Minimal CALMFormer runner.

Examples:
    python run_calmformer.py
    python run_calmformer.py --k_shot 5
    python run_calmformer.py --epochs 2 --n_episodes 20
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CALMFormer ORACLE mainline.")
    parser.add_argument("--dataset", default="ORACLE")
    parser.add_argument("--n_classes", type=int, default=16)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", default="checkpoints_oracle_calmformer")
    parser.add_argument("--split_seed", type=int, default=2027)
    parser.add_argument("--train_seed", type=int, default=2027)
    parser.add_argument("--episode_seed", type=int, default=9001)

    # The following defaults are the manuscript mainline.
    parser.add_argument("--mixer", default="rf_ilcm_anchor")
    parser.add_argument("--rf_anchor_init_eta", type=float, default=0.7)
    parser.add_argument("--rf_aug_recipe", default="full")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--base_ratio", type=float, default=0.7)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=48)
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--conv_kernel_size", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def fill_train_eval_defaults(args):
    args.name_suffix = ""
    args.pgr_weight = 0.0
    args.pgr_margin = 0.2
    args.pgr_compact_weight = 1.0
    args.pgr_margin_weight = 1.0
    args.pgr_warmup_epochs = 0
    args.rf_aug = False
    args.aug_phase = 0.0
    args.aug_amp = 0.0
    args.aug_awgn_min = 20.0
    args.aug_awgn = 0.0
    args.aug_shift = 0
    args.aug_cfo = 0.0
    args.aug_iq_gain = 0.0
    args.aug_iq_phase = 0.0
    args.skip_eval = False
    args.save_checkpoint = False
    return args


if __name__ == "__main__":
    args = fill_train_eval_defaults(parse_args())
    import torch
    from train_eval import train

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train(args, device)
