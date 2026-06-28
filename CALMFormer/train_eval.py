"""Minimal CALMFormer training and few-shot evaluation."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import SEIDataset
from fewshot_sampler import CrossSplitFewShotEpisodeSampler
from rf_metaformer import MIXER_TYPES, RFMetaFormer


def set_seed(seed=2027):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_seed(value, fallback):
    return fallback if value is None else value


def get_signal_length(dataset_name):
    from dataset import DATASET_CONFIG

    if dataset_name in DATASET_CONFIG:
        return DATASET_CONFIG[dataset_name]["signal_length"]
    return 6000


def configure_rf_aug(args):
    if args.rf_aug_recipe == "none":
        args.rf_aug = False
        return

    args.rf_aug = True
    if args.rf_aug_recipe == "lite":
        args.aug_phase = np.pi
        args.aug_amp = 3.0
        args.aug_awgn_min = 20.0
        args.aug_awgn = 35.0
        args.aug_shift = max(args.aug_shift, 16)
        args.aug_cfo = 0.0
        args.aug_iq_gain = 0.0
        args.aug_iq_phase = 0.0
    elif args.rf_aug_recipe == "full":
        args.aug_phase = np.pi
        args.aug_amp = 3.0
        args.aug_awgn_min = 15.0
        args.aug_awgn = 35.0
        args.aug_shift = max(args.aug_shift, 16)
        args.aug_cfo = 1e-4
        args.aug_iq_gain = 0.5
        args.aug_iq_phase = np.deg2rad(2.0)
    else:
        raise ValueError(f"Unknown rf_aug_recipe: {args.rf_aug_recipe}")


def prototype_classify(support_x, support_y, query_x, n_way):
    """Classify normalized query features by nearest class prototype."""
    support_x = F.normalize(support_x, dim=1)
    query_x = F.normalize(query_x, dim=1)
    prototypes = torch.stack([support_x[support_y == c].mean(0) for c in range(n_way)])
    dists = torch.cdist(query_x, prototypes)
    return dists.argmin(dim=1)


def prototype_geometry_regularization(
    feat,
    labels,
    margin=0.2,
    compact_weight=1.0,
    margin_weight=1.0,
):
    """Optional batch-level prototype compactness and margin regularization."""
    z = F.normalize(feat, dim=1)
    classes, inverse, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    if classes.numel() < 2:
        zero = z.new_tensor(0.0)
        return zero, zero, zero

    counts_f = counts.to(z.dtype).unsqueeze(1)
    prototypes = z.new_zeros(classes.numel(), z.size(1))
    prototypes.index_add_(0, inverse, z)
    prototypes = prototypes / counts_f.clamp_min(1.0)

    class_sums = prototypes * counts_f
    own_counts = counts[inverse].to(z.dtype).unsqueeze(1)
    leave_one_out = (class_sums[inverse] - z) / (own_counts - 1.0).clamp_min(1.0)
    own_proto = torch.where(own_counts > 1.0, leave_one_out, prototypes[inverse])

    d_pos = (z - own_proto).pow(2).sum(dim=1)
    dists = torch.cdist(z, prototypes, p=2).pow(2)
    same_class_mask = F.one_hot(inverse, num_classes=classes.numel()).bool()
    d_neg = dists.masked_fill(same_class_mask, float("inf")).min(dim=1).values

    compact_loss = d_pos.mean()
    margin_loss = F.relu(margin + d_pos - d_neg).mean()
    total_loss = compact_weight * compact_loss + margin_weight * margin_loss
    return total_loss, compact_loss, margin_loss


def apply_rf_augmentation(x, args):
    """Apply RF-domain nuisance augmentation during base-class training."""
    if not args.rf_aug:
        return x

    b, _, length = x.shape
    y = x

    if args.aug_shift > 0:
        max_shift = min(int(args.aug_shift), max(0, length - 1))
        if max_shift > 0:
            shifts = torch.randint(-max_shift, max_shift + 1, (b,), device=x.device)
            y = torch.stack([
                torch.roll(sample, int(shift.item()), dims=-1)
                for sample, shift in zip(y, shifts)
            ])

    if args.aug_amp > 0:
        amp_db = torch.empty(b, 1, 1, device=x.device, dtype=x.dtype).uniform_(
            -float(args.aug_amp), float(args.aug_amp)
        )
        amp = torch.pow(torch.tensor(10.0, device=x.device, dtype=x.dtype), amp_db / 20.0)
        y = y * amp

    if args.aug_phase > 0:
        phase = torch.empty(b, 1, device=x.device, dtype=x.dtype).uniform_(
            -float(args.aug_phase), float(args.aug_phase)
        )
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        i = y[:, 0, :].clone()
        q = y[:, 1, :].clone()
        y = torch.stack([i * cos_p - q * sin_p, i * sin_p + q * cos_p], dim=1)

    if args.aug_cfo > 0:
        cfo = torch.empty(b, 1, device=x.device, dtype=x.dtype).uniform_(
            -float(args.aug_cfo), float(args.aug_cfo)
        )
        n = torch.arange(length, device=x.device, dtype=x.dtype).unsqueeze(0)
        phase = 2.0 * torch.pi * cfo * n
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        i = y[:, 0, :].clone()
        q = y[:, 1, :].clone()
        y = torch.stack([i * cos_p - q * sin_p, i * sin_p + q * cos_p], dim=1)

    if args.aug_iq_gain > 0 or args.aug_iq_phase > 0:
        gain_db = torch.empty(b, 1, device=x.device, dtype=x.dtype).uniform_(
            -float(args.aug_iq_gain), float(args.aug_iq_gain)
        )
        gain = torch.pow(torch.tensor(10.0, device=x.device, dtype=x.dtype), gain_db / 20.0)
        phase = torch.empty(b, 1, device=x.device, dtype=x.dtype).uniform_(
            -float(args.aug_iq_phase), float(args.aug_iq_phase)
        )
        q_gain = y[:, 1, :] * gain
        y = torch.stack(
            [y[:, 0, :], q_gain * torch.cos(phase) + y[:, 0, :] * torch.sin(phase)],
            dim=1,
        )

    if args.aug_awgn > 0:
        snr_db = torch.empty(b, 1, 1, device=x.device, dtype=x.dtype).uniform_(
            float(args.aug_awgn_min), float(args.aug_awgn)
        )
        power = y.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        noise_power = power / torch.pow(
            torch.tensor(10.0, device=x.device, dtype=x.dtype), snr_db / 10.0
        )
        y = y + torch.randn_like(y) * torch.sqrt(noise_power)

    return y


def eval_few_shot(encoder, sampler, n_way, n_episodes, device):
    """Run few-shot episodes and return mean accuracy with a 95% CI."""
    encoder.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            sx, sy, qx, qy, _ = sampler.sample_episode()
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)

            sf = encoder(sx)
            qf = encoder(qx)
            preds = prototype_classify(sf, sy, qf, n_way)
            accs.append((preds == qy).float().mean().item())

    mean = float(np.mean(accs))
    ci95 = float(1.96 * np.std(accs) / np.sqrt(n_episodes))
    return mean, ci95


def get_anchor_eta_values(encoder):
    """Return RF residual release coefficients for anchored RF-ILCM blocks."""
    values = []
    for block in getattr(encoder, "blocks", []):
        mixer = getattr(block, "mixer", None)
        eta_logit = getattr(mixer, "eta_logit", None)
        if eta_logit is not None:
            values.append(float(torch.sigmoid(eta_logit.detach()).cpu().item()))
    return values


def make_memory_dataset(x, y, dataset_name):
    ds = SEIDataset.__new__(SEIDataset)
    ds.X = x
    ds.Y = y
    ds.n_classes = len(set(y.tolist()))
    ds.dataset_name = dataset_name
    return ds


def train(args, device):
    args.split_seed = resolve_seed(args.split_seed, args.seed)
    args.train_seed = resolve_seed(args.train_seed, args.seed)
    args.episode_seed = resolve_seed(args.episode_seed, args.seed)
    configure_rf_aug(args)
    set_seed(args.train_seed)

    signal_length = get_signal_length(args.dataset)
    full_train = SEIDataset(args.dataset, "train", args.n_classes, normalize=True)

    all_classes = sorted(set(full_train.Y.tolist()))
    rng = np.random.default_rng(args.split_seed)
    all_classes = rng.permutation(all_classes).tolist()
    n_base = int(len(all_classes) * args.base_ratio)
    base_classes = sorted(all_classes[:n_base])
    novel_classes = sorted(all_classes[n_base:])

    if len(novel_classes) == 0:
        raise ValueError("No novel classes left. Please reduce --base_ratio.")
    if args.n_way > len(novel_classes):
        print(
            f"Warning: n_way={args.n_way} exceeds novel classes={len(novel_classes)}. "
            f"Using n_way={len(novel_classes)}."
        )

    print(f"Total classes: {len(all_classes)} | Base: {len(base_classes)} | Novel: {len(novel_classes)}")
    print(f"Base classes: {base_classes}")
    print(f"Novel classes: {novel_classes}")
    print(f"Mixer: {args.mixer} | Dataset: {args.dataset} | Signal length: {signal_length}")
    print(f"Seeds: split={args.split_seed} | train={args.train_seed} | episode={args.episode_seed}")
    if args.rf_aug:
        print(
            "RF augmentation: "
            f"recipe={args.rf_aug_recipe} | phase={args.aug_phase:g} rad | "
            f"amp=+/-{args.aug_amp:g} dB | awgn=[{args.aug_awgn_min:g},{args.aug_awgn:g}] dB | "
            f"shift=+/-{args.aug_shift} | cfo=+/-{args.aug_cfo:g} | "
            f"iq_gain=+/-{args.aug_iq_gain:g} dB | iq_phase=+/-{args.aug_iq_phase:g} rad"
        )

    base_mask = np.isin(full_train.Y, base_classes)
    base_x = full_train.X[base_mask]
    base_y_raw = full_train.Y[base_mask]
    base_label_map = {c: i for i, c in enumerate(base_classes)}
    base_y = np.array([base_label_map[y] for y in base_y_raw], dtype=np.int64)

    base_dataset = torch.utils.data.TensorDataset(torch.from_numpy(base_x), torch.from_numpy(base_y))
    class_counts = np.bincount(base_y, minlength=len(base_classes))
    sample_weights = 1.0 / class_counts[base_y]
    train_generator = torch.Generator()
    train_generator.manual_seed(args.train_seed)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=train_generator,
    )
    train_loader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        generator=train_generator,
    )

    train_novel_mask = np.isin(full_train.Y, novel_classes)
    novel_train_x = full_train.X[train_novel_mask]
    novel_train_y_raw = full_train.Y[train_novel_mask]

    test_ds = SEIDataset(args.dataset, "test", args.n_classes, normalize=True)
    test_novel_mask = np.isin(test_ds.Y, novel_classes)
    novel_test_x = test_ds.X[test_novel_mask]
    novel_test_y_raw = test_ds.Y[test_novel_mask]

    novel_label_map = {c: i for i, c in enumerate(novel_classes)}
    novel_train_y = np.array([novel_label_map[y] for y in novel_train_y_raw], dtype=np.int64)
    novel_test_y = np.array([novel_label_map[y] for y in novel_test_y_raw], dtype=np.int64)
    novel_train_ds = make_memory_dataset(novel_train_x, novel_train_y, args.dataset)
    novel_test_ds = make_memory_dataset(novel_test_x, novel_test_y, args.dataset)

    n_way = min(args.n_way, len(novel_classes))

    def make_eval_sampler():
        return CrossSplitFewShotEpisodeSampler(
            support_dataset=novel_train_ds,
            query_dataset=novel_test_ds,
            n_way=n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            seed=args.episode_seed,
        )

    encoder = RFMetaFormer(
        mixer_type=args.mixer,
        dim=args.dim,
        depth=args.depth,
        patch_size=args.patch_size,
        signal_length=signal_length,
        mlp_ratio=4,
        drop=args.drop,
        conv_kernel_size=args.conv_kernel_size,
        rf_anchor_init_eta=args.rf_anchor_init_eta,
    ).to(device)
    classifier = nn.Linear(args.dim, len(base_classes)).to(device)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder params: {n_params / 1e6:.2f}M")

    optimizer = AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    method_tag = f"{args.mixer}{args.name_suffix}"
    if args.rf_aug:
        method_tag = f"{method_tag}_rfaug_{args.rf_aug_recipe}"
    save_path = save_dir / (
        f"{args.dataset}_{args.n_classes}Class_{args.n_way}way_{args.k_shot}shot_"
        f"{method_tag}_split{args.split_seed}_train{args.train_seed}_episode{args.episode_seed}.pt"
    )

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_pgr_loss = 0.0
        total_correct = 0
        total_n = 0
        pgr_scale = 0.0
        if args.pgr_weight > 0:
            pgr_scale = 1.0 if args.pgr_warmup_epochs <= 0 else min(1.0, epoch / args.pgr_warmup_epochs)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = apply_rf_augmentation(x, args)
            feat = encoder(x)
            logits = classifier(feat)
            ce_loss = F.cross_entropy(logits, y)
            pgr_loss = feat.new_tensor(0.0)
            if args.pgr_weight > 0 and pgr_scale > 0:
                pgr_loss, _, _ = prototype_geometry_regularization(
                    feat,
                    y,
                    margin=args.pgr_margin,
                    compact_weight=args.pgr_compact_weight,
                    margin_weight=args.pgr_margin_weight,
                )
            loss = ce_loss + args.pgr_weight * pgr_scale * pgr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_ce_loss += ce_loss.item() * len(y)
            total_pgr_loss += pgr_loss.item() * len(y)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += len(y)

        scheduler.step()
        avg_loss = total_loss / total_n
        train_acc = total_correct / total_n
        loss_msg = f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}"
        if args.pgr_weight > 0:
            loss_msg = (
                f"Loss: {avg_loss:.4f} | CE: {total_ce_loss / total_n:.4f} | "
                f"PGR: {total_pgr_loss / total_n:.4f} | Train Acc: {train_acc:.4f}"
            )

        if not args.skip_eval and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            fs_acc, fs_ci = eval_few_shot(encoder, make_eval_sampler(), n_way, args.n_episodes, device)
            print(
                f"Epoch {epoch:3d}/{args.epochs} | {loss_msg} | "
                f"{n_way}-way {args.k_shot}-shot: {fs_acc * 100:.2f}% +/- {fs_ci * 100:.2f}%"
            )
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | {loss_msg}")

    eta_values = get_anchor_eta_values(encoder)
    if eta_values:
        eta_text = ", ".join(f"{v:.4f}" for v in eta_values)
        print(f"RF residual release eta by block: [{eta_text}] | mean={np.mean(eta_values):.4f}")

    if args.skip_eval:
        if args.save_checkpoint:
            torch.save(encoder.state_dict(), save_path)
            print(f"Final checkpoint saved to {save_path}")
        print("\nFinal few-shot evaluation skipped by --skip_eval.")
        return None

    final_acc, final_ci = eval_few_shot(encoder, make_eval_sampler(), n_way, args.n_episodes, device)
    torch.save(encoder.state_dict(), save_path)
    print(f"\nFinal {n_way}-way {args.k_shot}-shot: {final_acc * 100:.2f}% +/- {final_ci * 100:.2f}%")
    print(f"Final checkpoint saved to {save_path}")
    return final_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate CALMFormer.")
    parser.add_argument("--dataset", default="ORACLE", choices=["ADS-B", "ORACLE", "WiSig", "WiSig_ManyTx"])
    parser.add_argument("--n_classes", type=int, default=16)
    parser.add_argument("--base_ratio", type=float, default=0.7)
    parser.add_argument("--mixer", default="rf_ilcm_anchor", choices=MIXER_TYPES)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=48)
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--conv_kernel_size", type=int, default=7)
    parser.add_argument("--rf_anchor_init_eta", type=float, default=0.7)
    parser.add_argument("--name_suffix", default="")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--pgr_weight", type=float, default=0.0)
    parser.add_argument("--pgr_margin", type=float, default=0.2)
    parser.add_argument("--pgr_compact_weight", type=float, default=1.0)
    parser.add_argument("--pgr_margin_weight", type=float, default=1.0)
    parser.add_argument("--pgr_warmup_epochs", type=int, default=0)
    parser.add_argument("--rf_aug", action="store_true")
    parser.add_argument("--rf_aug_recipe", default="full", choices=["none", "lite", "full"])
    parser.add_argument("--aug_phase", type=float, default=0.0)
    parser.add_argument("--aug_amp", type=float, default=0.0)
    parser.add_argument("--aug_awgn_min", type=float, default=20.0)
    parser.add_argument("--aug_awgn", type=float, default=0.0)
    parser.add_argument("--aug_shift", type=int, default=0)
    parser.add_argument("--aug_cfo", type=float, default=0.0)
    parser.add_argument("--aug_iq_gain", type=float, default=0.0)
    parser.add_argument("--aug_iq_phase", type=float, default=0.0)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--split_seed", type=int, default=2027)
    parser.add_argument("--train_seed", type=int, default=2027)
    parser.add_argument("--episode_seed", type=int, default=9001)
    parser.add_argument("--save_dir", default="checkpoints_oracle_calmformer")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train(args, device)
