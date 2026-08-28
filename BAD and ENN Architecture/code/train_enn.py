"""
ENN Supervised Training Script
================================
Trains the Estimation Neural Network (ENN) from bad_bridge.py on the
dataset produced by bridge_dataset.py.

Follows Rong et al.'s supervised pretraining setup:
  - Loss     : binary cross-entropy (independent Bernoulli per card)
  - Optimiser: Adam
  - Split    : 70% train / 10% val / 20% test (done in bridge_dataset.py)
  - Metric   : per-card accuracy and recall (matching Rong et al. Fig 6a)

Usage:
    # First generate data:
    python bridge_dataset.py --num_games 500000 --out_path bridge_enn_data.pt

    # Then train:
    python train_enn.py --data bridge_enn_data.pt --out enn_weights.pt

    # Quick smoke test on the toy dataset from the smoke test:
    python train_enn.py --data bridge_enn_data.pt --epochs 3 --smoke_test
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Import ENN from our main module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from bad_bridge import ENN, NUM_CARDS, HAND_SIZE, STATE_DIM


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class BridgeEnnDataset(Dataset):
    """
    Simple Dataset wrapper around the (states, targets) tensors
    produced by bridge_dataset.py.
    """
    def __init__(self, states: torch.Tensor, targets: torch.Tensor):
        assert states.shape[0] == targets.shape[0]
        assert states.shape[1]  == STATE_DIM, \
            f"Expected STATE_DIM={STATE_DIM}, got {states.shape[1]}. " \
            f"Re-run bridge_dataset.py or update STATE_DIM in bad_bridge.py."
        assert targets.shape[1] == NUM_CARDS
        self.states  = states
        self.targets = targets

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    enn      : ENN,
    loader   : DataLoader,
    device   : str,
    threshold: float = 0.5,
) -> dict:
    """
    Compute per-card accuracy and recall over a DataLoader.

    Accuracy: fraction of (sample, card) pairs correctly predicted
    Recall  : of cards partner actually holds, fraction correctly predicted
    BCE loss: average binary cross-entropy

    These match the evaluation metrics in Rong et al. Fig 6a.
    """
    enn.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    num_batches = 0

    for states, targets in loader:
        states  = states.to(device)
        targets = targets.to(device)

        probs = enn(states)                          # (B, 52)
        loss  = enn.loss(states, targets)
        preds = (probs >= threshold).float()         # (B, 52) binary

        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())
        total_loss  += loss.item()
        num_batches += 1

    preds  = torch.cat(all_preds,  dim=0)   # (N, 52)
    labels = torch.cat(all_labels, dim=0)   # (N, 52)

    # Per-card accuracy: fraction correct across all samples
    accuracy = (preds == labels).float().mean().item()

    # Recall: TP / (TP + FN) — over cards partner actually holds
    true_positives  = (preds * labels).sum().item()
    false_negatives = ((1 - preds) * labels).sum().item()
    recall = true_positives / max(true_positives + false_negatives, 1)

    # Precision: TP / (TP + FP)
    false_positives = (preds * (1 - labels)).sum().item()
    precision = true_positives / max(true_positives + false_positives, 1)

    # Expected cards predicted per sample (should approach 13)
    avg_predicted = preds.sum(dim=1).mean().item()

    return {
        'loss'         : total_loss / max(num_batches, 1),
        'accuracy'     : accuracy,
        'recall'       : recall,
        'precision'    : precision,
        'avg_predicted': avg_predicted,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    # --- Load dataset ---
    print(f"\nLoading dataset from {args.data}...")
    data = torch.load(args.data, weights_only=True)

    # Handle both split and unsplit saves from bridge_dataset.py
    if isinstance(data, dict) and 'train' in data:
        train_states  = data['train']['states']
        train_targets = data['train']['targets']
        val_states    = data['val']['states']
        val_targets   = data['val']['targets']
        test_states   = data['test']['states']
        test_targets  = data['test']['targets']
    else:
        # Unsplit — do a quick 70/10/20 split here
        states, targets = data['states'], data['targets']
        N = states.shape[0]
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
        n_tr = int(N * 0.7); n_va = int(N * 0.1)
        train_states, train_targets = states[perm[:n_tr]],         targets[perm[:n_tr]]
        val_states,   val_targets   = states[perm[n_tr:n_tr+n_va]],targets[perm[n_tr:n_tr+n_va]]
        test_states,  test_targets  = states[perm[n_tr+n_va:]],    targets[perm[n_tr+n_va:]]

    print(f"  Train: {len(train_states):,}  Val: {len(val_states):,}  Test: {len(test_states):,}")

    train_loader = DataLoader(
        BridgeEnnDataset(train_states, train_targets),
        batch_size=args.batch_size, shuffle=True,  num_workers=0,
    )
    val_loader = DataLoader(
        BridgeEnnDataset(val_states, val_targets),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        BridgeEnnDataset(test_states, test_targets),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # --- Build ENN ---
    enn = ENN(
        input_dim  = STATE_DIM,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
    ).to(device)

    total_params = sum(p.numel() for p in enn.parameters())
    print(f"\nENN parameters: {total_params:,}")

    optimizer = torch.optim.Adam(enn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # --- Training ---
    best_val_loss = float('inf')
    best_epoch    = 0
    history       = []

    print(f"\nTraining for up to {args.epochs} epochs...\n")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>8}  "
          f"{'ValAcc':>7}  {'ValRecall':>9}  {'AvgPred':>7}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # --- Train epoch ---
        enn.train()
        total_train_loss = 0.0
        num_batches = 0

        for states_b, targets_b in train_loader:
            states_b  = states_b.to(device)
            targets_b = targets_b.to(device)

            optimizer.zero_grad()
            loss = enn.loss(states_b, targets_b)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(enn.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()
            num_batches += 1

        train_loss = total_train_loss / max(num_batches, 1)

        # --- Validation ---
        val_metrics = compute_metrics(enn, val_loader, device)
        scheduler.step(val_metrics['loss'])

        # Log
        row = {
            'epoch'     : epoch,
            'train_loss': train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_metrics['loss']:>8.4f}  "
              f"{val_metrics['accuracy']:>7.4f}  {val_metrics['recall']:>9.4f}  "
              f"{val_metrics['avg_predicted']:>7.2f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch    = epoch
            torch.save({
                'epoch'     : epoch,
                'state_dict': enn.state_dict(),
                'val_metrics': val_metrics,
                'args'      : vars(args),
            }, args.out)

        # Early stopping
        if epoch - best_epoch >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    # --- Final test evaluation ---
    print(f"\nBest model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"Loading best weights from {args.out}...")

    checkpoint = torch.load(args.out, weights_only=True)
    enn.load_state_dict(checkpoint['state_dict'])

    test_metrics = compute_metrics(enn, test_loader, device)
    print(f"\nTest set results:")
    for k, v in test_metrics.items():
        print(f"  {k:>15s}: {v:.4f}")

    # Save test metrics alongside checkpoint
    checkpoint['test_metrics'] = test_metrics
    torch.save(checkpoint, args.out)
    print(f"\nCheckpoint saved to {args.out}")

    # --- How belief accuracy grows with bidding length ---
    # Rong et al. Fig 6: accuracy increases as more bids observed.
    # We approximate this by binning samples by how many bids have
    # been made (inferred from the non-zero dims in the bidding history
    # region of the state tensor).
    print("\nAccuracy by auction length (approx):")
    _accuracy_by_length(enn, test_states, test_targets, device)


def _accuracy_by_length(
    enn    : ENN,
    states : torch.Tensor,
    targets: torch.Tensor,
    device : str,
    threshold: float = 0.5,
    num_bins: int = 5,
):
    """
    Bin test samples by approximate auction length and report accuracy.
    Auction length is approximated by counting non-zero dims in the
    bidding history region (dims 0-431 in our 484-dim tensor, since
    dims 432-483 are the hand encoding).
    """
    # Count non-zero dims in bidding history region as proxy for auction length
    bid_region  = states[:, :432]          # history encoding region
    bid_density = bid_region.sum(dim=1)    # (N,) — proxy for auction length

    enn.eval()
    with torch.no_grad():
        probs  = enn(states.to(device)).cpu()
        preds  = (probs >= threshold).float()
        correct = (preds == targets).float().mean(dim=1)  # (N,) per-sample accuracy

    # Bin by bid density
    bins = torch.quantile(bid_density, torch.linspace(0, 1, num_bins + 1))
    print(f"  {'Bin':>4}  {'AvgBidDensity':>14}  {'Accuracy':>9}  {'N':>7}")
    for i in range(num_bins):
        lo, hi  = bins[i].item(), bins[i + 1].item()
        mask    = (bid_density >= lo) & (bid_density <= hi)
        if mask.sum() == 0:
            continue
        avg_acc = correct[mask].mean().item()
        avg_den = bid_density[mask].mean().item()
        print(f"  {i+1:>4}  {avg_den:>14.1f}  {avg_acc:>9.4f}  {mask.sum().item():>7,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ENN on OpenSpiel bridge data")
    parser.add_argument('--data',       type=str,   default='bridge_enn_data.pt')
    parser.add_argument('--out',        type=str,   default='enn_weights.pt')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int,   default=512)
    parser.add_argument('--num_layers', type=int,   default=6)
    parser.add_argument('--patience',   type=int,   default=10,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--cpu',        action='store_true')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run 3 epochs only to verify pipeline')
    args = parser.parse_args()

    if args.smoke_test:
        print("=== SMOKE TEST: 3 epochs only ===")
        args.epochs = 3

    train(args)
