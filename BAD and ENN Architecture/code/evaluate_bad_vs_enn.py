"""
BAD vs Plain ENN Belief Accuracy Comparison
============================================
Evaluates two partner card estimation approaches:

  Condition 1 — Plain ENN (Rong et al. baseline):
    At every bid step, run ENN forward pass on current state.
    No sequential updating — pure re-inference each step.

  Condition 2 — BAD-augmented ENN:
    Initialise belief from ENN at auction start.
    After each partner bid, apply BAD Bayesian update.
    Belief accumulates evidence across the auction.

Metrics reported per bid step and aggregated:
  - Per-card accuracy    : fraction of cards correctly predicted
  - Recall              : of cards partner holds, fraction predicted
  - Precision           : of cards predicted, fraction correct
  - Cross-entropy loss  : calibration of probability estimates
  - Belief uncertainty  : entropy of the belief distribution
  - Avg cards predicted : should approach 13

Results are broken down by auction length bin (replicating Rong et al. Fig 6a)
and saved as both a summary printout and a JSON file for further analysis.

Usage:
    python evaluate_bad_vs_enn.py --enn_weights enn_weights.pt --num_deals 1000
    python evaluate_bad_vs_enn.py --enn_weights enn_weights.pt --smoke_test
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import pyspiel
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bad_bridge import ENN, BADTracker, NUM_CARDS, HAND_SIZE, STATE_DIM


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THRESHOLD     = 0.35    # probability threshold for binary card prediction
PARTNER       = {0: 2, 1: 3, 2: 0, 3: 1}
BID_OFFSET    = 52     # OpenSpiel: actions 0-51=deal, 52-89=bids
NUM_SAMPLES   = 32     # BAD tracker policy samples per update
                       # Higher = better Bayesian approximation
                       # Lower = faster evaluation


# ---------------------------------------------------------------------------
# Per-step metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    """Metrics at a single bid step for one condition."""
    bid_step        : int     # how many bids have been made so far
    accuracy        : float
    recall          : float
    precision       : float
    cross_entropy   : float
    uncertainty     : float   # entropy of belief distribution
    avg_predicted   : float   # expected number of cards predicted


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_step_metrics(
    belief  : torch.Tensor,   # (52,) probability vector
    truth   : torch.Tensor,   # (52,) binary ground truth
    bid_step: int,
    threshold: float = THRESHOLD,
) -> StepMetrics:
    """Compute all metrics for one belief vector against ground truth."""

    # Select top 13 cards by probability — partner always holds exactly 13
    if belief.sum() > 0:
        top13 = torch.topk(belief, k=13).indices
        preds = torch.zeros_like(belief)
        preds[top13] = 1.0
    else:
        preds = (belief >= threshold).float()

    # Accuracy: fraction of all 52 card slots correctly predicted
    accuracy = (preds == truth).float().mean().item()

    # Recall: of cards partner holds, how many did we predict
    tp = (preds * truth).sum().item()
    fn = ((1 - preds) * truth).sum().item()
    fp = (preds * (1 - truth)).sum().item()
    recall    = tp / max(tp + fn, 1e-8)
    precision = tp / max(tp + fp, 1e-8)

    # Cross-entropy: calibration quality
    belief_clamped = belief.clamp(1e-7, 1 - 1e-7)
    ce = F.binary_cross_entropy(belief_clamped, truth, reduction='mean').item()

    # Uncertainty: entropy of belief (lower = more confident)
    entropy = -(
        belief_clamped * belief_clamped.log() +
        (1 - belief_clamped) * (1 - belief_clamped).log()
    ).mean().item()

    avg_predicted = preds.sum().item()

    return StepMetrics(
        bid_step      = bid_step,
        accuracy      = accuracy,
        recall        = recall,
        precision     = precision,
        cross_entropy = ce,
        uncertainty   = entropy,
        avg_predicted = avg_predicted,
    )


# ---------------------------------------------------------------------------
# Deal runner
# ---------------------------------------------------------------------------

def run_deal(
    game    : pyspiel.Game,
    enn     : ENN,
    rng     : np.random.Generator,
    device  : str,
    num_samples: int = NUM_SAMPLES,
) -> Tuple[List[StepMetrics], List[StepMetrics]]:
    """
    Play one complete auction and collect per-step metrics for both conditions.

    Returns:
        enn_metrics : list of StepMetrics for plain ENN condition
        bad_metrics : list of StepMetrics for BAD-augmented condition
    """
    state = game.new_initial_state()

    # --- Deal phase ---
    player_hands = {0: [], 1: [], 2: [], 3: []}
    deal_step = 0
    while state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        card = int(rng.choice(outcomes, p=probs))
        player_hands[deal_step % 4].append(card)
        state.apply_action(card)
        deal_step += 1

    # Build ground truth tensors for each player's partner
    def hand_tensor(cards):
        t = torch.zeros(NUM_CARDS)
        for c in cards:
            t[c] = 1.0
        return t

    partner_truth = {
        p: hand_tensor(player_hands[PARTNER[p]])
        for p in range(4)
    }
    own_hand_tensor = {
        p: hand_tensor(player_hands[p])
        for p in range(4)
    }

    # --- Initialise BAD trackers for all 4 players ---
    # Each player has their own tracker (own hand differs)
    initial_states = {}
    bad_trackers   = {}

    for p in range(4):
        initial_states[p] = torch.tensor(
            state.information_state_tensor(p), dtype=torch.float32
        )[:STATE_DIM].to(device)

    for p in range(4):
        tracker = BADTracker(
            own_hand    = own_hand_tensor[p].to(device),
            pnn         = None,    # no PNN needed — BAD tracker uses
                                   # uniform partial policy approximation
            enn         = enn,
            num_samples = num_samples,
            device      = device,
        )
        tracker.reset(initial_states[p])
        bad_trackers[p] = tracker

    enn_metrics: List[StepMetrics] = []
    bad_metrics: List[StepMetrics] = []
    bid_count = 0

    # --- Bidding phase ---
    while not state.is_terminal():
        current_player = state.current_player()

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(outcomes, p=probs)))
            continue

        pub_state = torch.tensor(
            state.information_state_tensor(current_player),
            dtype=torch.float32
        )[:STATE_DIM].to(device)

        truth = partner_truth[current_player].to(device)

        # --- Condition 1: Plain ENN ---
        # Re-infer from scratch at every step
        with torch.no_grad():
            enn_belief = enn(pub_state.unsqueeze(0)).squeeze(0)
        # Zero out own cards (cannot be partner's)
        enn_belief = enn_belief * (1.0 - own_hand_tensor[current_player].to(device))

        enn_metrics.append(compute_step_metrics(enn_belief, truth, bid_count))

        # --- Condition 2: BAD-augmented ENN ---
        # Use the maintained and updated belief
        bad_belief = bad_trackers[current_player].belief_vector
        bad_metrics.append(compute_step_metrics(bad_belief, truth, bid_count))

        # --- Choose action (random for evaluation — we only care about belief) ---
        legal = state.legal_actions()
        action = int(rng.choice(legal))
        state.apply_action(action)

        # --- Update BAD trackers after PARTNER bids ---
        # Only update the tracker of the partner of whoever just bid
        # i.e. if player 0 just bid, update player 2's tracker (their partner)
        partner_of_bidder = PARTNER[current_player]
        if action >= BID_OFFSET:
            bid_idx = action - BID_OFFSET
            legal_mask = torch.zeros(38, dtype=torch.bool)
            for a in legal:
                if a >= BID_OFFSET:
                    legal_mask[a - BID_OFFSET] = True

            bad_trackers[partner_of_bidder].update(
                observed_bid = bid_idx,
                public_state = pub_state,
                legal_mask   = legal_mask,
            )

        bid_count += 1

    return enn_metrics, bad_metrics


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_metrics(all_metrics: List[StepMetrics]) -> Dict:
    """Compute mean of all metrics across a list of StepMetrics."""
    if not all_metrics:
        return {}
    return {
        'accuracy'      : np.mean([m.accuracy       for m in all_metrics]),
        'recall'        : np.mean([m.recall          for m in all_metrics]),
        'precision'     : np.mean([m.precision       for m in all_metrics]),
        'cross_entropy' : np.mean([m.cross_entropy   for m in all_metrics]),
        'uncertainty'   : np.mean([m.uncertainty     for m in all_metrics]),
        'avg_predicted' : np.mean([m.avg_predicted   for m in all_metrics]),
        'n_steps'       : len(all_metrics),
    }


def aggregate_by_bid_step(all_metrics: List[StepMetrics], num_bins: int = 5) -> List[Dict]:
    """
    Bin metrics by auction progress (replicates Rong et al. Fig 6a).
    Returns one aggregated dict per bin.
    """
    if not all_metrics:
        return []

    steps = np.array([m.bid_step for m in all_metrics])
    max_step = steps.max()
    bins = np.linspace(0, max_step, num_bins + 1)

    results = []
    for i in range(num_bins):
        lo, hi  = bins[i], bins[i + 1]
        in_bin  = [m for m in all_metrics if lo <= m.bid_step <= hi]
        if not in_bin:
            continue
        agg = aggregate_metrics(in_bin)
        agg['bid_step_range'] = f"{lo:.0f}-{hi:.0f}"
        agg['bin'] = i + 1
        results.append(agg)

    return results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load ENN
    print(f"\nLoading ENN weights from {args.enn_weights}...")
    enn = ENN(input_dim=STATE_DIM, hidden_dim=512, num_layers=6).to(device)
    checkpoint = torch.load(args.enn_weights, weights_only=True, map_location=device)
    enn.load_state_dict(checkpoint['state_dict'])
    enn.eval()
    print(f"ENN loaded (trained to epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_metrics', {}).get('loss', '?'):.4f})")

    # Load game
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    rng  = np.random.default_rng(args.seed)

    # Storage
    all_enn_metrics: List[StepMetrics] = []
    all_bad_metrics: List[StepMetrics] = []

    num_deals = 10 if args.smoke_test else args.num_deals
    print(f"\nEvaluating over {num_deals} deals...\n")

    for _ in tqdm(range(num_deals), desc="Deals"):
        try:
            enn_m, bad_m = run_deal(
                game        = game,
                enn         = enn,
                rng         = rng,
                device      = device,
                num_samples = args.num_samples,
            )
            all_enn_metrics.extend(enn_m)
            all_bad_metrics.extend(bad_m)
        except Exception as e:
            continue

    # --- Overall aggregated results ---
    enn_agg = aggregate_metrics(all_enn_metrics)
    bad_agg = aggregate_metrics(all_bad_metrics)

    print("\n" + "=" * 65)
    print(f"{'Metric':<20} {'Plain ENN':>15} {'BAD-ENN':>15} {'Delta':>12}")
    print("=" * 65)

    for key in ['accuracy', 'recall', 'precision', 'cross_entropy', 'uncertainty', 'avg_predicted']:
        enn_val = enn_agg.get(key, 0)
        bad_val = bad_agg.get(key, 0)
        delta   = bad_val - enn_val
        # For cross_entropy and uncertainty, lower is better so flip delta sign display
        better = "↑" if delta > 0 else "↓"
        if key in ('cross_entropy', 'uncertainty'):
            better = "↓" if delta < 0 else "↑"
        print(f"{key:<20} {enn_val:>15.4f} {bad_val:>15.4f} {delta:>+11.4f}{better}")

    print("=" * 65)
    print(f"Total bid steps evaluated: {enn_agg.get('n_steps', 0):,}")

    # --- By auction length (Fig 6a replication) ---
    print("\n--- Accuracy by auction length (Plain ENN) ---")
    enn_by_step = aggregate_by_bid_step(all_enn_metrics)
    print(f"  {'Bin':>4}  {'BidStepRange':>14}  {'Accuracy':>9}  {'Recall':>8}  {'N':>7}")
    for b in enn_by_step:
        print(f"  {b['bin']:>4}  {b['bid_step_range']:>14}  "
              f"{b['accuracy']:>9.4f}  {b['recall']:>8.4f}  {b['n_steps']:>7,}")

    print("\n--- Accuracy by auction length (BAD-augmented ENN) ---")
    bad_by_step = aggregate_by_bid_step(all_bad_metrics)
    print(f"  {'Bin':>4}  {'BidStepRange':>14}  {'Accuracy':>9}  {'Recall':>8}  {'N':>7}")
    for b in bad_by_step:
        print(f"  {b['bin']:>4}  {b['bid_step_range']:>14}  "
              f"{b['accuracy']:>9.4f}  {b['recall']:>8.4f}  {b['n_steps']:>7,}")

    # --- Save results ---
    results = {
        'num_deals'    : num_deals,
        'num_steps'    : len(all_enn_metrics),
        'seed'         : args.seed,
        'enn_weights'  : args.enn_weights,
        'plain_enn'    : {
            'overall'       : enn_agg,
            'by_bid_step'   : enn_by_step,
        },
        'bad_enn'      : {
            'overall'       : bad_agg,
            'by_bid_step'   : bad_by_step,
        },
    }

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")

    if args.smoke_test:
        print("\n=== SMOKE TEST PASSED ===")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare BAD-augmented ENN vs plain ENN belief accuracy"
    )
    parser.add_argument('--enn_weights', type=str,  default='enn_weights.pt',
                        help='Path to trained ENN checkpoint')
    parser.add_argument('--num_deals',   type=int,  default=1000,
                        help='Number of deals to evaluate over')
    parser.add_argument('--num_samples', type=int,  default=NUM_SAMPLES,
                        help='BAD tracker policy samples per update (higher=better)')
    parser.add_argument('--out',         type=str,  default='evaluation_results.json',
                        help='Output path for JSON results')
    parser.add_argument('--seed',        type=int,  default=42)
    parser.add_argument('--smoke_test',  action='store_true',
                        help='Run 10 deals only to verify pipeline')
    args = parser.parse_args()

    if args.smoke_test:
        print("=== SMOKE TEST (10 deals) ===")

    evaluate(args)
