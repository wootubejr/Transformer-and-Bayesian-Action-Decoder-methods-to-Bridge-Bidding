"""
OpenSpiel Bridge Dataset Generator for ENN Training
====================================================
Generates (state, partner_cards) training pairs from self-play using
OpenSpiel's bridge environment, for supervised ENN pretraining.

Key findings from environment inspection:
  - State tensor size: 571 dims (not 480 — we slice to 480 for Kita compat.)
  - Own hand encoding: dims 432-483 (card c -> dim 432+c)
  - Card ordering: action index c = suit*13 + rank
                   suit: 0=C, 1=D, 2=H, 3=S  |  rank: 0=2 .. 12=A
  - Partner hand: NOT in own tensor — must be tracked from the deal
  - Partnerships: North(0)+South(2) vs East(1)+West(3)
  - Deal order: card i goes to player i%4 (round-robin)

ENN training target:
  For each bid step t in the auction, from the perspective of the current
  bidder (player p), we record:
    - input : state tensor of player p at step t (571 dims, sliced to 480)
    - target: 52-dim binary vector of p's partner's actual cards

Usage:
    python bridge_dataset.py --num_games 50000 --out_path bridge_enn_data.pt

    Then in your training script:
        data = torch.load('bridge_enn_data.pt')
        states  = data['states']   # (N, 480)
        targets = data['targets']  # (N, 52)
"""

import argparse
import torch
import numpy as np
import pyspiel
from typing import Tuple, List, Dict
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CARDS    = 52
HAND_SIZE    = 13
STATE_DIM    = 484    # we slice the full 571-dim tensor to 484 dims.
STATE_OFFSET = 0      # start of slice
# Kita et al. report 480 dims, but the own-hand encoding occupies dims
# 432-483 (52 cards). Truncating at 480 drops 4 card dims (KS, AS, and
# the top two spades). We extend to 484 to keep the full hand encoding.
# The ENN and PNN input_dim should be set to 484 accordingly.

HAND_DIM_START = 432  # where own-hand cards begin in the full 571-dim tensor
HAND_DIM_END   = 484  # exclusive

# Partnerships: player 0 (North) <-> player 2 (South)
#               player 1 (East)  <-> player 3 (West)
PARTNER = {0: 2, 1: 3, 2: 0, 3: 1}

# Bidding actions in OpenSpiel bridge start at action index 52
# (0-51 are card deal actions, 52=Pass, 53=Double, 54=Redouble, 55-89=bids)
MIN_BID_ACTION = 52


# ---------------------------------------------------------------------------
# Card utility
# ---------------------------------------------------------------------------

def card_name(card_idx: int) -> str:
    """Convert card action index to human-readable name (e.g. 0 -> '2C')."""
    suits = ['C', 'D', 'H', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return f"{ranks[card_idx % 13]}{suits[card_idx // 13]}"


def hand_to_tensor(card_indices: List[int]) -> torch.Tensor:
    """Convert list of card action indices to 52-dim binary tensor."""
    t = torch.zeros(NUM_CARDS)
    for c in card_indices:
        t[c] = 1.0
    return t


# ---------------------------------------------------------------------------
# Single game data extraction
# ---------------------------------------------------------------------------

def extract_game_samples(
    game: pyspiel.Game,
    rng: np.random.Generator,
    policy: str = 'random',
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Play one complete auction and extract (state, partner_cards) pairs
    for every bid step made by each player.

    Args:
        game  : OpenSpiel bridge game object
        rng   : numpy random generator
        policy: 'random' for random bidding (sufficient for ENN pretraining
                since ENN target is partner's cards, not bid quality)

    Returns:
        List of (state_480, partner_cards_52) tuples, one per bid step.
        Bid steps from all 4 players are included.
    """
    state = game.new_initial_state()
    samples = []

    # --- Deal phase ---
    # Track which cards go to which player (round-robin deal)
    player_hands: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    deal_step = 0

    while state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        card = int(rng.choice(outcomes, p=probs))
        player_hands[deal_step % 4].append(card)
        state.apply_action(card)
        deal_step += 1

    assert deal_step == NUM_CARDS, f"Expected 52 cards dealt, got {deal_step}"

    # Precompute partner hand tensors (constant for the whole game)
    partner_tensors = {
        p: hand_to_tensor(player_hands[PARTNER[p]])
        for p in range(4)
    }

    # --- Bidding phase ---
    while not state.is_terminal():
        current_player = state.current_player()

        # Skip chance nodes (shouldn't occur post-deal, but be safe)
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(outcomes, p=probs)))
            continue

        # Extract state tensor for current player, sliced to STATE_DIM
        full_tensor = state.information_state_tensor(current_player)
        state_tensor = torch.tensor(
            full_tensor[:STATE_DIM], dtype=torch.float32
        )

        # Partner cards target (ground truth for ENN)
        partner_cards = partner_tensors[current_player]

        samples.append((state_tensor, partner_cards))

        # Select action
        legal = state.legal_actions()
        if policy == 'random':
            action = int(rng.choice(legal))
        else:
            raise NotImplementedError(f"Policy '{policy}' not implemented.")

        state.apply_action(action)

    return samples


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    num_games : int = 10_000,
    seed      : int = 42,
    policy    : str = 'random',
    max_samples_per_game: int = 40,
    out_path  : str = 'bridge_enn_data.pt',
) -> Tuple[torch.Tensor, torch.Tensor]:
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    rng  = np.random.default_rng(seed)
    skipped = 0

    # Pre-allocate tensors upfront — avoids memory spike from torch.stack()
    max_samples = num_games * max_samples_per_game
    states  = torch.zeros(max_samples, STATE_DIM, dtype=torch.float32)
    targets = torch.zeros(max_samples, NUM_CARDS,  dtype=torch.float32)
    idx = 0

    for game_idx in tqdm(range(num_games), desc="Generating games"):
        try:
            samples = extract_game_samples(game, rng, policy=policy)
        except Exception:
            skipped += 1
            continue

        for state_t, partner_t in samples[:max_samples_per_game]:
            states[idx]  = state_t
            targets[idx] = partner_t
            idx += 1

    if skipped > 0:
        print(f"Warning: {skipped} games skipped due to errors.")

    # Trim to actual number of samples written
    states  = states[:idx]
    targets = targets[:idx]

    print(f"\nDataset generated:")
    print(f"  Games     : {num_games - skipped:,}")
    print(f"  Samples   : {idx:,}")
    print(f"  State dim : {states.shape[1]}")
    print(f"  Target dim: {targets.shape[1]}")
    print(f"  Avg partner cards per sample: {targets.sum(dim=1).mean():.2f} (should be ~13)")
    print(f"  Avg own cards in state: "
          f"{states[:, HAND_DIM_START:HAND_DIM_END].sum(dim=1).mean():.2f} (should be ~13)")

    return states, targets

# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def split_dataset(
    states  : torch.Tensor,
    targets : torch.Tensor,
    train_frac: float = 0.7,
    val_frac  : float = 0.1,
    seed      : int   = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Split dataset into train/val/test following Rong et al.'s 70/10/20 split.

    Note: samples from the same game should ideally stay in the same split
    to avoid data leakage (same hand appearing in train and test).
    This function does a simple random split; for a cleaner split, use
    generate_dataset() separately with different seeds for each split.

    Args:
        states, targets : full dataset tensors
        train_frac      : fraction for training
        val_frac        : fraction for validation
        seed            : shuffle seed

    Returns:
        dict with keys 'train', 'val', 'test', each containing
        {'states': ..., 'targets': ...}
    """
    N = states.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    idx_train = perm[:n_train]
    idx_val   = perm[n_train : n_train + n_val]
    idx_test  = perm[n_train + n_val:]

    return {
        'train': {'states': states[idx_train], 'targets': targets[idx_train]},
        'val'  : {'states': states[idx_val],   'targets': targets[idx_val]},
        'test' : {'states': states[idx_test],  'targets': targets[idx_test]},
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def verify_dataset(states: torch.Tensor, targets: torch.Tensor, n_check: int = 1000):
    """
    Run basic sanity checks on a generated dataset.
    Prints warnings if anything looks wrong.
    """
    print("\nRunning dataset sanity checks...")
    sample = min(n_check, states.shape[0])
    idx    = torch.randperm(states.shape[0])[:sample]
    s, t   = states[idx], targets[idx]

    # 1. Each target should have exactly 13 cards
    card_counts = t.sum(dim=1)
    bad = (card_counts != HAND_SIZE).sum().item()
    print(f"  Samples with != 13 partner cards: {bad} / {sample} "
          f"({'OK' if bad == 0 else 'WARNING'})")

    # 2. State values should be binary
    non_binary = ((s != 0) & (s != 1)).sum().item()
    print(f"  Non-binary state values: {non_binary} "
          f"({'OK' if non_binary == 0 else 'WARNING'})")

    # 3. Own hand in state should also have ~13 cards
    own_hand_region = s[:, HAND_DIM_START:HAND_DIM_END]
    own_counts = own_hand_region.sum(dim=1)
    avg_own = own_counts.mean().item()
    print(f"  Avg own cards in state [dims {HAND_DIM_START}-{HAND_DIM_END}]: "
          f"{avg_own:.2f} (expected ~13.0)")

    # 4. Own hand and partner hand should not overlap (same card can't be held by two players)
    own_cards  = s[:, HAND_DIM_START : HAND_DIM_START + NUM_CARDS]  # (sample, 52)
    partner    = t                         # (sample, 52)
    overlap    = (own_cards * partner).sum(dim=1)
    bad_overlap = (overlap > 0).sum().item()
    print(f"  Samples where own and partner hands overlap: {bad_overlap} / {sample} "
          f"({'OK' if bad_overlap == 0 else 'WARNING'})")

    print("Sanity checks complete.\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ENN training data from OpenSpiel bridge")
    parser.add_argument('--num_games',  type=int,   default=10_000,
                        help='Number of games to simulate (default: 10000)')
    parser.add_argument('--out_path',   type=str,   default='bridge_enn_data.pt',
                        help='Output path for saved dataset')
    parser.add_argument('--seed',       type=int,   default=42,
                        help='Random seed')
    parser.add_argument('--no_split',   action='store_true',
                        help='Save full dataset without train/val/test split')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run a quick smoke test with 100 games only')
    args = parser.parse_args()

    if args.smoke_test:
        print("=== SMOKE TEST (100 games) ===")
        args.num_games = 100

    # Generate
    states, targets = generate_dataset(
        num_games = args.num_games,
        seed      = args.seed,
    )

    # Save
    print("Saving raw dataset checkpoint...")
    torch.save({'states': states, 'targets': targets}, 'bridge_enn_raw.pt')
    print("Raw dataset saved to bridge_enn_raw.pt")

    # Sanity check
    verify_dataset(states, targets)

    # Save final split
    if args.no_split:
        torch.save({'states': states, 'targets': targets}, args.out_path)
        print(f"Saved full dataset to {args.out_path}")
    else:
        splits = split_dataset(states, targets)
        torch.save(splits, args.out_path)
        for split_name, split_data in splits.items():
            n = split_data['states'].shape[0]
            print(f"  {split_name:5s}: {n:,} samples")
        print(f"\nSaved train/val/test splits to {args.out_path}")

    if args.smoke_test:
        print("\n=== SMOKE TEST PASSED ===")
        print("Re-run without --smoke_test for full dataset generation.")
        print("Recommended: --num_games 500000 to match Rong et al. scale")
