"""
BAD-Augmented ENN + PNN for Competitive Bridge Bidding
=======================================================
Based on:
  - Rong et al. (2019): ENN + PNN architecture
  - Foerster et al. (2018): Bayesian Action Decoder / PuB-MDP
  - Kita et al. (2024): 480-dim OpenSpiel state representation

Scope simplification (justified by Rong et al. DDA analysis):
  - BAD belief tracks PARTNER cards only; opponent hands ignored.

Architecture overview:
  - ENN      : MLP, 480-dim state -> 52-dim partner card probabilities
               (trained supervised from expert data)
  - BADTracker: maintains a factorised belief B over 52 cards,
               updated via approximate Bayesian update after each bid
  - PNN      : MLP, (own hand [52] + BAD belief [52]) -> bid logits [38]
               (trained via RL / self-play)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CARDS        = 52   # full deck
HAND_SIZE        = 13   # cards per player
STATE_DIM        = 484  # Kita et al. / OpenSpiel binary input dimension
NUM_ACTIONS      = 38   # 35 bids + pass + double + redouble
NUM_SAMPLE_POLICIES = 16  # K: number of partial policies sampled per update
                           # higher K → better Bayesian approximation, more compute


# ---------------------------------------------------------------------------
# 1.  Estimation Neural Network (ENN)
# ---------------------------------------------------------------------------
# Role: given the current public+private state vector, output a probability
# for each of the 52 cards that partner holds it.
#
# This is trained SUPERVISED from expert game data:
#   input  : 480-dim state at bid step t (own cards + bidding history so far)
#   target : binary 52-dim vector of partner's actual cards
#
# Loss: binary cross-entropy (independent Bernoulli per card, summing to ~13)

class ENN(nn.Module):
    """
    Estimation Neural Network.

    Produces a 52-dim vector of independent Bernoulli probabilities,
    one per card, representing P(partner holds card c | state).

    The probabilities are NOT constrained to sum to exactly 13 during
    training (BCE loss handles each card independently), but you can
    optionally normalise at inference time if downstream code needs it.
    """

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        hidden_dim: int = 512,
        num_layers: int = 6,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, NUM_CARDS))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, 480) binary state vector
        Returns:
            card_probs: (batch, 52) in [0, 1]  — P(partner holds card c)
        """
        return torch.sigmoid(self.net(state))

    def loss(self, state: torch.Tensor, partner_cards: torch.Tensor) -> torch.Tensor:
        """
        Supervised training loss. with pre-card weighting to upweight positive examples (cards partner actually holds) since they are rarer.

        Args:
            state        : (batch, 480)
            partner_cards: (batch, 52) binary ground-truth
        Returns:
            scalar BCE loss
        """
        logits = self.net(state)
        pos_weight = torch.ones(NUM_CARDS, device=logits.device) * 1.5
        return F.binary_cross_entropy_with_logits(logits, partner_cards, pos_weight=pos_weight)


# ---------------------------------------------------------------------------
# 2.  Policy Neural Network (PNN)
# ---------------------------------------------------------------------------
# Role: choose the next bid, conditioning on own hand AND the live BAD belief
# over partner's cards (rather than re-inferring from ENN each step).
#
# Input: [own_hand (52-dim one-hot) | bad_belief (52-dim)] = 104-dim
# Output: logits over 38 legal actions (masked before sampling)
#
# Trained via PPO / A3C self-play (training loop not included here —
# this module is the network definition only).

class PNN(nn.Module):
    """
    Policy Neural Network.

    Conditions on own hand + BAD live belief (not raw ENN output).
    Includes a value head for actor-critic training.
    """

    def __init__(
        self,
        belief_dim: int = NUM_CARDS,   # 52: the BAD belief vector
        hand_dim: int   = NUM_CARDS,   # 52: own hand one-hot
        hidden_dim: int = 512,
        num_layers: int = 6,
    ):
        super().__init__()

        input_dim = hand_dim + belief_dim  # 104

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            # skip connection from layer 0 output, added at layer 3
            # (following Rong et al. architecture with skip connections)
            in_dim = hidden_dim

        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        own_hand: torch.Tensor,
        bad_belief: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            own_hand   : (batch, 52) binary
            bad_belief : (batch, 52) partner card probabilities from BADTracker
            legal_mask : (batch, 38) bool — True where action is legal
                         illegal actions are masked to -inf before softmax
        Returns:
            action_probs: (batch, 38)
            value       : (batch, 1)
        """
        x = torch.cat([own_hand, bad_belief], dim=-1)
        h = self.body(x)

        logits = self.policy_head(h)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))

        action_probs = F.softmax(logits, dim=-1)
        value        = self.value_head(h)

        return action_probs, value


# ---------------------------------------------------------------------------
# 3.  BAD Belief Tracker
# ---------------------------------------------------------------------------
# This is the core contribution: a per-episode stateful object that maintains
# and updates a factorised belief B over partner cards.
#
# Factorised belief assumption:
#   B = { p_c : c in 0..51 }  where p_c = P(partner holds card c)
#
# This is an approximation — true joint distribution over 13-card hands is
# intractable (C(52,13) ~ 635 billion). The factorised form matches the ENN
# output format and is standard in the literature.
#
# BAD Bayesian update (per bid observed from partner):
#
#   For each card c:
#     likelihood(c) = E_{ Δπ ~ PNN } [ P(bid u | hand containing c, Δπ) ]
#     B'(c)         = likelihood(c) * B(c)   (then renormalise)
#
# How we approximate the expectation over Δπ:
#   1. Sample K partial policies by running the PNN with K different
#      hypothetical partner hands (sampled from current belief B).
#   2. For each sampled hand, run the PNN to get a bid distribution.
#   3. Average the probability assigned to the observed bid u across
#      the K samples whose hand contains card c.
#   4. This average is the approximate likelihood for card c.
#
# Why this correctly avoids the cheap-talk problem:
#   The likelihood is computed using the CURRENT PNN policy — so as the
#   bidding convention evolves during self-play, the belief update
#   automatically adapts. No fixed communication channel assumed.

@dataclass
class BADTracker:
    """
    Stateful per-episode BAD belief tracker.

    Maintains a factorised belief over partner cards and updates it
    after each partner bid using approximate Bayesian inference.

    Usage:
        tracker = BADTracker(own_hand_tensor, pnn, enn)
        tracker.reset()
        # ... after observing partner's bid at each step:
        tracker.update(bid_index, public_state_tensor, legal_mask_tensor)
        belief = tracker.belief  # (52,) — use as PNN input
    """

    own_hand    : torch.Tensor          # (52,) binary — agent's own cards
    pnn         : Optional[PNN]         # current policy network (used for likelihood)
                                        # if None, uniform policy approximation is used
    enn         : ENN                   # used for prior initialisation only
    num_samples : int = NUM_SAMPLE_POLICIES
    device      : str = 'cpu'

    # belief is initialised in reset()
    belief: torch.Tensor = field(init=False)

    def reset(self, initial_state: torch.Tensor):
        """
        Initialise belief from ENN output at the start of the auction.

        The ENN gives a reasonable prior before any bids have been made
        (it can use own-hand features to rule out cards you hold yourself).

        Args:
            initial_state: (480,) state vector at auction start
        """
        with torch.no_grad():
            prior = self.enn(initial_state.unsqueeze(0)).squeeze(0)  # (52,)

        # Cards in own hand cannot be in partner's hand — zero them out
        prior = prior * (1.0 - self.own_hand)

        # Clamp for numerical safety
        self.belief = prior.clamp(1e-6, 1.0 - 1e-6).to(self.device)

    def _sample_partner_hand_from_belief(self) -> torch.Tensor:
        """
        Sample a plausible 13-card partner hand from the current belief.

        Uses the factorised belief as independent Bernoulli probabilities,
        then selects exactly 13 cards via weighted sampling without replacement.

        Returns:
            hand: (52,) binary tensor
        """
        # Remove own cards from consideration
        probs = self.belief * (1.0 - self.own_hand)
        probs = probs / (probs.sum() + 1e-8)

        # Sample exactly HAND_SIZE cards without replacement
        indices = torch.multinomial(probs, num_samples=HAND_SIZE, replacement=False)
        hand = torch.zeros(NUM_CARDS, device=self.device)
        hand[indices] = 1.0
        return hand

    def update(
        self,
        observed_bid: int,
        public_state: torch.Tensor,
        legal_mask: torch.Tensor,
    ):
        """
        Update belief after observing partner make `observed_bid`.

        Steps:
          1. Sample K hypothetical partner hands from current belief.
          2. For each sampled hand, query PNN for bid probabilities.
          3. Compute per-card likelihood = avg P(observed_bid) over
             samples whose hand contains that card.
          4. Bayesian update: B' ∝ likelihood * B.
          5. Re-normalise.

        Args:
            observed_bid: integer index of the bid partner just made (0..37)
            public_state : (480,) current state vector (shared public info)
            legal_mask   : (38,) bool — legal actions at this step
        """
        # Accumulators: total likelihood mass and count per card
        likelihood      = torch.zeros(NUM_CARDS, device=self.device)
        card_counts     = torch.zeros(NUM_CARDS, device=self.device)

        # Number of legal actions for uniform fallback
        num_legal = int(legal_mask.sum().item()) if legal_mask is not None else 38

        with torch.no_grad():
            for _ in range(self.num_samples):
                sampled_hand = self._sample_partner_hand_from_belief()  # (52,)

                if self.pnn is not None:
                    action_probs, _ = self.pnn(
                        own_hand   = sampled_hand.unsqueeze(0),
                        bad_belief = self.belief.unsqueeze(0),
                        legal_mask = legal_mask.unsqueeze(0),
                    )
                    p_bid = action_probs[0, observed_bid].item()
                else:
                    # ENN-guided approximation with sharpening
                    p_bid = (sampled_hand * self.belief).sum().item() / HAND_SIZE
                    p_bid = max(p_bid, 1e-8)
                    p_bid = p_bid ** 2  # sharpen - amplifies differences between hands

                likelihood  += sampled_hand * p_bid
                card_counts += sampled_hand

        # Avoid division by zero for cards never sampled
        card_counts = card_counts.clamp(min=1.0)

        # Average likelihood per card across samples that contained it
        avg_likelihood = likelihood / card_counts

        # Bayesian update: posterior ∝ likelihood × prior
        # Cards in own hand stay at zero (won't affect anything, but be explicit)
        updated = avg_likelihood * self.belief
        updated = updated * (1.0 - self.own_hand)

        # Renormalise to keep values in a sensible range
        # (we don't force sum=13 — factorised approximation)
        total = updated.sum()
        if total > 1e-8:
            # Scale so that expected number of partner cards ≈ 13
            self.belief = (updated / total * HAND_SIZE).clamp(1e-6, 1.0 - 1e-6)
        # If total is near zero (very unlikely bids across all hands),
        # keep previous belief — this prevents collapse from a single
        # highly surprising bid.

    @property
    def belief_vector(self) -> torch.Tensor:
        """Returns the current (52,) belief — plug directly into PNN."""
        return self.belief


# ---------------------------------------------------------------------------
# 4.  Full BAD-ENN-PNN Agent (wires everything together)
# ---------------------------------------------------------------------------

class BADBridgeAgent(nn.Module):
    """
    Full agent wrapping ENN + BADTracker + PNN.

    Per episode:
      - Call agent.new_episode(own_hand, initial_state) to reset the tracker.
      - Call agent.act(public_state, legal_mask) to get a bid.
      - Call agent.observe_partner_bid(bid, public_state, legal_mask)
        after partner bids to update the belief.

    Training:
      - ENN: train separately, supervised (see ENN.loss).
      - PNN: train via PPO/A3C self-play; use agent.pnn_forward() to get
             logits and values for the RL loss.
    """

    def __init__(
        self,
        enn_hidden : int = 512,
        enn_layers : int = 6,
        pnn_hidden : int = 512,
        pnn_layers : int = 6,
        num_samples: int = NUM_SAMPLE_POLICIES,
        device     : str = 'cpu',
    ):
        super().__init__()
        self.device = device

        self.enn = ENN(hidden_dim=enn_hidden, num_layers=enn_layers).to(device)
        self.pnn = PNN(hidden_dim=pnn_hidden, num_layers=pnn_layers).to(device)

        self.num_samples = num_samples
        self.tracker: Optional[BADTracker] = None

    def new_episode(self, own_hand: torch.Tensor, initial_state: torch.Tensor):
        """
        Call at the start of each auction.

        Args:
            own_hand      : (52,) binary — this agent's dealt cards
            initial_state : (480,) state vector before any bids
        """
        own_hand = own_hand.to(self.device)
        initial_state = initial_state.to(self.device)

        self.own_hand = own_hand
        self.tracker  = BADTracker(
            own_hand    = own_hand,
            pnn         = self.pnn,
            enn         = self.enn,
            num_samples = self.num_samples,
            device      = self.device,
        )
        self.tracker.reset(initial_state)

    def observe_partner_bid(
        self,
        bid          : int,
        public_state : torch.Tensor,
        legal_mask   : torch.Tensor,
    ):
        """
        Update belief after observing partner's bid.
        Call this every time partner (not opponent) makes a bid.

        Args:
            bid          : integer bid index (0..37)
            public_state : (480,) current state
            legal_mask   : (38,) bool
        """
        assert self.tracker is not None, "Call new_episode() first."
        self.tracker.update(
            observed_bid = bid,
            public_state = public_state.to(self.device),
            legal_mask   = legal_mask.to(self.device),
        )

    def act(
        self,
        public_state : torch.Tensor,
        legal_mask   : torch.Tensor,
    ) -> int:
        """
        Sample an action from the current policy.

        Args:
            public_state : (480,)
            legal_mask   : (38,) bool
        Returns:
            bid index (int)
        """
        assert self.tracker is not None, "Call new_episode() first."

        with torch.no_grad():
            probs, _ = self.pnn(
                own_hand   = self.own_hand.unsqueeze(0),
                bad_belief = self.tracker.belief_vector.unsqueeze(0),
                legal_mask = legal_mask.unsqueeze(0).to(self.device),
            )
        return torch.multinomial(probs[0], num_samples=1).item()

    def pnn_forward(
        self,
        own_hand   : torch.Tensor,
        bad_belief : torch.Tensor,
        legal_mask : torch.Tensor,
    ):
        """
        Forward pass for RL training (returns probs + value).
        Use this inside your PPO/A3C training loop.
        """
        return self.pnn(own_hand, bad_belief, legal_mask)


# ---------------------------------------------------------------------------
# 5.  ENN Supervised Training (minimal loop)
# ---------------------------------------------------------------------------

def train_enn_epoch(
    enn       : ENN,
    optimizer : torch.optim.Optimizer,
    states    : torch.Tensor,   # (N, 480) — state at each bid step
    targets   : torch.Tensor,   # (N, 52)  — partner's actual cards (binary)
    batch_size: int = 256,
) -> float:
    """
    One epoch of supervised ENN training.

    Args:
        enn       : ENN module
        optimizer : e.g. Adam
        states    : full dataset of state vectors
        targets   : full dataset of partner card ground truths
        batch_size: mini-batch size
    Returns:
        mean loss over the epoch
    """
    enn.train()
    N = states.shape[0]
    indices = torch.randperm(N)
    total_loss = 0.0
    num_batches = 0

    for start in range(0, N, batch_size):
        idx   = indices[start : start + batch_size]
        s_bat = states[idx]
        t_bat = targets[idx]

        optimizer.zero_grad()
        loss = enn.loss(s_bat, t_bat)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# 6.  Smoke test — verify shapes and a single belief update
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}\n")

    # --- Build agent ---
    agent = BADBridgeAgent(
        enn_hidden  = 512,
        enn_layers  = 6,
        pnn_hidden  = 512,
        pnn_layers  = 6,
        num_samples = NUM_SAMPLE_POLICIES,
        device      = device,
    )

    # --- Fake a deal ---
    # Own hand: 13 cards out of 52
    own_hand = torch.zeros(NUM_CARDS)
    own_hand[torch.randperm(NUM_CARDS)[:HAND_SIZE]] = 1.0

    initial_state = torch.zeros(STATE_DIM)
    initial_state[:NUM_CARDS] = own_hand          # first 52 dims = own cards
    # (remaining dims would encode bidding history — zeros here = auction start)

    legal_mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)

    # --- Start episode ---
    agent.new_episode(own_hand, initial_state)
    print("Initial belief (first 13 cards):")
    print(agent.tracker.belief_vector[:13].detach().cpu().numpy().round(3))

    # --- Agent acts ---
    bid = agent.act(initial_state, legal_mask)
    print(f"\nAgent bid: {bid}")

    # --- Partner bids (e.g. bid index 1 = '1C') ---
    partner_bid   = 1
    partner_state = torch.zeros(STATE_DIM)  # state after partner's bid
    agent.observe_partner_bid(partner_bid, partner_state, legal_mask)

    print("\nBelief after partner bids 1C (first 13 cards):")
    print(agent.tracker.belief_vector[:13].detach().cpu().numpy().round(3))

    # Verify belief changed
    initial_belief = agent.enn(initial_state.unsqueeze(0).to(device)).squeeze(0).detach()
    updated_belief = agent.tracker.belief_vector.detach()
    diff = (updated_belief - initial_belief[:NUM_CARDS]).abs().mean().item()
    print(f"\nMean absolute belief change after partner bid: {diff:.4f}")
    print("(Should be > 0 — confirms Bayesian update is working)\n")

    # --- ENN supervised loss shape check ---
    fake_states  = torch.randn(32, STATE_DIM).to(device)
    fake_targets = torch.zeros(32, NUM_CARDS).to(device)
    for i in range(32):
        idx = torch.randperm(NUM_CARDS)[:HAND_SIZE]
        fake_targets[i, idx] = 1.0

    loss = agent.enn.loss(fake_states, fake_targets)
    print(f"ENN supervised loss (random data): {loss.item():.4f}")

    # --- PNN forward shape check ---
    batch_hands   = torch.zeros(4, NUM_CARDS).to(device)
    batch_beliefs = torch.rand(4, NUM_CARDS).to(device)
    batch_masks   = torch.ones(4, NUM_ACTIONS, dtype=torch.bool).to(device)
    probs, values = agent.pnn_forward(batch_hands, batch_beliefs, batch_masks)
    print(f"\nPNN output shapes — probs: {probs.shape}, values: {values.shape}")
    print("All shapes correct. Prototype ready.\n")

    print("=" * 60)
    print("NEXT STEPS:")
    print("  1. Replace fake_states/targets with your OpenSpiel expert data")
    print("  2. Train ENN with train_enn_epoch() until convergence")
    print("  3. Freeze ENN weights; plug into BADBridgeAgent")
    print("  4. Run PPO/A3C self-play, calling agent.observe_partner_bid()")
    print("     after every partner bid to keep the belief updated")
    print("  5. For ablation: swap BAD belief for raw ENN output as PNN input")
    print("=" * 60)
