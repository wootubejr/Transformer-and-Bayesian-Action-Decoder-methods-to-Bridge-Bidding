"""Generate bridge self-play data for model-vs-model evaluation.

Two play modes:
  - single table: team1 sits N/S, team2 sits E/W, on one deal.
  - duplicate: the same deal is played twice. Team1 sits N/S at table A
    and E/W at table B (and vice versa for team2). The per-board swing
    (team1 raw points at table A plus team1 raw points at table B) is
    converted to IMPs on the standard WBF 0–24 scale.

Each team's params can be supplied as either a .pkl (pickled param pytree)
or a .npz (flat arrays rebuilt into the model's param pytree). 
Mix and match: pkl-vs-npz, npz-vs-npz, or pkl-vs-pkl all work.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from pgx.bridge_bidding import BridgeBidding, _state_to_pbn, _value_to_dds_tricks
from brl.models import make_forward_pass
# from brl.src.duplicate import _imp_reward
import haiku as hk

# Restating the originals without brl dependency for rewards
def _imp_reward(
    table_a_reward: jnp.ndarray, table_b_reward: jnp.ndarray
) -> jnp.ndarray:
    """Convert score reward to IMP reward

    >>> table_a_reward = jnp.array([0, 0, 0, 0])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> table_a_reward = jnp.array([0, 0, 0, 0])
    >>> table_b_reward = jnp.array([100, 100, -100, -100])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([ 3.,  3., -3., -3.], dtype=float32)
    >>> table_a_reward = jnp.array([-100, -100, 100, 100])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([-3., -3.,  3.,  3.], dtype=float32)
    >>> table_a_reward = jnp.array([-100, -100, 100, 100])
    >>> table_b_reward = jnp.array([100, 100, -100, -100])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> table_a_reward = jnp.array([-3500, -3500, 3500, 3500])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([-23., -23.,  23.,  23.], dtype=float32)
    >>> table_a_reward = jnp.array([2000, 2000, -2000, -2000])
    >>> table_b_reward = jnp.array([2000, 2000, -2000, -2000])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([ 24.,  24., -24., -24.], dtype=float32)
    """
    # fmt: off
    IMP_LIST = jnp.array([20, 50, 90, 130, 170,
                          220, 270, 320, 370, 430,
                          500, 600, 750, 900, 1100,
                          1300, 1500, 1750, 2000, 2250,
                          2500, 3000, 3500, 4000], dtype=jnp.float32)
    # fmt: on
    win = jax.lax.cond(
        table_a_reward[0] + table_b_reward[0] >= 0, lambda: 1, lambda: -1
    )

    def condition_fun(imp_diff):
        imp, difference_point = imp_diff
        return (difference_point >= IMP_LIST[imp]) & (imp < 24)

    def body_fun(imp_diff):
        imp, difference_point = imp_diff
        imp += 1
        return (imp, difference_point)

    imp, difference_point = jax.lax.while_loop(
        condition_fun,
        body_fun,
        (0, abs(table_a_reward[0] + table_b_reward[0])),
    )
    return jnp.array([imp * win, imp * win, -imp * win, -imp * win], dtype=jnp.float32)

SEATS = ["N", "E", "S", "W"]
BID_NAMES = ["Pass", "X", "XX"] + [
    f"{lvl}{s}" for lvl in range(1, 8) for s in ["C", "D", "H", "S", "NT"]
]

# ---------------------------------------------------------------------------
# Deal / auction decoding helpers (unchanged logic from original, trimmed)
# ---------------------------------------------------------------------------

def pbn_to_deal_dict(pbn: str) -> Dict[str, List[str]]:
    first_seat, hands_str = pbn.split(":")
    order = SEATS[SEATS.index(first_seat):] + SEATS[:SEATS.index(first_seat)]
    deal = {}
    for seat, hand_str in zip(order, hands_str.strip().split()):
        cards = []
        for suit, ranks in zip("SHDC", hand_str.split(".")):
            cards.extend(f"{suit}{r}" for r in ranks)
        deal[seat] = cards
    return deal


def decode_vulnerability(vul_ns: bool, vul_ew: bool) -> str:
    if vul_ns and vul_ew:
        return "Both"
    if vul_ns:
        return "NS"
    if vul_ew:
        return "EW"
    return "None"


def extract_dds_table(state) -> Optional[Dict]:
    """DDS tricks[seat][denom] — seat in N/E/S/W, denom in C/D/H/S/NT."""
    try:
        tricks_flat = np.array(_value_to_dds_tricks(state._dds_val))
        denoms = ["C", "D", "H", "S", "NT"]
        return {
            seat: {d: int(tricks_flat[pos * 5 + i]) for i, d in enumerate(denoms)}
            for pos, seat in enumerate(SEATS)
        }
    except Exception:
        return None


def interpret_auction(bids) -> str:
    if all(a == 0 for _, a in bids):
        return "Passed Out"

    last_bid_name, last_bid_seat = None, None
    doubled, redoubled = False, False

    for seat, action_id in bids:
        if action_id >= 3:
            last_bid_name = BID_NAMES[action_id]
            last_bid_seat = seat
            doubled = redoubled = False
        elif action_id == 1:
            doubled, redoubled = True, False
        elif action_id == 2:
            doubled, redoubled = False, True

    if last_bid_name is None:
        return "Passed Out"

    denom = last_bid_name[1:]
    pair = {"N", "S"} if last_bid_seat in ("N", "S") else {"E", "W"}
    declarer = last_bid_seat
    for seat, action_id in bids:
        if seat in pair and action_id >= 3 and BID_NAMES[action_id][1:] == denom:
            declarer = seat
            break

    suffix = "-XX" if redoubled else ("-X" if doubled else "")
    return f"{last_bid_name}{suffix} by {declarer}"


def dd_evaluate(contract_str: str, dds: Dict, vulnerability: str) -> Optional[Dict]:
    if contract_str == "Passed Out":
        return {"result": "Passed Out", "score": 0}

    parts = contract_str.split(" by ")
    declarer = parts[1]
    bid_part = parts[0]
    rdbl = "-XX" in bid_part
    dbl = "-X" in bid_part and not rdbl
    bid_part = bid_part.replace("-XX", "").replace("-X", "")
    level, denom = int(bid_part[0]), bid_part[1:]

    tricks = dds[declarer][denom]
    needed = 6 + level
    diff = tricks - needed
    vul = vulnerability == "Both" or (vulnerability == "NS" and declarer in "NS") or (
        vulnerability == "EW" and declarer in "EW"
    )

    if diff >= 0:
        result_str = f"Made {level}" + (f"+{diff}" if diff > 0 else "")
    else:
        result_str = f"Down {-diff}"

    return {
        "contract": contract_str,
        "dd_tricks": tricks,
        "tricks_needed": needed,
        "result": result_str,
        "is_vulnerable": vul,
        "doubled": dbl,
        "redoubled": rdbl,
    }


# ---------------------------------------------------------------------------
# Model loading — team1 (pkl, new algorithm) vs team2 (npz, baseline)
# ---------------------------------------------------------------------------

def make_dummy_observation(env) -> jnp.ndarray:
    state = env.init(jax.random.PRNGKey(0))
    return state.observation.astype(jnp.float32)

def load_npz_leaves(npz_path: str):
    archive = np.load(npz_path)
    keys = sorted(
        [k for k in archive.files if k.startswith("arr_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return [archive[k] for k in keys]

def rebuild_params_from_npz(forward_pass, env, npz_path: str):
    """Rebuild haiku-style param pytree from flat .npz arrays."""
    dummy_obs = make_dummy_observation(env)
    skeleton_params = forward_pass.init(jax.random.PRNGKey(123), dummy_obs)
    skeleton_leaves, skeleton_treedef = jax.tree_util.tree_flatten(skeleton_params)
    npz_leaves = load_npz_leaves(npz_path)

    if len(npz_leaves) != len(skeleton_leaves):
        raise ValueError(
            f"NPZ has {len(npz_leaves)} leaves; skeleton has {len(skeleton_leaves)}."
        )

    rebuilt = []
    for i, (saved, skel) in enumerate(zip(npz_leaves, skeleton_leaves)):
        saved_arr = np.asarray(saved)
        skel_arr = np.asarray(skel)
        if saved_arr.shape != skel_arr.shape:
            raise ValueError(
                f"Shape mismatch at leaf {i}: {saved_arr.shape} vs {skel_arr.shape}."
            )
        rebuilt.append(jnp.asarray(saved_arr, dtype=skel_arr.dtype))

    return jax.tree_util.tree_unflatten(skeleton_treedef, rebuilt)

def load_pkl_params(pkl_path: str):
    """Load a pickled param pytree (your new .pkl model)."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_team_params(
    forward_pass,
    env,
    pkl_path: Optional[str] = None,
    npz_path: Optional[str] = None,
    label: str = "team",
):
    """Load params from either a .pkl or a .npz. Exactly one must be given."""
    if bool(pkl_path) == bool(npz_path):
        raise ValueError(
            f"{label}: specify exactly one of pkl_path / npz_path "
            f"(got pkl={pkl_path!r}, npz={npz_path!r})."
        )
    if pkl_path:
        params = load_pkl_params(pkl_path)
        print(f"Loaded {label} (pkl): {pkl_path}")
    else:
        params = rebuild_params_from_npz(forward_pass, env, npz_path)
        print(f"Loaded {label} (npz): {npz_path}")
    return params

# ---------------------------------------------------------------------------
# Policy sampling
# ---------------------------------------------------------------------------

def sample_action(forward_pass, params, obs, mask, rng, deterministic: bool = False):
    """Sample (or argmax) from a masked categorical policy."""
    logits, _ = forward_pass.apply(params, obs.astype(jnp.float32))
    logits = np.array(logits)
    logits[~mask] = -1e18
    if deterministic:
        return int(np.argmax(logits)), rng
    rng, sk = jax.random.split(rng)
    action = int(distrax.Categorical(logits=jnp.array(logits)).sample(seed=sk))
    return action, rng

# ---------------------------------------------------------------------------
# One table of play
# ---------------------------------------------------------------------------

def play_one_table(
    env,
    init_fn,
    step_fn,
    key,
    pair_01_forward_pass,
    pair_01_params,
    pair_23_forward_pass,
    pair_23_params,
    rng,
    deterministic: bool = False,
):
    """Play a single deal. Player ids {0, 1} use the first model, {2, 3} the
    second. This is the convention _imp_reward expects.

    Returns (record_dict, rng). The record's raw_rewards_by_player is always
    in player-id order 0..3 — no reindexing needed later.
    """
    state = init_fn(key)

    shuffled = np.array(state._shuffled_players)
    dealer_seat = SEATS[int(state._dealer)]
    vul = decode_vulnerability(bool(state._vul_NS), bool(state._vul_EW))
    pbn = str(_state_to_pbn(state))
    inverse_shuffled = np.argsort(shuffled)  # player_id -> seat index

    bids = []
    while not bool(state.terminated):
        cur = int(state.current_player)
        seat = SEATS[int(inverse_shuffled[cur])]
        mask = np.array(state.legal_action_mask)

        if cur in (0, 1):
            action, rng = sample_action(
                pair_01_forward_pass, pair_01_params,
                state.observation, mask, rng, deterministic,
            )
        else:
            action, rng = sample_action(
                pair_23_forward_pass, pair_23_params,
                state.observation, mask, rng, deterministic,
            )

        bids.append((seat, action))
        state = step_fn(state, action)

    contract = interpret_auction(bids)
    dds = extract_dds_table(state)
    reward_arr = np.array(state.rewards)
    rewards_by_seat = {
        SEATS[int(inverse_shuffled[p])]: float(reward_arr[p]) for p in range(4)
    }
    # Seats held by the {0,1} partnership at this table.
    pair_01_seats = [SEATS[int(inverse_shuffled[p])] for p in (0, 1)]

    dd_eval = dd_evaluate(contract, dds, vul) if dds else None

    record = {
        "dealer": dealer_seat,
        "vulnerability": vul,
        "deal": pbn_to_deal_dict(pbn),
        "pbn": pbn,
        "auction": [{"seat": s, "action": a, "bid": BID_NAMES[a]} for s, a in bids],
        "final_contract": contract,
        "score_reward": rewards_by_seat,
        "pair_01_seats": pair_01_seats,
        # Raw pgx rewards indexed by player id (0..3). _imp_reward consumes
        # this directly without reindexing.
        "raw_rewards_by_player": [float(x) for x in reward_arr],
        **({"dds": dds} if dds else {}),
        **({"dd_evaluation": dd_eval} if dd_eval else {}),
    }
    return record, rng


# ---------------------------------------------------------------------------
# Generate — single-table model vs model
# ---------------------------------------------------------------------------

def generate_model_vs_model(
    num_deals: int,
    team1_pkl: Optional[str] = None,
    team1_npz: Optional[str] = None,
    team2_pkl: Optional[str] = None,
    team2_npz: Optional[str] = None,
    seed: int = 42,
    team1_activation: str = "relu",
    team1_model_type: str = "DeepMind",
    team2_activation: str = "relu",
    team2_model_type: str = "DeepMind",
    dds_file: Optional[str] = None,
    deterministic: bool = False,
    verbose_index: Optional[int] = None,
):
    env = BridgeBidding(dds_file) if dds_file else BridgeBidding()
    init_fn, step_fn = jax.jit(env.init), jax.jit(env.step)

    team1_forward_pass = make_forward_pass(team1_activation, team1_model_type)
    team2_forward_pass = make_forward_pass(team2_activation, team2_model_type)

    team1_params = load_team_params(
        team1_forward_pass, env, team1_pkl, team1_npz, label="team1"
    )
    team2_params = load_team_params(
        team2_forward_pass, env, team2_pkl, team2_npz, label="team2"
    )

    rng = jax.random.PRNGKey(seed)
    records = []

    for i in range(num_deals):
        rng, deal_key = jax.random.split(rng)
        # team1 = pair {0,1}, team2 = pair {2,3}
        record, rng = play_one_table(
            env, init_fn, step_fn, deal_key,
            team1_forward_pass, team1_params,
            team2_forward_pass, team2_params,
            rng=rng,
            deterministic=deterministic,
        )
        record["board_id"] = i
        # Convenience: team1_reward from player 0 (partners always share).
        record["team1_reward"] = record["raw_rewards_by_player"][0]
        # Rename pair_01_seats -> team1_seats for the inspector.
        record["team1_seats"] = record.pop("pair_01_seats")
        records.append(record)
        
        # Verbose printing in duplicate
        if (i + 1) % max(1, num_deals // 5) == 0:
            print(f"{i + 1}/{num_deals} deals done")

    return records


# ---------------------------------------------------------------------------
# Generate — duplicate-team model vs model
# ---------------------------------------------------------------------------

def generate_duplicate_model_vs_model(
    num_deals: int,
    team1_pkl: Optional[str] = None,
    team1_npz: Optional[str] = None,
    team2_pkl: Optional[str] = None,
    team2_npz: Optional[str] = None,
    seed: int = 42,
    team1_activation: str = "relu",
    team1_model_type: str = "DeepMind",
    team2_activation: str = "relu",
    team2_model_type: str = "DeepMind",
    dds_file: Optional[str] = None,
    deterministic: bool = False,
    verbose_index: Optional[int] = None,
):
    """Duplicate-teams evaluation.

    For each deal we run two tables:
      Table A: team1 holds player_ids {0, 1}, team2 holds {2, 3}.
      Table B: team1 holds player_ids {2, 3}, team2 holds {0, 1}  (swapped).

    The same hands are dealt at both tables (same PRNG key into env.init),
    so any advantage/disadvantage from card strength is cancelled — only
    bidding skill differences remain.

    The per-board team1 score is converted from raw pgx rewards to IMPs
    using the repo's own `_imp_reward` function (WBF standard table).
    Table B's rewards are reindexed so the `{0,1}` = team1 convention
    expected by `_imp_reward` holds at both tables.
    """
    env = BridgeBidding(dds_file) if dds_file else BridgeBidding()
    init_fn, step_fn = jax.jit(env.init), jax.jit(env.step)

    team1_forward_pass = make_forward_pass(team1_activation, team1_model_type)
    team2_forward_pass = make_forward_pass(team2_activation, team2_model_type)

    team1_params = load_team_params(
        team1_forward_pass, env, team1_pkl, team1_npz, label="team1"
    )
    team2_params = load_team_params(
        team2_forward_pass, env, team2_pkl, team2_npz, label="team2"
    )

    rng = jax.random.PRNGKey(seed)
    records = []

    for i in range(num_deals):
        rng, deal_key = jax.random.split(rng)

        # Both tables use the same deal_key => identical cards / vul / dealer.
        # Table A: team1 = pair {0,1}, team2 = pair {2,3}.
        record_a, rng = play_one_table(
            env, init_fn, step_fn, deal_key,
            team1_forward_pass, team1_params,
            team2_forward_pass, team2_params,
            rng=rng,
            deterministic=deterministic,
        )
        # Table B: swap — team2 plays the {0,1} pair, team1 plays {2,3}.
        record_b, rng = play_one_table(
            env, init_fn, step_fn, deal_key,
            team2_forward_pass, team2_params,  
            team1_forward_pass, team1_params,   
            rng=rng,
            deterministic=deterministic,
        )

        # At both tables, player ids {0,1} and {2,3} are the two partnerships exactly as _imp_reward expects. 
        # At table A, {0,1} IS team1; at table B, {0,1} is team2

        raw_a = jnp.array(record_a["raw_rewards_by_player"], dtype=jnp.float32)
        raw_b = jnp.array(record_b["raw_rewards_by_player"], dtype=jnp.float32)
        imp_per_player = np.array(_imp_reward(raw_a, -raw_b))
        team1_imps = float(imp_per_player[0])

        team1_raw_at_a = float(raw_a[0])         
        team1_raw_at_b = float(-raw_b[0])      
        board_team1_raw = team1_raw_at_a + team1_raw_at_b

        # Annotate the per-table records with team1's seats pair_01_seats is the seats that held player-ids
        seats_of_01_at_a = record_a.pop("pair_01_seats")
        seats_of_01_at_b = record_b.pop("pair_01_seats")
        record_a["team1_seats"] = seats_of_01_at_a
        record_a["team2_seats"] = [s for s in SEATS if s not in seats_of_01_at_a]
        record_b["team2_seats"] = seats_of_01_at_b
        record_b["team1_seats"] = [s for s in SEATS if s not in seats_of_01_at_b]
        record_a["team1_reward"] = team1_raw_at_a
        record_b["team1_reward"] = team1_raw_at_b

        records.append({
            "board_id": i,
            "table_a": record_a,
            "table_b": record_b,
            "team1_imps": team1_imps,
            "team1_raw_swing": board_team1_raw,
        })

        if verbose_index is not None and i == verbose_index:
            print(f"\n Sample board evaluation")
            print(f"Open room contract: {record_a['final_contract']}  "
                  f"(team1 raw {team1_raw_at_a:+})")
            print(f"Closed room contract: {record_b['final_contract']}  "
                  f"(team1 raw {team1_raw_at_b:+})")
            print(f"Team1 board: raw swing {board_team1_raw:+}  "
                  f"→ {team1_imps:+.0f} IMPs\n")

        if (i + 1) % max(1, num_deals // 5) == 0:
            print(f"{i + 1}/{num_deals} boards done")

    # Summary stats on team1 IMPs per board
    imps = np.array([r["team1_imps"] for r in records])
    summary = {
        "num_boards": int(len(imps)),
        "team1_imps_per_board_mean": float(imps.mean()),
        "team1_imps_per_board_se": float(imps.std(ddof=1) / np.sqrt(len(imps))) if len(imps) > 1 else 0.0,
        "team1_win_rate": float((imps > 0).mean()),
        "team1_loss_rate": float((imps < 0).mean()),
        "team1_push_rate": float((imps == 0).mean()),
    }
    print(
        f"Duplicate summary: {summary['team1_imps_per_board_mean']:+.3f} "
        f"± {summary['team1_imps_per_board_se']:.3f} IMPs/board "
        f"over {summary['num_boards']} boards "
        f"(wins {summary['team1_win_rate']:.1%}, "
        f"losses {summary['team1_loss_rate']:.1%}, "
        f"pushes {summary['team1_push_rate']:.1%})"
    )
    return records, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Model-vs-model bridge self-play.")
    p.add_argument("--mode", choices=["single", "duplicate"], default="duplicate",
                   help="single = one table per board; duplicate = two tables with swapped seats.")
    p.add_argument("--num_deals", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)

    t1 = p.add_mutually_exclusive_group(required=True)
    t1.add_argument("--team1_pkl", type=str, help="Team 1 params as a .pkl file.")
    t1.add_argument("--team1_npz", type=str, help="Team 1 params as an .npz file.")

    t2 = p.add_mutually_exclusive_group(required=True)
    t2.add_argument("--team2_pkl", type=str, help="Team 2 params as a .pkl file.")
    t2.add_argument("--team2_npz", type=str, help="Team 2 params as an .npz file.")

    p.add_argument("--team1_activation", type=str, default="relu")
    p.add_argument("--team1_model_type", type=str, default="DeepMind")
    p.add_argument("--team2_activation", type=str, default="relu")
    p.add_argument("--team2_model_type", type=str, default="DeepMind")
    p.add_argument("--team1_name", type=str, default="Team 1",
                   help="Display name for team 1 (used in the inspector).")
    p.add_argument("--team2_name", type=str, default="Team 2",
                   help="Display name for team 2 (used in the inspector).")
    p.add_argument("--dds_file", type=str, default="dds/dds_results_500K.npy")
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax instead of sampling for both policies.")
    p.add_argument("--verbose_index", type=int, default=0,
                   help="Deal index to print in detail (set <0 to disable).")
    p.add_argument("--output", type=str, default="play_data.json")
    return p.parse_args()


def main():
    args = parse_args()
    verbose_index = args.verbose_index if args.verbose_index is not None and args.verbose_index >= 0 else None

    common_kwargs = dict(
        num_deals=args.num_deals,
        team1_pkl=args.team1_pkl,
        team1_npz=args.team1_npz,
        team2_pkl=args.team2_pkl,
        team2_npz=args.team2_npz,
        seed=args.seed,
        team1_activation=args.team1_activation,
        team1_model_type=args.team1_model_type,
        team2_activation=args.team2_activation,
        team2_model_type=args.team2_model_type,
        dds_file=args.dds_file,
        deterministic=args.deterministic,
        verbose_index=verbose_index,
    )

    if args.mode == "single":
        records = generate_model_vs_model(**common_kwargs)
        payload = {
            "mode": "single",
            "team_names": {"team1": args.team1_name, "team2": args.team2_name},
            "logs": records,
        }
    else:
        records, summary = generate_duplicate_model_vs_model(**common_kwargs)
        payload = {
            "mode": "duplicate",
            "team_names": {"team1": args.team1_name, "team2": args.team2_name},
            "summary": summary,
            "logs": records,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
