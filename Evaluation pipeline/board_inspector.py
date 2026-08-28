"""
Bridge Board Inspector
======================
Opens your self-play JSON in the browser as an interactive board viewer.
Click through deals, see all four hands, the auction, contract, DDS table,
and rewards.

Usage:
  python board_inspector.py selfplay_data.json
  python board_inspector.py selfplay_data.json --port 8050
"""

import argparse
import json
import http.server
import socketserver
import threading
import webbrowser
import os
import sys
from urllib.parse import urlparse, parse_qs

# ─────────────────────────────────────────────────────────────────────────────
# USER INPUTS — edit these to change the team display names without touching
# the JSON or using the command line. Leave as None to fall back to whatever
# names were stored in the JSON (or "Team 1"/"Team 2" if missing).
# ─────────────────────────────────────────────────────────────────────────────
TEAM1_NAME_OVERRIDE = None    # e.g. "RL+FSP"
TEAM2_NAME_OVERRIDE = None    # e.g. "SL baseline"

# ── Suit symbols and sorting ──
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
SUIT_ORDER = ["S", "H", "D", "C"]
RANK_ORDER = "AKQJT98765432"


def sort_hand_by_suit(cards):
    """Group cards by suit in S/H/D/C order, ranks high-to-low."""
    suits = {s: [] for s in SUIT_ORDER}
    for card in cards:
        suit = card[0]
        rank = card[1:]
        suits[suit].append(rank)
    for s in SUIT_ORDER:
        suits[s].sort(key=lambda r: RANK_ORDER.index(r))
    return suits


def hand_html(cards):
    """Render a hand as suited lines."""
    suits = sort_hand_by_suit(cards)
    lines = []
    for s in SUIT_ORDER:
        color = "#e74c3c" if s in ("H", "D") else "#ecf0f1"
        symbol = SUIT_SYMBOLS[s]
        ranks = " ".join(suits[s]) if suits[s] else "—"
        lines.append(f'<div class="suit-line">'
                     f'<span class="suit-symbol" style="color:{color}">{symbol}</span>'
                     f'<span class="ranks">{ranks}</span></div>')
    return "\n".join(lines)


def auction_html(auction):
    """Render the auction as a bidding table."""
    if not auction:
        return '<div class="no-auction">No auction data</div>'

    # Determine starting seat
    seat_order = ["N", "E", "S", "W"]
    first_seat = auction[0].get("seat", "N")
    start_idx = seat_order.index(first_seat)
    # Rotate header to start with dealer
    headers = seat_order[start_idx:] + seat_order[:start_idx]

    rows_html = '<tr>' + ''.join(f'<th>{s}</th>' for s in headers) + '</tr>\n'

    # Headers are already rotated to start with dealer, so no padding needed.
    cells = []

    for bid_entry in auction:
        bid_name = bid_entry.get("bid", "?")
        css_class = "bid-pass" if bid_name == "Pass" else \
                    "bid-double" if bid_name == "X" else \
                    "bid-redouble" if bid_name == "XX" else "bid-normal"
        # Color code by denomination
        if len(bid_name) >= 2 and bid_name not in ("Pass", "X", "XX"):
            denom = bid_name[1:]
            if denom in ("H",):
                css_class += " denom-heart"
            elif denom in ("D",):
                css_class += " denom-diamond"
            elif denom in ("S",):
                css_class += " denom-spade"
            elif denom in ("C",):
                css_class += " denom-club"
            elif denom == "NT":
                css_class += " denom-nt"
        cells.append(f'<span class="{css_class}">{bid_name}</span>')

    # Build rows of 4
    while len(cells) % 4 != 0:
        cells.append('')
    for i in range(0, len(cells), 4):
        row_cells = cells[i:i+4]
        rows_html += '<tr>' + ''.join(f'<td>{c}</td>' for c in row_cells) + '</tr>\n'

    return f'<table class="auction-table">{rows_html}</table>'


def dda_html(dda):
    """Render the DDA table."""
    if not dda:
        return '<div class="no-dda">No DDA data</div>'

    denoms = ["C", "D", "H", "S", "NT"]
    header = '<tr><th></th>' + ''.join(
        f'<th>{SUIT_SYMBOLS.get(d, d)}</th>' for d in denoms
    ) + '</tr>'
    rows = ""
    for seat in ["N", "E", "S", "W"]:
        if seat not in dda:
            continue
        cells = ''.join(
            f'<td>{dda[seat].get(d, "—")}</td>' for d in denoms
        )
        rows += f'<tr><th>{seat}</th>{cells}</tr>'

    return f'<table class="dda-table">{header}{rows}</table>'


def dd_eval_html(dd_eval):
    """Render the DD evaluation result for a board."""
    if not dd_eval:
        return ''

    result = dd_eval.get("result", "?")
    contract = dd_eval.get("contract", "?")
    tricks = dd_eval.get("dd_tricks", "?")
    needed = dd_eval.get("tricks_needed", "?")
    vul = dd_eval.get("is_vulnerable", False)
    dbl = dd_eval.get("doubled", False)
    rdbl = dd_eval.get("redoubled", False)

    status_class = "dd-made" if "Made" in str(result) else "dd-down"
    modifier = " (XX)" if rdbl else (" (X)" if dbl else "")

    return (f'<div class="dd-eval">'
            f'<span class="{status_class}">{result}</span>{modifier}<br>'
            f'<span class="dd-detail">DD tricks: {tricks} / {needed} needed</span><br>'
            f'<span class="dd-detail">Vulnerable: {"Yes" if vul else "No"}</span>'
            f'</div>')


def seat_teams_map(team1_seats, team_names):
    """Return {seat: {'name': ..., 'id': 't1'|'t2'}} for all four seats."""
    t1 = set(team1_seats or [])
    m = {}
    for s in ["N", "E", "S", "W"]:
        if s in t1:
            m[s] = {"name": team_names.get("team1", "Team 1"), "id": "t1"}
        else:
            m[s] = {"name": team_names.get("team2", "Team 2"), "id": "t2"}
    return m


def build_viewable_entries(data):
    """Flatten the JSON into two buckets based on who sits N/S.

    Duplicate mode:
      Every duplicate board produces two table records. Team 1 sits N/S at
      exactly one of them (the other has team 1 in E/W). We bucket all 2N
      table-records into two lists of N entries each:
        bucket_t1_ns: entries where team 1 sits N/S.
        bucket_t2_ns: entries where team 2 sits N/S.
      Each entry carries a pointer to its partner entry (same board_id, in
      the other bucket) so the 'Other Tables' button can swap the view.

    Single mode:
      Everything goes in bucket_t1_ns since there's no companion table.
    """
    mode = data.get("mode", "single")

    json_team_names = data.get("team_names", {})
    team_names = {
        "team1": TEAM1_NAME_OVERRIDE or json_team_names.get("team1", "Team 1"),
        "team2": TEAM2_NAME_OVERRIDE or json_team_names.get("team2", "Team 2"),
    }

    logs = data.get("logs", [])

    # Build one entry per table-record, keeping enough pairing info to cross-link.
    all_entries = []

    if mode == "duplicate":
        for board in logs:
            a = board["table_a"]
            b = board["table_b"]
            imps = board.get("team1_imps")
            raw_swing = board.get("team1_raw_swing")
            bid = board.get("board_id")

            # Assign fixed room labels: table A = Open Room, table B = Closed Room.
            # (Matches tournament convention — Open Room is always the first table.)
            for rec, room in [(a, "Open Room"), (b, "Closed Room")]:
                e = dict(rec)
                e["board_id"] = bid
                e["room"] = room
                e["team1_imps"] = imps
                e["team1_raw_swing"] = raw_swing
                e["team_names"] = team_names
                # Which team is sitting N/S at this specific table?
                t1_seats = set(rec.get("team1_seats") or [])
                team1_is_ns = ("N" in t1_seats) and ("S" in t1_seats)
                e["team1_is_ns"] = team1_is_ns
                all_entries.append(e)
    else:
        for board in logs:
            e = dict(board)
            e["room"] = None
            e["team_names"] = team_names
            # In single mode we assume "team1_seats" exists; if not, default to NS.
            t1_seats = set(board.get("team1_seats") or ["N", "S"])
            e["team1_is_ns"] = ("N" in t1_seats) and ("S" in t1_seats)
            all_entries.append(e)

    # Bucket by team position.
    bucket_t1_ns = [e for e in all_entries if e["team1_is_ns"]]
    bucket_t2_ns = [e for e in all_entries if not e["team1_is_ns"]]

    # Pair across buckets by board_id → companion index. Since each board has
    # exactly one entry in each bucket (in duplicate mode), we can build a
    # simple map from (board_id, bucket) to its position in the flattened
    # "all buckets" list (bucket_t1_ns followed by bucket_t2_ns).
    bid_to_t2_idx = {e["board_id"]: len(bucket_t1_ns) + i
                     for i, e in enumerate(bucket_t2_ns)}
    bid_to_t1_idx = {e["board_id"]: i for i, e in enumerate(bucket_t1_ns)}

    for i, e in enumerate(bucket_t1_ns):
        e["entry_idx"] = i
        e["companion_idx"] = bid_to_t2_idx.get(e["board_id"])
    for i, e in enumerate(bucket_t2_ns):
        e["entry_idx"] = len(bucket_t1_ns) + i
        e["companion_idx"] = bid_to_t1_idx.get(e["board_id"])

    # Combined list: all bucket-1 entries, then all bucket-2 entries.
    entries = bucket_t1_ns + bucket_t2_ns

    return entries, mode, team_names, len(bucket_t1_ns), len(bucket_t2_ns)


def compact_contract_line(entry, entries):
    """Build a one-line 'contract, direction, score' string for the companion.

    Used on the main view to show what happened at the other table without
    leaving the current screen.
    """
    comp_idx = entry.get("companion_idx")
    if comp_idx is None or not (0 <= comp_idx < len(entries)):
        return None
    comp = entries[comp_idx]
    contract = comp.get("final_contract") or "—"
    # Team 1's raw points at the companion table.
    raw = comp.get("team1_reward")
    if raw is None:
        # Fall back to reading per-seat rewards.
        sr = comp.get("score_reward", {})
        seats = comp.get("team1_seats") or []
        raw = sum(sr.get(s, 0) for s in seats) / max(len(seats), 1)
    if raw is None:
        score_str = "?"
    else:
        n = int(round(raw))
        score_str = f"{'+' if n > 0 else ''}{n}"
    return f"{contract}, {score_str}"


def build_html(data):
    """Build the complete single-page app HTML."""
    entries, mode, team_names, n_bucket1, n_bucket2 = build_viewable_entries(data)

    # Pre-render each entry's hand HTML and seat→team mapping server-side
    board_renders = []
    for e in entries:
        deal = e.get("deal", {})
        board_renders.append({
            "north": hand_html(deal.get("N", [])),
            "east": hand_html(deal.get("E", [])),
            "south": hand_html(deal.get("S", [])),
            "west": hand_html(deal.get("W", [])),
            "auction": auction_html(e.get("auction", [])),
            "dda": dda_html(e.get("dds", e.get("dda", None))),
            "dd_eval": dd_eval_html(e.get("dd_evaluation", None)),
            "seat_teams": seat_teams_map(e.get("team1_seats"), team_names),
            "companion_line": compact_contract_line(e, entries),
        })

    entries_json = json.dumps(entries)
    renders_json = json.dumps(board_renders)
    mode_json = json.dumps(mode)
    team_names_json = json.dumps(team_names)
    bucket_sizes_json = json.dumps({"t1_ns": n_bucket1, "t2_ns": n_bucket2})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bridge Board Inspector</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Libre+Baskerville:wght@400;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: #1a1a2e;
    color: #ecf0f1;
    font-family: 'JetBrains Mono', monospace;
    min-height: 100vh;
    padding: 20px;
  }}

  .header {{
    text-align: center;
    margin-bottom: 24px;
  }}
  .header h1 {{
    font-family: 'Libre Baskerville', serif;
    font-size: 1.6em;
    color: #e2b45a;
    letter-spacing: 2px;
  }}

  .nav {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin-bottom: 24px;
  }}
  .nav button {{
    background: #16213e;
    color: #e2b45a;
    border: 1px solid #e2b45a44;
    padding: 8px 20px;
    font-family: inherit;
    font-size: 0.9em;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s;
  }}
  .nav button:hover {{
    background: #e2b45a;
    color: #1a1a2e;
  }}
  .nav button:disabled {{
    opacity: 0.3;
    cursor: default;
  }}
  .nav button:disabled:hover {{
    background: #16213e;
    color: #e2b45a;
  }}
  .board-num {{
    font-size: 1.1em;
    min-width: 120px;
    text-align: center;
  }}

  .meta-bar {{
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-bottom: 20px;
    font-size: 0.85em;
    color: #8899aa;
  }}
  .meta-bar span {{
    background: #16213e;
    padding: 4px 14px;
    border-radius: 4px;
    border: 1px solid #ffffff0a;
  }}
  .meta-bar .contract {{
    color: #e2b45a;
    font-weight: 600;
  }}

  .main-layout {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    max-width: 960px;
    margin: 0 auto;
  }}

  /* ── Bridge table ── */
  .table-area {{
    display: grid;
    grid-template-areas:
      ".     north ."
      "west  center east"
      ".     south .";
    grid-template-columns: 1fr auto 1fr;
    grid-template-rows: auto auto auto;
    gap: 8px;
    align-items: center;
    justify-items: center;
  }}

  .hand {{
    background: #16213e;
    border: 1px solid #ffffff0a;
    border-radius: 6px;
    padding: 12px 16px;
    min-width: 150px;
  }}
  .hand-north {{ grid-area: north; }}
  .hand-east  {{ grid-area: east; }}
  .hand-south {{ grid-area: south; }}
  .hand-west  {{ grid-area: west; }}

  .hand-label {{
    font-size: 0.7em;
    color: #e2b45a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
    text-align: center;
  }}

  .suit-line {{
    display: flex;
    gap: 6px;
    font-size: 0.9em;
    line-height: 1.6;
  }}
  .suit-symbol {{
    font-weight: 600;
    width: 16px;
    text-align: center;
  }}

  .center-marker {{
    grid-area: center;
    width: 60px;
    height: 60px;
    border: 1px solid #e2b45a33;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7em;
    color: #e2b45a88;
    background: #16213e;
  }}

  /* ── Right panel ── */
  .right-panel {{
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}

  .panel-section {{
    background: #16213e;
    border: 1px solid #ffffff0a;
    border-radius: 6px;
    padding: 14px 16px;
  }}
  .panel-title {{
    font-size: 0.7em;
    color: #e2b45a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
  }}

  /* Auction table */
  .auction-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
  }}
  .auction-table th {{
    color: #8899aa;
    font-weight: 400;
    padding: 2px 8px;
    border-bottom: 1px solid #ffffff0a;
  }}
  .auction-table td {{
    padding: 3px 8px;
    text-align: center;
  }}
  .bid-pass {{ color: #667788; }}
  .bid-double {{ color: #e74c3c; font-weight: 600; }}
  .bid-redouble {{ color: #3498db; font-weight: 600; }}
  .bid-normal {{ color: #ecf0f1; }}
  .denom-heart, .denom-diamond {{ color: #e74c3c !important; }}
  .denom-spade, .denom-club {{ color: #ecf0f1 !important; }}
  .denom-nt {{ color: #e2b45a !important; }}

  /* DDA table */
  .dda-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
    text-align: center;
  }}
  .dda-table th {{
    color: #8899aa;
    font-weight: 400;
    padding: 2px 6px;
  }}
  .dda-table td {{
    padding: 2px 6px;
  }}
  .dda-table tr th:first-child {{
    color: #e2b45a;
  }}

  /* Rewards */
  .rewards {{
    display: flex;
    gap: 12px;
    font-size: 0.85em;
  }}
  .reward-item {{
    display: flex;
    gap: 4px;
  }}
  .reward-seat {{ color: #e2b45a; }}
  .reward-val {{ color: #ecf0f1; }}
  .reward-pos {{ color: #2ecc71; }}
  .reward-neg {{ color: #e74c3c; }}

  /* DD eval */
  .dd-eval {{
    font-size: 0.85em;
    line-height: 1.8;
  }}
  .dd-eval .label {{ color: #8899aa; }}
  .dd-made {{ color: #2ecc71; font-weight: 600; }}
  .dd-down {{ color: #e74c3c; font-weight: 600; }}
  .dd-detail {{ color: #8899aa; }}

  .no-auction, .no-dda {{
    color: #667788;
    font-size: 0.85em;
    font-style: italic;
  }}

  .footer {{
    text-align: center;
    margin-top: 32px;
    font-size: 0.7em;
    color: #667788;
  }}

  /* Team coloring on hand labels */
  .hand-label.team-t1 {{ color: #5cc8ff; }}
  .hand-label.team-t2 {{ color: #ff9f5c; }}

  /* Team banner at top of page */
  .team-banner {{
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-bottom: 18px;
    font-size: 0.85em;
  }}
  .team-banner .team {{
    padding: 6px 14px;
    border-radius: 4px;
    background: #16213e;
    border: 1px solid #ffffff0a;
  }}
  .team-banner .team.t1 {{ color: #5cc8ff; border-color: #5cc8ff44; }}
  .team-banner .team.t2 {{ color: #ff9f5c; border-color: #ff9f5c44; }}
  .team-banner .vs {{ color: #667788; align-self: center; }}

  /* Table label pill (A or B) and IMPs */
  .meta-bar .table-pill {{
    background: #e2b45a;
    color: #1a1a2e;
    font-weight: 700;
  }}
  .meta-bar .imp-pos {{ color: #2ecc71; font-weight: 600; }}
  .meta-bar .imp-neg {{ color: #e74c3c; font-weight: 600; }}
  .meta-bar .imp-zero {{ color: #8899aa; }}

  /* Companion (other table) one-line pill */
  .companion-compact {{
    font-size: 0.9em;
    color: #ecf0f1;
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .companion-compact .label {{ color: #8899aa; font-size: 0.85em; }}
  .companion-compact .value {{ color: #ecf0f1; font-weight: 600; }}
  .companion-compact .value.reward-pos {{ color: #2ecc71; }}
  .companion-compact .value.reward-neg {{ color: #e74c3c; }}

  /* "Other Tables" swap button (BBO-style) */
  .swap-btn {{
    background: #3498db;
    color: #fff;
    border: none;
    padding: 7px 16px;
    font-family: inherit;
    font-size: 0.85em;
    cursor: pointer;
    border-radius: 4px;
    font-weight: 600;
    margin-left: auto;
  }}
  .swap-btn:hover {{ background: #5dade2; }}
  .swap-btn.active {{ background: #e2b45a; color: #1a1a2e; }}

  /* Bucket tabs */
  .bucket-tabs {{
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 14px;
  }}
  .bucket-tabs button {{
    background: #16213e;
    color: #8899aa;
    border: 1px solid #ffffff0a;
    padding: 7px 16px;
    font-family: inherit;
    font-size: 0.8em;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s;
  }}
  .bucket-tabs button.active {{
    background: #e2b45a;
    color: #1a1a2e;
    border-color: #e2b45a;
    font-weight: 600;
  }}
  .bucket-tabs button:hover:not(.active) {{
    color: #ecf0f1;
    border-color: #ffffff22;
  }}

  /* Room pill — replaces the old table A/B pill */
  .meta-bar .room-pill {{
    background: #2ecc71;
    color: #1a1a2e;
    font-weight: 700;
  }}
  .meta-bar .room-pill.closed {{
    background: #8899aa;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Bridge Board Inspector</h1>
</div>

<div class="team-banner" id="team-banner"></div>

<div class="bucket-tabs" id="bucket-tabs"></div>

<div class="nav">
  <button id="btn-prev" onclick="go(-1)">◀ Prev</button>
  <span class="board-num" id="board-num">Board 1 / 1</span>
  <button id="btn-next" onclick="go(1)">Next ▶</button>
</div>

<div class="meta-bar" id="meta-bar"></div>

<div class="main-layout">
  <div class="table-area">
    <div class="hand hand-north">
      <div class="hand-label" id="label-n">North</div>
      <div id="hand-n"></div>
    </div>
    <div class="hand hand-west">
      <div class="hand-label" id="label-w">West</div>
      <div id="hand-w"></div>
    </div>
    <div class="center-marker" id="center-marker">N</div>
    <div class="hand hand-east">
      <div class="hand-label" id="label-e">East</div>
      <div id="hand-e"></div>
    </div>
    <div class="hand hand-south">
      <div class="hand-label" id="label-s">South</div>
      <div id="hand-s"></div>
    </div>
  </div>

  <div class="right-panel">
    <div class="panel-section">
      <div class="panel-title">Auction</div>
      <div id="auction"></div>
    </div>
    <div class="panel-section">
      <div class="panel-title">Double Dummy Analysis</div>
      <div id="dda"></div>
    </div>
    <div class="panel-section">
      <div class="panel-title">Rewards</div>
      <div id="rewards"></div>
    </div>
    <div class="panel-section" id="companion-section" style="display:none">
      <div class="panel-title">
        <span id="companion-title">Other Room</span>
        <button class="swap-btn" id="swap-btn" onclick="swapToCompanion()">Other Tables ⇄</button>
      </div>
      <div id="companion"></div>
    </div>
    <div class="panel-section" id="dd-eval-section">
      <div class="panel-title">DD Evaluation</div>
      <div id="dd-eval"></div>
    </div>
  </div>
</div>

<div class="footer">
  Arrow keys ← → to navigate &nbsp;|&nbsp; Bridge Board Inspector
</div>

<script>
const entries = {entries_json};
const renders = {renders_json};
const mode = {mode_json};
const teamNames = {team_names_json};
const bucketSizes = {bucket_sizes_json};

// Bucket "t1_ns": indices [0, bucketSizes.t1_ns).
// Bucket "t2_ns": indices [bucketSizes.t1_ns, entries.length).
const bucketRanges = {{
  t1_ns: [0, bucketSizes.t1_ns],
  t2_ns: [bucketSizes.t1_ns, bucketSizes.t1_ns + bucketSizes.t2_ns],
}};

let currentBucket = 't1_ns';
let idx = 0;              // index into entries[]
let savedMainIdx = null;  // remembered position when viewing companion

// Render the top team banner once.
(function renderBanner() {{
  const el = document.getElementById('team-banner');
  if (!teamNames || !teamNames.team1) {{ el.style.display = 'none'; return; }}
  el.innerHTML =
    `<span class="team t1">${{teamNames.team1}}</span>` +
    `<span class="vs">vs</span>` +
    `<span class="team t2">${{teamNames.team2}}</span>`;
}})();

// Render bucket tabs once.
(function renderBucketTabs() {{
  const el = document.getElementById('bucket-tabs');
  if (mode !== 'duplicate' || (bucketSizes.t1_ns === 0 && bucketSizes.t2_ns === 0)) {{
    el.style.display = 'none';
    return;
  }}
  el.innerHTML =
    `<button id="tab-t1-ns" onclick="switchBucket('t1_ns')">` +
    `${{teamNames.team1}} sits N/S &nbsp;(${{bucketSizes.t1_ns}} boards)</button>` +
    `<button id="tab-t2-ns" onclick="switchBucket('t2_ns')">` +
    `${{teamNames.team2}} sits N/S &nbsp;(${{bucketSizes.t2_ns}} boards)</button>`;
}})();

function switchBucket(name) {{
  if (name === currentBucket) return;
  currentBucket = name;
  savedMainIdx = null;
  idx = bucketRanges[name][0];
  show(idx);
}}

function impClass(v) {{
  if (v === null || v === undefined) return 'imp-zero';
  if (v > 0) return 'imp-pos';
  if (v < 0) return 'imp-neg';
  return 'imp-zero';
}}

function fmtSigned(v) {{
  if (v === null || v === undefined) return '?';
  const n = Number(v);
  if (!isFinite(n)) return String(v);
  return (n > 0 ? '+' : '') + n.toFixed(0);
}}

function show(i) {{
  const b = entries[i];
  const r = renders[i];

  // Detect whether this entry is the "main" view (inside the current bucket)
  // or a companion view (outside the current bucket, reached via swap).
  const [lo, hi] = bucketRanges[currentBucket];
  const inBucket = (i >= lo && i < hi);
  const posInBucket = i - lo;
  const bucketSize = hi - lo;

  // Board number in tournament style (1-indexed within bucket for nav display).
  let label;
  if (inBucket) {{
    label = `Board ${{posInBucket + 1}} / ${{bucketSize}}`;
  }} else {{
    // Viewing companion — label by board id instead of position.
    label = `Board id ${{b.board_id}} · other table`;
  }}
  document.getElementById('board-num').textContent = label;

  if (inBucket) {{
    document.getElementById('btn-prev').disabled = (posInBucket === 0);
    document.getElementById('btn-next').disabled = (posInBucket === bucketSize - 1);
  }} else {{
    document.getElementById('btn-prev').disabled = true;
    document.getElementById('btn-next').disabled = true;
  }}

  // Bucket-tab active state
  const t1tab = document.getElementById('tab-t1-ns');
  const t2tab = document.getElementById('tab-t2-ns');
  if (t1tab && t2tab) {{
    t1tab.classList.toggle('active', currentBucket === 't1_ns');
    t2tab.classList.toggle('active', currentBucket === 't2_ns');
  }}

  // Meta bar: Open/Closed room + dealer + vul + contract + board-level IMPs
  const meta = document.getElementById('meta-bar');
  const metaBits = [];
  if (b.room) {{
    const roomCls = b.room === 'Closed Room' ? 'room-pill closed' : 'room-pill';
    metaBits.push(`<span class="${{roomCls}}">${{b.room}}</span>`);
  }}
  metaBits.push(`<span>Dealer: ${{b.dealer || '?'}}</span>`);
  metaBits.push(`<span>Vul: ${{b.vulnerability || '?'}}</span>`);
  metaBits.push(`<span class="contract">${{b.final_contract || '?'}}</span>`);
  if (mode === 'duplicate' && b.team1_imps !== undefined && b.team1_imps !== null) {{
    metaBits.push(
      `<span class="${{impClass(b.team1_imps)}}">` +
      `${{teamNames.team1}}: ${{fmtSigned(b.team1_imps)}} IMPs (board)` +
      `</span>`
    );
  }}
  meta.innerHTML = metaBits.join('');

  // Hands
  document.getElementById('hand-n').innerHTML = r.north;
  document.getElementById('hand-e').innerHTML = r.east;
  document.getElementById('hand-s').innerHTML = r.south;
  document.getElementById('hand-w').innerHTML = r.west;
  document.getElementById('center-marker').textContent = b.dealer || '?';

  // Seat labels with team ownership + color
  const seatMap = r.seat_teams || {{}};
  for (const [seat, elId] of [['N','label-n'],['E','label-e'],['S','label-s'],['W','label-w']]) {{
    const lbl = document.getElementById(elId);
    const full = {{'N':'North','E':'East','S':'South','W':'West'}}[seat];
    const info = seatMap[seat];
    lbl.className = 'hand-label';
    if (info) {{
      lbl.classList.add(`team-${{info.id}}`);
      lbl.textContent = `${{full}} · ${{info.name}}`;
    }} else {{
      lbl.textContent = full;
    }}
  }}

  document.getElementById('auction').innerHTML = r.auction;
  document.getElementById('dda').innerHTML = r.dda;

  // Rewards
  const seats = ['N','E','S','W'];
  const rw = b.score_reward || {{}};
  let rwHtml = '<div class="rewards">';
  for (const seat of seats) {{
    const v = rw[seat];
    if (v === undefined) continue;
    const cls = v > 0 ? 'reward-pos' : v < 0 ? 'reward-neg' : 'reward-val';
    rwHtml += `<div class="reward-item"><span class="reward-seat">${{seat}}</span>` +
              `<span class="${{cls}}">${{typeof v === 'number' ? v.toFixed(0) : v}}</span></div>`;
  }}
  rwHtml += '</div>';
  document.getElementById('rewards').innerHTML = rwHtml;

  // Companion one-liner + swap button
  const compSection = document.getElementById('companion-section');
  const compTitle = document.getElementById('companion-title');
  const swapBtn = document.getElementById('swap-btn');
  if (b.companion_idx !== undefined && b.companion_idx !== null) {{
    compSection.style.display = '';
    const otherRoom = (b.room === 'Open Room') ? 'Closed Room' : 'Open Room';
    compTitle.textContent = otherRoom;
    // Use server-pre-rendered compact line so we don't recompute in JS.
    const line = r.companion_line || '—';
    document.getElementById('companion').innerHTML =
      `<div class="companion-compact">` +
      `<span class="label">Contract &amp; result:</span>` +
      `<span class="value">${{line}}</span>` +
      `</div>`;
    // Show the button in main view, change label in companion view.
    swapBtn.style.display = '';
    swapBtn.textContent = inBucket ? 'Other Tables ⇄' : '← Back to main';
    swapBtn.classList.toggle('active', !inBucket);
  }} else {{
    compSection.style.display = 'none';
  }}

  // DD evaluation
  const ddSection = document.getElementById('dd-eval-section');
  const ddContent = r.dd_eval || '';
  if (ddContent) {{
    ddSection.style.display = '';
    document.getElementById('dd-eval').innerHTML = ddContent;
  }} else {{
    ddSection.style.display = 'none';
  }}
}}

function swapToCompanion() {{
  const b = entries[idx];
  const [lo, hi] = bucketRanges[currentBucket];
  const inBucket = (idx >= lo && idx < hi);
  if (inBucket) {{
    // Go TO the companion; remember where we came from.
    if (b.companion_idx === undefined || b.companion_idx === null) return;
    savedMainIdx = idx;
    idx = b.companion_idx;
    show(idx);
  }} else {{
    // Already on companion — go back.
    if (savedMainIdx !== null) {{
      idx = savedMainIdx;
      savedMainIdx = null;
      show(idx);
    }}
  }}
}}

function go(delta) {{
  const [lo, hi] = bucketRanges[currentBucket];
  if (idx < lo || idx >= hi) {{
    // Viewing companion — ignore prev/next.
    return;
  }}
  idx = Math.max(lo, Math.min(hi - 1, idx + delta));
  savedMainIdx = null;
  show(idx);
}}

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') go(-1);
  if (e.key === 'ArrowRight') go(1);
}});

// Start on the first entry of the default bucket.
idx = bucketRanges[currentBucket][0];
show(idx);
</script>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Bridge Board Inspector")
    parser.add_argument("json_file", nargs="?", default="play_data.json",
                        help="Path to play JSON file (default: play_data.json)")
    args = parser.parse_args()

    # Load data
    with open(args.json_file) as f:
        data = json.load(f)

    mode = data.get("mode", "single")
    boards = data.get("logs", [])
    n_entries = len(boards) * 2 if mode == "duplicate" else len(boards)
    print(f"Loaded {len(boards)} boards ({mode} mode → {n_entries} viewable tables) "
          f"from {args.json_file}")
    if mode == "duplicate" and "summary" in data:
        s = data["summary"]
        names = data.get("team_names", {"team1": "Team 1"})
        print(f"  {names.get('team1', 'Team 1')}: "
              f"{s.get('team1_imps_per_board_mean', 0):+.3f} ± "
              f"{s.get('team1_imps_per_board_se', 0):.3f} IMPs/board")

    # Build HTML
    html = build_html(data)
    out_path = os.path.splitext(args.json_file)[0] + "_inspector.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("HTML file generated.")
    
if __name__ == "__main__":
    main()
