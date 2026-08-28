"""
Convert selfplay_data.json deals into BBO .lin format.

Usage:
    python json_to_lin.py                          # outputs selfplay_data.lin
    python json_to_lin.py input.json output.lin    # custom paths
"""

import json
import sys


DEALER_CODE = {"S": "1", "W": "2", "N": "3", "E": "4"}

VULN_CODE = {"NS": "n", "EW": "e", "BOTH": "b", "NONE": "o"}


def hand_to_lin(cards: list[str]) -> str:
    """Convert a list of cards like ['SA','HK','D3'] to LIN hand string 'SAHKD3C...'"""
    suits = {"S": [], "H": [], "D": [], "C": []}
    for card in cards:
        suits[card[0]].append(card[1])
    return (
        "S" + "".join(suits["S"])
        + "H" + "".join(suits["H"])
        + "D" + "".join(suits["D"])
        + "C" + "".join(suits["C"])
    )


def bid_to_lin(bid: str) -> str:
    """Normalise a bid string to LIN format."""
    bid = bid.strip()
    upper = bid.upper()
    if upper in ("PASS", "P"):
        return "p"
    if upper in ("DOUBLE", "X", "D"):
        return "d"
    if upper in ("REDOUBLE", "XX", "R"):
        return "r"
    # e.g. "2NT" -> "2N", "4H" -> "4H"
    if upper.endswith("NT"):
        return bid[0] + "N"
    return bid.upper()


def board_to_lin(board: dict, board_num: int) -> str:
    """Render one board as a LIN string."""
    dealer = board["dealer"]           # "N"/"E"/"S"/"W"
    vuln   = board["vulnerability"]    # "NS"/"EW"/"BOTH"/"NONE"

    deal = board["deal"]
    s_hand = hand_to_lin(deal["S"])
    w_hand = hand_to_lin(deal["W"])
    n_hand = hand_to_lin(deal["N"])

    dealer_digit = DEALER_CODE.get(dealer, "1")
    vuln_char    = VULN_CODE.get(vuln, "o")

    # md|<dealer_digit><S_hand>,<W_hand>,<N_hand>|
    md = f"md|{dealer_digit}{s_hand},{w_hand},{n_hand}|"

    # Bids
    bids_lin = "".join(f"mb|{bid_to_lin(b['bid'])}|" for b in board["auction"])

    lines = [
        f"qx|o{board_num}|",
        md,
        f"sv|{vuln_char}|",
        f"ah|Board {board_num}|",
        "pn|South,West,North,East|",
        "pg||",
        bids_lin,
        "pg||",
    ]

    return "".join(lines)


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    boards = data.get("logs", [])
    if not boards:
        print("No boards found under 'logs' key.")
        sys.exit(1)

    lin_blocks = []
    for i, board in enumerate(boards, start=1):
        lin_blocks.append(board_to_lin(board, i))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lin_blocks))

    print(f"Wrote {len(lin_blocks)} boards to {output_path}")


if __name__ == "__main__":
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "selfplay_data.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "selfplay_data.lin"
    convert(input_file, output_file)
