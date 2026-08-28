#
#
# Archived JAX version (pre requirements) used for the conversion 
#
#
#

# Usage
# conda create -n jax-old python=3.10 -y
# conda activate jax-old
# pip install "numpy<2" "jax[cpu]==0.4.23" "jaxlib==0.4.23"
# python convert_brl_pickle_checkpoint.py --input our_models/mlp_rl_baseline.pkl --output mlp_rl_baseline.npz

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

def tree_flatten_with_paths(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return leaves, treedef


def to_numpy_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def to_jax_tree(tree):
    return jax.tree_util.tree_map(lambda x: jax.numpy.asarray(x), tree)


def save_npz(tree, path: Path):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    arrays = {f"arr_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    meta = {
        "num_leaves": len(leaves),
        "treedef_repr": repr(treedef),
    }
    np.savez(path, **arrays)
    with open(path.with_suffix(path.suffix + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load a BRL/Haiku pickle checkpoint in a compatible old JAX environment and "
            "re-save the raw parameter leaves as a NumPy .npz archive."
        )
    )
    parser.add_argument("--input", required=True, help="Path to old .pkl checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .npz archive")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "rb") as f:
        params = pickle.load(f)

    params_np = to_numpy_tree(params)
    save_npz(params_np, output_path)

    print(f"Loaded pickle from: {input_path}")
    print(f"Saved NumPy checkpoint to: {output_path}")
    print(f"Saved metadata to: {output_path}.meta.json")
    print(
        "Next step: in your newer JAX env, load the arrays from the .npz and rebuild the exact "
        "Haiku parameter tree structure before calling forward_pass.apply(...)."
    )


if __name__ == "__main__":
    main()
