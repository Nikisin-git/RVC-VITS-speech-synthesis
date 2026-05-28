"""Build FAISS IVF index from extracted features.

Reimplemented from upstream's `train_index()` in `infer-web.py` so we don't
have to import the Gradio module.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import rvc_core  # noqa: F401


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    p.add_argument("--dataset-dir", required=False)
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--version", default="v2", choices=["v1", "v2"])
    args = p.parse_args()

    import faiss  # type: ignore
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
        _have_sklearn = True
    except ImportError:
        _have_sklearn = False

    exp_dir = Path(args.logs_dir).resolve()
    feature_dir = exp_dir / ("3_feature256" if args.version == "v1" else "3_feature768")
    if not feature_dir.exists():
        print(f"ERROR: feature dir not found: {feature_dir}", flush=True)
        return 1
    feats = sorted(feature_dir.glob("*.npy"))
    if not feats:
        print("ERROR: no .npy features extracted", flush=True)
        return 1

    npys = [np.load(p) for p in feats]
    big = np.concatenate(npys, axis=0)
    idx = np.arange(big.shape[0])
    np.random.shuffle(idx)
    big = big[idx]

    if big.shape[0] > 2e5 and _have_sklearn:
        print(f"k-means {big.shape[0]} -> 10k centers", flush=True)
        big = (
            MiniBatchKMeans(
                n_clusters=10000,
                verbose=False,
                batch_size=256 * max(1, os.cpu_count() or 1),
                compute_labels=False,
                init="random",
            )
            .fit(big)
            .cluster_centers_
        )

    np.save(exp_dir / "total_fea.npy", big)
    n_ivf = min(int(16 * np.sqrt(big.shape[0])), big.shape[0] // 39)
    dim = 256 if args.version == "v1" else 768

    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    print(f"training IVF{n_ivf} (dim={dim}, samples={big.shape[0]})", flush=True)
    index.train(big)
    trained = exp_dir / (
        f"trained_IVF{n_ivf}_Flat_nprobe_1_{args.exp_name}_{args.version}.index"
    )
    faiss.write_index(index, str(trained))

    batch_add = 8192
    for i in range(0, big.shape[0], batch_add):
        index.add(big[i:i + batch_add])
    added = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_1_{args.exp_name}_{args.version}.index"
    faiss.write_index(index, str(added))
    print(f"index built: {added}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
