#!/usr/bin/env python
"""
Minimum edit distance from the 500K synthetic AD design set to several
reference sequence sets, parallelized for Savio with multiprocessing.

Dependencies: numpy, pandas, editdistance. Nothing else.

Differences from the notebook version:
  * Parallel across cores (multiprocessing) and optionally across nodes
    (SLURM array -> --shard/--nshards, then --merge).
  * Self-comparison masks the query's own sequence, so you get the nearest
    *other* sequence instead of a column of zeros.
  * Length-bucket pruning visits buckets in order of |len(query) - len(target)|
    so the lower bound can actually prune; the notebook went in ascending
    length order, which prunes nothing.
  * Exact-match short circuit via a hash lookup before any distance is computed.
  * Computes on unique queries and maps back.
  * --inspect prints columns/shapes so you can confirm column names first.

Usage
-----
  python edit_distance_claude.py --inspect
  python edit_distance_claude.py --target train --workers $SLURM_CPUS_PER_TASK
  python edit_distance_claude.py --target self --merge --outdir results
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import editdistance
except ImportError:
    editdistance = None

# --------------------------------------------------------------------------
# Paths on Savio
# --------------------------------------------------------------------------
BASE = "/global/scratch/projects/fc_mvslab/OpenProjects/Marissa"
QUERY_PATH = os.path.join(BASE, "DesignFiles", "500k_Design_AA.csv")
REF_DIR = os.path.join(BASE, "SynADs", "2026-10-10_500K_many_edit_distance")

TARGETS = {
    "train": dict(
        path=os.path.join(REF_DIR, "TrainingSeqs_seqfeats.csv"),
        read_kwargs={"index_col": 0},
        filters=[("AD Origin", "More_Plant_data.PADI_library.xlsx")],
    ),
    "proteome": dict(path=os.path.join(REF_DIR, "TiledProteome.csv")),
    "choptf": dict(path=os.path.join(REF_DIR, "ChopTF_seqfeat.csv")),
    "synth18750": dict(
        path=os.path.join(REF_DIR, "SynADs_seqfeat.csv"),
        filters=[("Ctrl", 0), ("NegCtrl", 0)],
    ),
    "self": dict(path=QUERY_PATH, exclude_self=True),
}

SEQ_COL_CANDIDATES = ["AAseq", "AA_seq", "Seq", "seq", "sequence", "Sequence", "AA"]
NO_MATCH = -1


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def find_seq_col(df, explicit=None, label=""):
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"{label}: column {explicit!r} not in {list(df.columns)[:20]}")
        return explicit
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        s = df[c].dropna().astype(str).head(50)
        if len(s) and s.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY*]{5,}").mean() > 0.8:
            return c
    raise KeyError(f"{label}: no sequence column found in {list(df.columns)[:20]}")


def load_sequences(path, seq_col=None, read_kwargs=None, filters=None, label=""):
    if not os.path.exists(path):
        sys.exit(f"[fatal] {label}: file not found: {path}")
    df = pd.read_csv(path, **(read_kwargs or {}))
    col = find_seq_col(df, seq_col, label=label or path)
    n0 = len(df)
    for fcol, fval in (filters or []):
        if fcol not in df.columns:
            sys.exit(f"[fatal] {label}: filter column {fcol!r} missing from {path}")
        df = df[df[fcol] == fval]
    seqs = df[col].dropna().astype(str).str.strip().str.upper()
    seqs = seqs[seqs.str.len() > 0]
    print(f"[load] {label}: {n0} rows -> {len(seqs)} sequences "
          f"(col={col!r}, {seqs.nunique()} unique)", flush=True)
    return seqs.tolist()


# --------------------------------------------------------------------------
# Worker (globals are inherited by fork, so targets are not re-pickled)
# --------------------------------------------------------------------------
_G = {}


def _setup(targets, exclude_self):
    buckets = {}
    for j, s in enumerate(targets):
        buckets.setdefault(len(s), []).append((j, s))
    exact = {}
    for j, s in enumerate(targets):
        exact.setdefault(s, j)
    _G["targets"] = targets
    _G["buckets"] = buckets
    _G["lengths"] = sorted(buckets)
    _G["exact"] = exact
    _G["self"] = exclude_self


def _nearest(s1):
    buckets, lengths = _G["buckets"], _G["lengths"]
    exclude_self = _G["self"]

    if not exclude_self:
        j = _G["exact"].get(s1)
        if j is not None:
            return 0, j

    L = len(s1)
    best, best_j = np.inf, NO_MATCH
    # visit length buckets nearest-length-first so |L - l| prunes
    for l in sorted(lengths, key=lambda x: abs(L - x)):
        if abs(L - l) >= best:
            break
        for j, s2 in buckets[l]:
            if exclude_self and s2 == s1:
                continue
            d = editdistance.eval(s1, s2)
            if d < best:
                best, best_j = d, j
                if best == 0:
                    return 0, j
    return (int(best) if np.isfinite(best) else NO_MATCH), best_j


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(args):
    if editdistance is None:
        sys.exit("[fatal] the 'editdistance' package is not importable in this env.\n"
                 "        try:  pip install --user editdistance")

    spec = TARGETS[args.target]
    queries_all = load_sequences(args.query, args.query_col, label="query")
    targets = load_sequences(spec["path"], args.target_col, spec.get("read_kwargs"),
                             spec.get("filters"), label=f"target:{args.target}")
    targets = list(dict.fromkeys(targets))

    rows = list(range(args.shard, len(queries_all), args.nshards))
    queries = [queries_all[i] for i in rows]
    if args.nshards > 1:
        print(f"[shard] {args.shard}/{args.nshards}: {len(queries)} queries", flush=True)

    uniq = list(dict.fromkeys(queries))
    pairs = len(uniq) * len(targets)
    print(f"[info] {len(queries)} queries -> {len(uniq)} unique", flush=True)
    print(f"[info] {pairs:,} pairs; at ~0.3M pairs/s/core on {args.workers} cores "
          f"that is ~{pairs / 3e5 / max(args.workers, 1) / 3600:.1f} h (upper bound; "
          f"pruning makes it faster)", flush=True)
    if spec.get("exclude_self"):
        print("[info] self-comparison: identical sequences are masked", flush=True)

    _setup(targets, bool(spec.get("exclude_self")))

    mins = np.empty(len(uniq), dtype=np.int32)
    idxs = np.empty(len(uniq), dtype=np.int64)
    t0 = time.time()

    if args.workers > 1:
        ctx = mp.get_context("fork")  # fork shares the target list copy-on-write
        with ctx.Pool(args.workers) as pool:
            it = pool.imap(_nearest, uniq, chunksize=args.chunk_size)
            for k, (d, j) in enumerate(it):
                mins[k], idxs[k] = d, j
                if (k + 1) % args.report_every == 0 or k + 1 == len(uniq):
                    el = time.time() - t0
                    print(f"[prog] {k + 1}/{len(uniq)}  {el / 60:7.1f} min  "
                          f"eta {el / (k + 1) * (len(uniq) - k - 1) / 60:7.1f} min",
                          flush=True)
    else:
        for k, s in enumerate(uniq):
            mins[k], idxs[k] = _nearest(s)
            if (k + 1) % args.report_every == 0:
                el = time.time() - t0
                print(f"[prog] {k + 1}/{len(uniq)}  {el / 60:7.1f} min", flush=True)

    dist_map = dict(zip(uniq, mins.tolist()))
    match_map = {q: (targets[j] if j >= 0 else None) for q, j in zip(uniq, idxs.tolist())}

    out = pd.DataFrame({"orig_row": rows, "AAseq": queries})
    out["min_edit_dist"] = [dist_map[q] for q in queries]
    out["match_AAseq"] = [match_map[q] for q in queries]
    out.loc[out["min_edit_dist"] == NO_MATCH, "min_edit_dist"] = np.nan

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f".part{args.shard:03d}" if args.nshards > 1 else ""
    path = os.path.join(args.outdir, f"MinSynth500K_to_{args.target}{suffix}.csv")
    if args.nshards == 1:
        out = out.drop(columns="orig_row")
    out.to_csv(path, index=False)
    print(f"[done] wrote {path}  ({len(out)} rows, {(time.time() - t0) / 60:.1f} min)",
          flush=True)


def merge(args):
    import glob
    pat = os.path.join(args.outdir, f"MinSynth500K_to_{args.target}.part*.csv")
    parts = sorted(glob.glob(pat))
    if not parts:
        sys.exit(f"no parts matching {pat}")
    df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    df = df.sort_values("orig_row").drop(columns="orig_row").reset_index(drop=True)
    path = os.path.join(args.outdir, f"MinSynth500K_to_{args.target}.csv")
    df.to_csv(path, index=False)
    print(f"[merge] {len(parts)} parts -> {path} ({len(df)} rows)")
    print(df["min_edit_dist"].describe())


def inspect(args):
    print(f"editdistance importable: {editdistance is not None}")
    files = [("query", QUERY_PATH, {})] + [
        (k, v["path"], v.get("read_kwargs") or {})
        for k, v in TARGETS.items() if k != "self"
    ]
    for name, path, kw in files:
        print("\n" + "=" * 70)
        print(f"{name}: {path}")
        if not os.path.exists(path):
            print("  !! FILE NOT FOUND")
            continue
        head = pd.read_csv(path, nrows=5, **kw)
        full = pd.read_csv(path, usecols=[head.columns[0]], **kw)
        print(f"  rows={len(full)}  cols={list(head.columns)[:15]}")
        try:
            col = find_seq_col(head, label=name)
            print(f"  detected sequence column: {col!r}")
            print(f"  example: {head[col].iloc[0]}")
            print(f"  length range (first 5): "
                  f"{sorted(head[col].astype(str).str.len().unique())}")
        except KeyError as e:
            print(f"  !! {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=sorted(TARGETS))
    p.add_argument("--query", default=QUERY_PATH)
    p.add_argument("--query-col", default=None)
    p.add_argument("--target-col", default=None)
    p.add_argument("--outdir", default="results")
    p.add_argument("--workers", type=int, default=0, help="processes; 0 = all cores")
    p.add_argument("--chunk-size", type=int, default=50, help="queries per worker task")
    p.add_argument("--report-every", type=int, default=2000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--inspect", action="store_true")
    args = p.parse_args()

    if args.workers <= 0:
        args.workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count()

    if args.inspect:
        return inspect(args)
    if not args.target:
        sys.exit("--target is required (or use --inspect)")
    if args.merge:
        return merge(args)
    return run(args)


if __name__ == "__main__":
    main()
