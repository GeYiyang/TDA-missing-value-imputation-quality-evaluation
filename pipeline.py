# -*- coding: utf-8 -*-
"""
Created on Wed Sep  25 09:07:10 2025

@author: Ge Yiyang

"""

import os, warnings, random, pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

import networkx as nx
from scipy.sparse.csgraph import connected_components, dijkstra

from gudhi.cover_complex import MapperComplex

# --- Lock low-level threads to avoid nested parallelism blow-up --- #
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
np.random.seed(100)
random.seed(100)

# =========================================================
# Cache utilities
# =========================================================
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def save_cache(obj, name):
    with open(os.path.join(CACHE_DIR, name + ".pkl"), "wb") as f:
        pickle.dump(obj, f)

def load_cache(name):
    pth = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(pth):
        with open(pth, "rb") as f:
            return pickle.load(f)
    return None

# ==== (Optional) Result recorder that can be removed safely ====
import json, datetime
from pathlib import Path

def _flatten_results_for_df(results: dict) -> pd.DataFrame:
    """
    Expand results[(res, gain)] into a compact DataFrame (excluding large arrays),
    convenient for CSV export and reports.
    """
    rows = []
    for (res, gain), d in results.items():
        row = {
            "res": res,
            "gain": gain,
            "Tobs_max": float(d["Tobs_max"]),
            "p_max": float(d["p_max"]),
        }
        # Observed stats and p-values for four topology types
        parts = d.get("Tobs_parts", {})
        pparts = d.get("p_parts", {})
        for topo in ["connected_components", "downbranch", "upbranch", "loop"]:
            row[f"Tobs_{topo}"] = float(parts.get(topo, float("nan")))
            row[f"p_{topo}"] = float(pparts.get(topo, float("nan")))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["res", "gain"]).reset_index(drop=True)
    return df

def write_run_reports(results: dict,
                      meta: dict = None,
                      out_dir: str = "run_reports",
                      basename: str = "mapper_run"):
    """
    Write a one-shot report bundle:
      1) CSV summary (compact, for quick inspection)
      2) Full pickle (contains permutation distributions; for exact reproducibility)
      3) Simple Markdown report
      4) A .DONE flag file to indicate successful completion
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{basename}_{ts}"

    # 1) Save summary CSV
    df = _flatten_results_for_df(results)
    csv_path = Path(out_dir) / f"{stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 2) Save full results as pickle (includes permutation distributions)
    pkl_path = Path(out_dir) / f"{stem}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"results": results, "meta": meta or {}}, f)

    # 3) Simple Markdown report
    md_path = Path(out_dir) / f"{stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Mapper Run Report\n\n")
        f.write(f"- Generated at: {ts}\n")
        if meta:
            # keep ascii to avoid escaping issues in some environments
            f.write(f"- Parameter summary: {json.dumps(meta, ensure_ascii=True)}\n")
        f.write(f"- Summary CSV: `{csv_path.name}`\n")
        f.write(f"- Full results PKL: `{pkl_path.name}`\n\n")

        if not df.empty:
            # Write a small Markdown table
            cols = ["res","gain","Tobs_max","p_max","Tobs_connected_components","p_connected_components",
                    "Tobs_downbranch","p_downbranch","Tobs_upbranch","p_upbranch","Tobs_loop","p_loop"]
            cols = [c for c in cols if c in df.columns]
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + " --- |"*len(cols) + "\n")
            for _, r in df.iterrows():
                f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")

    # 4) Completion flag file
    done_flag = Path(out_dir) / f"{stem}.DONE"
    with open(done_flag, "w") as f:
        f.write("OK\n")

    print(f"[REPORT] Written: {csv_path} / {md_path} / {pkl_path} (and optional DOCX)")

# ==== Optional "in-progress" partial dump (try to preserve intermediate results on exceptions) ====
def safe_dump_partial(results_partial: dict,
                      out_dir: str = "run_reports",
                      tag: str = "partial"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"partial_{tag}_{ts}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(results_partial, f)
        print(f"[REPORT] Partial results saved: {path}")
    except Exception:
        pass

# =========================================================
# 1) Preprocessing: standardization + filters (PCA1, kNN distance)
# =========================================================
def _cdf01(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x)+1, dtype=float)
    return (ranks - 0.5) / len(x)

def preprocess_pool(A: pd.DataFrame, B: pd.DataFrame, knn_k=20, use_2d_filter=True):
    n1, n2 = len(A), len(B)
    pool_df = pd.concat([A, B], axis=0, ignore_index=True)

    scaler = StandardScaler().fit(pool_df)
    coords = scaler.transform(pool_df).astype(np.float32)

    pca1 = PCA(n_components=1, random_state=0).fit_transform(coords).ravel()
    nn = NearestNeighbors(n_neighbors=knn_k, metric="euclidean").fit(coords)
    dists, _ = nn.kneighbors(coords)
    knnd = dists[:, -1]

    pca1u = _cdf01(pca1)
    knndu = _cdf01(knnd)

    if use_2d_filter:
        fil = np.vstack([pca1u, knndu]).T.astype(np.float32)
    else:
        fil = pca1u.reshape(-1, 1).astype(np.float32)

    # Diagnostics
    q = lambda v: np.quantile(v, [0, .25, .5, .75, 1.0])
    if fil.shape[1] == 1:
        print(f"[DBG] filter CDF stats (1D): q={q(fil[:,0])}")
    else:
        print(f"[DBG] filter CDF stats (PCA1): q={q(fil[:,0])}")
        print(f"[DBG] filter CDF stats (kNNd): q={q(fil[:,1])}")

    idx_A = np.arange(0, n1, dtype=int)
    idx_B = np.arange(n1, n1+n2, dtype=int)
    return coords, fil, idx_A, idx_B

# =========================================================
# 2) Adaptive KMeans
# =========================================================
class AdaptiveKMeans:
    def __init__(self,
                 base_n_clusters=5,
                 random_state=0,
                 n_init=10,
                 min_cluster_size=None,
                 max_n_clusters=None):
        self.base_n_clusters = int(base_n_clusters)
        self.random_state = random_state
        self.n_init = int(n_init)
        # Use None to indicate "disabled"
        if (min_cluster_size is None) or (int(min_cluster_size) <= 1):
            self.min_cluster_size = None
        else:
            self.min_cluster_size = int(min_cluster_size)
        self.max_n_clusters = None if max_n_clusters is None else int(max_n_clusters)

    def fit_predict(self, X, y=None):
        m = X.shape[0]
        if m <= 1:
            return np.zeros(m, dtype=int)

        # If min_cluster_size is disabled, only base_n_clusters and sample size constrain k
        if self.min_cluster_size is None:
            k = min(self.base_n_clusters, m)
        else:
            k_by_size = max(1, m // self.min_cluster_size)
            k = min(self.base_n_clusters, k_by_size)

        if self.max_n_clusters is not None:
            k = min(k, self.max_n_clusters)
        k = min(k, m)

        if k <= 1:
            return np.zeros(m, dtype=int)

        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
        return km.fit_predict(X)

# =========================================================
# 3) Build Mapper (min_cluster_size can be toggled on/off)
#    Usage:
#      - Off: min_cluster_size=None (default: off)
#      - On : min_cluster_size=10 (set a positive integer)
# =========================================================
def build_mapper_from_indices(coords_full, fil_full, idx, res, gain,
                              n_clusters=5, km_random_state=0,
                              min_cluster_size=10, max_n_clusters=None):
    sub_coords = coords_full[idx, :]
    sub_fil = fil_full[idx, :]

    n_filters = sub_fil.shape[1]
    filter_bnds = np.array([[0.0, 1.0]] * n_filters, dtype=np.float32)

    mapper = MapperComplex(
        filter_bnds=filter_bnds,
        resolutions=np.array([res] * n_filters, dtype=int),
        gains=np.array([gain] * n_filters, dtype=float),
        clustering=AdaptiveKMeans(
            base_n_clusters=n_clusters,
            random_state=km_random_state,
            n_init=10,
            min_cluster_size=min_cluster_size,   # None disables; integer enables
            max_n_clusters=max_n_clusters
        ),
        input_type="point cloud",
    )
    mapper.fit(sub_coords, filters=sub_fil, colors=sub_fil)
    return mapper, sub_fil

# =========================================================
# 4) statmapper-style features
# =========================================================
def _mapper2networkx_gudhi(M):
    st = M.mapper_ if hasattr(M, "mapper_") else M.simplex_tree_
    G = nx.Graph()
    for (splx, _) in st.get_skeleton(1):
        if len(splx) == 1: G.add_node(splx[0])
        elif len(splx) == 2: G.add_edge(splx[0], splx[1])
    return G

def compute_topological_features_statmapper(M, func=None, func_type="data",
                                            topo_type="downbranch", threshold=0.0):
    mapper = M.mapper_ if hasattr(M, "mapper_") else M.simplex_tree_
    node_info = M.node_info_
    num_nodes = len(node_info)

    if func is None:
        A = np.zeros((num_nodes, num_nodes), dtype=float)
        for (splx, _) in mapper.get_skeleton(1):
            if len(splx) == 2:
                A[splx[0], splx[1]] = 1.0
                A[splx[1], splx[0]] = 1.0
        dij = dijkstra(A, directed=False)
        D = np.where(np.isinf(dij), 0.0, dij)
        func = list(-D.max(axis=1))
        func_type = "node"

    if func_type == "data":
        function = [np.mean([func[i] for i in node_info[v]["indices"]]) for v in range(num_nodes)]
    else:
        function = list(func)

    dgm, bnd = [], []

    if topo_type == "connected_components":
        A = np.zeros((num_nodes, num_nodes), dtype=float)
        for (splx, _) in mapper.get_skeleton(1):
            if len(splx) == 2:
                A[splx[0], splx[1]] = 1.0
                A[splx[1], splx[0]] = 1.0
        _, labels = connected_components(A, directed=False)
        for cc in np.unique(labels):
            pts = np.where(labels == cc)[0]
            vals = [function[p] for p in pts]
            if abs(min(vals) - max(vals)) >= threshold:
                dgm.append((0, (min(vals), max(vals))))
                bnd.append(list(pts))

    elif topo_type in ("downbranch", "upbranch"):
        f = np.array(function, dtype=float)
        if topo_type == "upbranch": f = -f
        A = np.zeros((num_nodes, num_nodes), dtype=float)
        for (splx, _) in mapper.get_skeleton(1):
            if len(splx) == 2:
                A[splx[0], splx[1]] = 1.0
                A[splx[1], splx[0]] = 1.0

        order = np.argsort(f)
        rank = np.empty(num_nodes, dtype=int); rank[order] = np.arange(num_nodes)

        def find(i, parent): return i if parent[i] == i else find(parent[i], parent)
        def union(i, j, parent):
            if f[i] <= f[j]: parent[j] = i
            else: parent[i] = j

        parent = -np.ones(num_nodes, dtype=int)
        diag, comp, seen = {}, {}, {}

        for t in range(num_nodes):
            u = order[t]
            nbrs = np.where(A[u, :] == 1.0)[0]
            lower = [v for v in nbrs if rank[v] <= t] if nbrs.size > 0 else []
            if not lower:
                parent[u] = u
                continue
            neigh_pars = [find(v, parent) for v in lower]
            g = neigh_pars[np.argmin([f[w] for w in neigh_pars])]
            pg = find(g, parent)
            parent[u] = pg
            for v in lower:
                pv = find(v, parent)
                if pg != pv:
                    pp = pg if f[pg] > f[pv] else pv
                    comp[pp] = []
                    for w in order[:t]:
                        if find(w, parent) == pp and w not in seen:
                            seen[w] = True; comp[pp].append(w)
                    comp[pp].append(u)
                    if abs(f[pp] - f[u]) >= 0.0:  
                        diag[pp] = u
                    union(pg, pv, parent)
                else:
                    if len(nbrs) == len(lower):
                        comp[pg] = []
                        for w in order[:t+1]:
                            if find(w, parent) == pg and w not in seen:
                                seen[w] = True; comp[pg].append(w)
                        comp[pg].append(u)
                        if abs(f[pg] - f[u]) >= 0.0:
                            diag[pg] = u

        for key, val in diag.items():
            if topo_type == "downbranch": dgm.append((0, (f[key], f[val])))
            else: dgm.append((0, (-f[val], -f[key])))
            bnd.append(comp[key])

    elif topo_type == "loop":
        G = _mapper2networkx_gudhi(M)
        for pts in nx.cycle_basis(G):
            vals = [function[p] for p in pts]
            if abs(min(vals) - max(vals)) >= threshold:
                dgm.append((1, (min(vals), max(vals))))
                bnd.append(list(pts))

    return dgm, bnd

# =========================================================
# 5) Bottleneck proxy
# =========================================================
def _pairs_from_dgm(dgm):
    if not dgm: return np.empty((0,2), dtype=np.float64)
    arr = np.array([[bd[0], bd[1]] for dim, bd in dgm if dim <= 1],
                   dtype=np.float64, order="C")
    return arr

def _persist_vec(P, k=200):
    if P.size == 0: return np.zeros(k, dtype=np.float64)
    pers = P[:,1]-P[:,0]
    if pers.size > k:
        idx = np.argpartition(pers, -k)[-k:]
        v = pers[idx]; v.sort(); v = v[::-1]
    else:
        v = np.sort(pers)[::-1]
        if v.size < k: v = np.pad(v, (0, k-v.size))
    return np.ascontiguousarray(v, dtype=np.float64)

def bottleneck_statmapper(MF1, MF2, topo_type="connected_components", proxy_k=200):
    (M1, fil1), (M2, fil2) = MF1, MF2
    f1 = fil1[:,0] if fil1.ndim>1 else fil1
    f2 = fil2[:,0] if fil2.ndim>1 else fil2
    dgm1, _ = compute_topological_features_statmapper(M1, func=f1, func_type="data", topo_type=topo_type)
    dgm2, _ = compute_topological_features_statmapper(M2, func=f2, func_type="data", topo_type=topo_type)
    P = _pairs_from_dgm(dgm1); Q = _pairs_from_dgm(dgm2)
    if P.size==0 and Q.size==0: return 0.0
    if P.size==0 or Q.size==0: return float("inf")
    v1 = _persist_vec(P, k=proxy_k)
    v2 = _persist_vec(Q, k=proxy_k)
    return float(np.max(np.abs(v1-v2)))

# =========================================================
# 6) Diagnostics
# =========================================================
TOPO_TYPES = ["connected_components","downbranch","upbranch","loop"]

def summarize_mapper(M, tag=""):
    st = M.mapper_ if hasattr(M,"mapper_") else M.simplex_tree_
    n_nodes = len(M.node_info_)
    n_edges = sum(1 for (splx,_) in st.get_skeleton(1) if len(splx)==2)
    sizes = []
    for k in M.node_info_.keys():
        if "size" in M.node_info_[k]: sizes.append(M.node_info_[k]["size"])
        elif "indices" in M.node_info_[k]: sizes.append(len(M.node_info_[k]["indices"]))
    sizes = np.array(sizes) if sizes else np.array([0])
    print(f"[DBG] {tag} nodes={n_nodes}, edges={n_edges}, node_size[min/med/max]={sizes.min()}/{np.median(sizes)}/{sizes.max()}")

def count_features_all(M, Fcol):
    parts={}
    for topo in TOPO_TYPES:
        dgm,_=compute_topological_features_statmapper(M, func=Fcol, func_type="data", topo_type=topo)
        parts[topo]=len(dgm)
    return parts

# =========================================================
# 7) Permutation test with parallelism (loky backend)
# =========================================================
def _observed_stat_pair(coords, fil, idx_A, idx_B, res, gain, n_clusters, km_random_state, verbose=True):
    M_A, F_A = build_mapper_from_indices(coords, fil, idx_A, res, gain, n_clusters, km_random_state)
    M_B, F_B = build_mapper_from_indices(coords, fil, idx_B, res, gain, n_clusters, km_random_state)
    if verbose:
        summarize_mapper(M_A, "A")
        summarize_mapper(M_B, "B")
        featsA = count_features_all(M_A, F_A[:, 0])
        featsB = count_features_all(M_B, F_B[:, 0])
        print(f"[DBG] features A: {featsA}")
        print(f"[DBG] features B: {featsB}")
    parts = [bottleneck_statmapper((M_A, F_A), (M_B, F_B), topo) for topo in TOPO_TYPES]
    return max(parts), parts


def _one_perm_job(seed, n, n1, coords, fil, res, gain, n_clusters, km_random_state):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.arange(n))
    A_idx = perm[:n1]
    B_idx = perm[n1:]
    T, parts = _observed_stat_pair(coords, fil, A_idx, B_idx, res, gain, n_clusters, km_random_state, verbose=False)
    return T, parts


def run_mapper_analysis_permutation_statmapper(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    res_list, gain_list,
    num_perm=99, knn_k=20,
    n_clusters=4, km_random_state=0,
    n_jobs=1
):
    # ===== Preprocessing =====
    coords, fil, idx_A, idx_B = preprocess_pool(data1, data2, knn_k=knn_k, use_2d_filter=True)
    n1, n2 = len(idx_A), len(idx_B)
    n = n1 + n2
    print(f"[INFO] coords shape={coords.shape}, fil shape={fil.shape}, "
          f"approx mem={coords.nbytes / 1e6 + fil.nbytes / 1e6:.1f} MB")

    results = {}

    # ===== Main loop =====
    for res in res_list:
        for gain in gain_list:
            print(f"\n🔹 Processing res={res}, gain={gain}")

            # Observed statistic
            Tobs, parts_obs = _observed_stat_pair(
                coords, fil, idx_A, idx_B, res, gain, n_clusters, km_random_state, verbose=True
            )
            print(f"🟩 Observed (max over {TOPO_TYPES}): {Tobs:.4f}  parts={parts_obs}")

            # ===== Parallel permutations (automatic memmap + batch scheduling) =====
            perm_stats = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                max_nbytes="1M",
                temp_folder=CACHE_DIR,
                batch_size="auto",
                prefer="processes"
            )(
                delayed(_one_perm_job)(
                    seed, n, n1, coords, fil, res, gain, n_clusters, km_random_state
                )
                for seed in range(num_perm)
            )

            # ===== Collect results =====
            perm_T = np.array([t[0] for t in perm_stats])
            perm_parts = np.array([t[1] for t in perm_stats])

            p_max = (np.sum(perm_T >= Tobs) + 1) / (num_perm + 1)
            p_parts = [
                (np.sum(perm_parts[:, k] >= parts_obs[k]) + 1) / (num_perm + 1)
                for k in range(len(TOPO_TYPES))
            ]
            print(f"🟥 p (unbiased): max={p_max:.4f}  parts="
                  f"{dict(zip(TOPO_TYPES, [f'{x:.4f}' for x in p_parts]))}")

            # ===== Save results =====
            results[(res, gain)] = {
                "Tobs_max": Tobs,
                "Tobs_parts": dict(zip(TOPO_TYPES, parts_obs)),
                "perm_T": perm_T,
                "perm_parts": perm_parts,
                "p_max": p_max,
                "p_parts": dict(zip(TOPO_TYPES, p_parts)),
            }

    return results

# =========================
# Main entry (adjust filenames/parameters as needed)
# =========================
if __name__ == "__main__":
    data1 = pd.read_csv("complete_case.csv")
    data2 = pd.read_csv("imputed_data.csv")
    
    #Set "resolution" and "gain"
    res_list = [5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35]
    gain_list = [0.1, 0.2, 0.3, 0.4]

    results = run_mapper_analysis_permutation_statmapper(
        data1, data2,
        res_list, gain_list,
        num_perm=499,        # user configurable
        knn_k=35,
        n_clusters=4,
        km_random_state=0,
        n_jobs=18            # parallelism
    )
    
    # === Generate reports (CSV/MD/PKL) in one shot ===
    write_run_reports(
        results,
        meta={
            "num_perm": 499,
            "knn_k": 20,
            "n_jobs": 12,
            "n_clusters": 4,
            "res_list": res_list,
            "gain_list": gain_list
        },
        out_dir="run_reports",
        basename="mapper_perm"
    )





