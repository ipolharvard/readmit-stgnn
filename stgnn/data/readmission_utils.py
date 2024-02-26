import os
import pickle
import bisect
from itertools import chain, islice

import numpy as np
from joblib import Parallel, delayed, Memory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sortedcontainers import SortedList
from tqdm import tqdm

memory = Memory(location=".", verbose=0)


def balanced_splits(n, k):
    total_comparisons = n * (n + 1) // 2
    target_per_split = total_comparisons // k

    splits = []
    current_start = 0
    while current_start < n:
        current_end = current_start + 1
        while current_end <= n and ((current_end - current_start) * (
                n - current_start - (current_end - current_start) / 2)) < target_per_split:
            current_end += 1

        splits.append((current_start, min(current_end, n)))
        current_start = current_end

        remaining_splits = k - len(splits)
        if remaining_splits > 0:
            completed_comparisons = sum((e - s) * (n - s - (e - s) / 2) for s, e in splits)
            target_per_split = (total_comparisons - completed_comparisons) // remaining_splits
    return splits


@memory.cache(ignore=["demo_dict"])
def compute_dist_mat(demo_dict, scale=False):
    """
    Args:
        demo_dict: dict, key is node name, value is EHR feature vector
        scale: if True, will perform min-max scaling.
    Returns:
        dist_dict: dict of pairwise distances between nodes
    """

    demo_arr = []
    for _, arr in demo_dict.items():
        demo_arr.append(arr)
    demo_arr = np.stack(demo_arr, axis=0)

    # Scaler to scale each continuous variable to be between 0 and 1
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(demo_arr)
        demo_arr = scaler.transform(demo_arr)

    node_names = list(demo_dict.keys())

    print(f"This many to go: {len(node_names) * (len(node_names) + 1) // 2:,}")
    n_jobs = 1
    # splits = balanced_splits(len(node_names), n_jobs)
    splits = balanced_splits(100, n_jobs)

    def worker(start_idx, stop_idx, i):
        total = stop_idx - start_idx
        total *= (len(node_names) - start_idx + 1) // 2

        idx_generator = tqdm((
            (idx_node1, idx_node2)
            for idx_node1 in range(start_idx, stop_idx)
            for idx_node2 in range(idx_node1, 100)
        ), position=i, disable=i >= 10, total=total)
        max_len = int(0.01 * total)
        results = SortedList(key=lambda x: x[2])
        while True:
            for idx_node1, idx_node2 in islice(idx_generator, max_len):
                results.add(
                    (
                        np.int32(idx_node1),
                        np.int32(idx_node2),
                        np.float32(np.linalg.norm(demo_arr[idx_node1] - demo_arr[idx_node2]))
                    )
                )
            if len(results) < max_len:
                break
            del results[slice(-1, max_len - 1, -1)]
        return results

    result = Parallel(n_jobs=n_jobs, verbose=1000)(
        delayed(worker)(start_idx, stop_idx, i) for i, (start_idx, stop_idx) in
        enumerate(splits))

    src_nodes, dst_nodes, dist = list(zip(*chain.from_iterable(result)))
    return {"From": src_nodes, "To": dst_nodes, "Distance": dist}


def compute_cos_sim_mat(demo_dict, scale=False):
    """
    Args:
        demo_dict: key is patient, value is EHR feature vector
        scale: if True, will perform min-max scaling
    Returns:
        cos_sim_dict: dict of pairwise cosine similarity between nodes
    """

    # Scaler to scale each variable to be between 0 and 1
    demo_arr = []
    for _, arr in demo_dict.items():
        demo_arr.append(arr)
    demo_arr = np.stack(demo_arr, axis=0)

    # Scaler to scale each continuous variable to be between 0 and 1
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(demo_arr)
        demo_arr = scaler.transform(demo_arr)

    cos_sim = cosine_similarity(demo_arr, demo_arr)

    cos_sim_dict = {"From": [], "To": [], "CosineSim": [], "Mask": []}
    node_names = list(demo_dict.keys())
    for idx_node1 in tqdm(range(len(node_names))):
        for idx_node2 in range(idx_node1, len(node_names)):
            node1 = node_names[idx_node1]
            node2 = node_names[idx_node2]

            cos_sim_dict["From"].append(node_names[idx_node1])
            cos_sim_dict["To"].append(node_names[idx_node2])
            cos_sim_dict["CosineSim"].append(cos_sim[idx_node1, idx_node2])

    return cos_sim_dict


def get_feat_seq(
        node_names,
        feature_dict,
        max_seq_len,
        pad_front=False,
        time_deltas=None,
        padding_val=None,
):
    """
    Args:
        node_names: dict, key is node name, value is list of imaging files
        feature_dict: dict, key is node name, value is imaging/EHR feature vector
        max_seq_len: int, maximum sequence length
        pad_front: if True, will pad to the front with the first timestep, else pad to the end with the last timestep
        time_deltas: if not None, will pad time deltas to features
        padding_val: if not None, will pad with this value instead of last/first time step
    Returns:
        padded_features: numpy array, shape (sample_size, max_seq_len, feature_dim)
        seq_len: original sequence length without any padding
    """
    padded_features = []
    padded_time_deltas = []
    for name in node_names:
        feature = feature_dict[name]
        feature = feature[-max_seq_len:, :]  # get last max_seq_len time steps

        if time_deltas is not None:
            time_dt = time_deltas[name][-max_seq_len:]  # (max_seq_len,)
            assert len(time_dt) == feature.shape[0]

        if feature.shape[0] < max_seq_len:
            if not pad_front:
                # pad with last timestep or padding_val
                if padding_val is None:
                    padded = np.repeat(
                        feature[-1, :].reshape(1, -1),
                        repeats=max_seq_len - feature.shape[0],
                        axis=0,
                    )
                else:
                    padded = (
                            np.ones((max_seq_len - feature.shape[0], feature.shape[1]))
                            * padding_val
                    )
                feature = np.concatenate([feature, padded], axis=0)
                if time_deltas is not None:
                    padded_dt = np.zeros((max_seq_len - time_dt.shape[0]))
                    time_dt = np.concatenate([time_dt, padded_dt], axis=0)
            else:
                # pad with first timestep or padding_val
                if padding_val is None:
                    padded = np.repeat(
                        feature[0, :].reshape(1, -1),
                        repeats=max_seq_len - feature.shape[0],
                        axis=0,
                    )
                else:
                    padded = (
                            np.ones((max_seq_len - feature.shape[0], feature.shape[1]))
                            * padding_val
                    )
                feature = np.concatenate([padded, feature], axis=0)
                if time_deltas is not None:
                    padded_dt = np.zeros((max_seq_len - time_dt.shape[0]))
                    time_dt = np.concatenate([padded_dt, time_dt], axis=0)
        padded_features.append(feature)
        if time_deltas is not None:
            padded_time_deltas.append(time_dt)

    padded_features = np.stack(padded_features)
    if time_deltas is not None:
        padded_time_deltas = np.expand_dims(np.stack(padded_time_deltas), axis=-1)
        padded_features = np.concatenate([padded_features, padded_time_deltas], axis=-1)

    return padded_features


def get_img_features(feature_dir, node_included_files):
    """
    Args:
        feature_dir: dir to imaging features
        node_included_files: dict, key is node name, value is list of image paths
    Returns:
        img_feature_dict: dict, key is image path, value is image features within one hospitalization, shape (num_cxrs, feature_dim)
    """

    img_feature_dict = {}
    for name, files in tqdm(node_included_files.items()):
        curr_feat = []

        for img_dir in files:
            with open(
                    os.path.join(feature_dir, img_dir.split("/")[-1] + ".pkl"), "rb"
            ) as pf:
                feature = pickle.load(pf)
            curr_feat.append(feature)
        curr_feat = np.stack(curr_feat, axis=0)  # (num_cxrs, feature_dim)
        img_feature_dict[name] = curr_feat

    return img_feature_dict


def get_time_varying_edges(
        node_names,
        edge_dict,
        edge_modality,
        hospital_stay,
        cpt_dict=None,
        icd_dict=None,
        lab_dict=None,
        med_dict=None,
):
    """
    Args:
        node_names: list, with node names
        edge_dict: dict, key is node name, value is EHR features
        edge_modality: list of EHR sources for edges
        hospital_stay: numpy array, lengths of hospital stays, shape (sample_size,)
        cpt_dict: dict, key is node name, value is preprocessed CPT features
        icd_dict: dict, key is node name, value is preprocessed ICD features
        lab_dict: dict, key is node name, value is preprocessed lab features
        med_dict: dict, key is node name, value is preprocessed medication features
    Returns:
        edge_dict: dict, key is node name, value is EHR features for edges
    """
    if edge_dict is None:
        edge_dict = {}

    for i, name in enumerate(node_names):
        if name not in edge_dict:
            edge_dict[name] = []
        # for cpt or icd or med, we sum over all days & average by length of stay (in days)
        if "cpt" in edge_modality:
            edge_dict[name] = np.concatenate(
                [
                    edge_dict[name],
                    np.sum(cpt_dict[name], axis=0) / hospital_stay[name],
                ],
                axis=-1,
            )
        if "icd" in edge_modality:
            # MIMIC ICD is non-temporal
            edge_dict[name] = np.concatenate(
                [
                    edge_dict[name],
                    icd_dict[name][-1, :] / hospital_stay[name],
                ],
                axis=-1,
            )
        if "med" in edge_modality:
            edge_dict[name] = np.concatenate(
                [
                    edge_dict[name],
                    np.sum(med_dict[name], axis=0) / hospital_stay[name],
                ],
                axis=-1,
            )
        # NOTE: for lab, take the last time step
        if "lab" in edge_modality:
            edge_dict[name] = np.concatenate(
                [edge_dict[name], lab_dict[name][-1, :]], axis=-1
            )
    return edge_dict


def compute_edges(
        dist_dict,
        top_perc=0.1,
        gauss_kernel=True,
):
    """
    Computes edge weights
    Args:
        dist_dict: dict with computed distance measures between nodes
        node_names: list of node names
        top_perc: top percentage of edges to be kept
        gauss_kernel: if True, will apply Gaussian kernel to Euclidean distance measures
    Returns:
        edges: numpy array of edge weights, shape (num_edges,)
    """
    src_nodes, dst_nodes, dist = dist_dict.values()

    # sanity check shape, (num_nodes) * (num_nodes + 1) / 2, if consider self-edges
    # assert len(dist) == (len(node_names) * (len(node_names) + 1) / 2)

    # apply gaussian kernel, use cosine distance instead of cosine similarity
    std = dist.std()
    edges = np.exp(-np.square(dist / std))

    # mask the edges
    if top_perc is not None:
        num = len(edges)
        num_to_keep = int(num * top_perc)
        sorted_dist = np.sort(edges)[::-1]  # descending order
        thresh = sorted_dist[:num_to_keep][-1]
        mask = edges >= thresh & (edges > 0)
        edges = edges[mask]
        src_nodes = src_nodes[mask]
        dst_nodes = dst_nodes[mask]

    return src_nodes, dst_nodes, edges


def get_readmission_label_mimic(df_demo):
    """
    Args:
        df_demo: dataframe with patient readmission info and demographics:
        max_seq_len: maximum number of cxrs to use, count backwards from last cxr within the hospitalization
                    if max_seq_len=None, will use all cxrs
    Returns:
        labels: numpy array, readmission labels, same order as rows in df_demo, shape (num_admissions,)
        node_included_files: dict, key is node name, value is list of image files
        label_splits: list indicating the split of each datapoint in labels and node_included_files
        time_deltas: dict, key is node name, value is an array of day difference between currenct cxr to previous cxr
        total_stay: dict, key is node name, value is total length of stay (in days)
        time_idxs: dict, key is node name, value is the index of each cxr in terms of the day within hospitalization
    """
    labels = df_demo.readmitted
    node_names = (df_demo.subject_id.astype(str) + "_" + df_demo.hadm_id.astype(str)).tolist()
    label_splits = df_demo.splits.values
    total_stay = dict(zip(node_names, df_demo.readmission_gap_in_days))

    return (
        np.asarray(labels),
        node_names,
        label_splits,
        total_stay,
    )
