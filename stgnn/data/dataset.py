import pickle

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from sklearn.preprocessing import StandardScaler

from .readmission_utils import get_readmission_label_mimic, get_feat_seq, \
    get_time_varying_edges, compute_dist_mat, compute_edges


def construct_graph_readmission(
        df_demo,
        ehr_feature_file=None,
        edge_ehr_file=None,
        edge_modality=("demo",),
        top_perc=0.01,
        gauss_kernel=True,
        standardize=True,
        max_seq_len_ehr=9,
        ehr_types=("demo", "icd", "lab", "med"),
        is_graph=True,
):
    """
    Construct an admission graph
    Args:
        df_demo: dataframe of cohort with demographic and imaging information
        ehr_feature_file: file of preprocessed EHR feature
        edge_ehr_file: file of preprocessed EHR feature for edges
        edge_modality: list of EHR sources for edge
        top_perc: top percentage edges to be kept for graph
        gauss_kernel: whether to use Gaussian kernel for edge weights
        standardize: whether to standardize node features
        max_seq_len_ehr: maximum sequence length of EHR features
        ehr_types: list of EHR sources for node features
        is_graph: whether to construct graph, we dont use it because the graph does not work with
        our cohort
    Returns:
        node2idx: dict, key is node name, value is node index
        dgl_G: dgl graph
        cat_idxs: list of categorical column indices
        cat_dims: list of categorical dimensions
    """

    # node labels
    (
        labels,
        node_names,
        splits,
        hospital_stays,
    ) = get_readmission_label_mimic(df_demo)
    train_idxs = np.array([ind for ind in range(len(splits)) if splits[ind] == "train"])

    # node name to node index dict
    node2idx = {name: idx for idx, name in enumerate(node_names)}

    train_masks = torch.from_numpy(splits == "train")
    val_masks = torch.from_numpy(splits == "val")
    test_masks = torch.from_numpy(splits == "test")

    with open(ehr_feature_file, "rb") as pf:
        raw_feat_dict = pickle.load(pf)
    feat_dict = raw_feat_dict["feat_dict"]
    feat_cols = raw_feat_dict["feature_cols"]
    cols_to_keep = []

    for ehr_name in ehr_types:
        cols_to_keep = cols_to_keep + raw_feat_dict["{}_cols".format(ehr_name)]

    col_idxs = np.array(
        [feat_cols.index(col) for col in cols_to_keep]
    )  # wrt original cols
    feat_dict = {
        name: feat_dict[name][:, col_idxs] for name in node_names
        if name in feat_dict
    }  # get relevant cols
    node_names = list(feat_dict.keys())
    node_features = get_feat_seq(
        node_names,
        feat_dict,
        max_seq_len_ehr,
        pad_front=False,
        time_deltas=None,
        padding_val=None,
    ).astype(np.float32)

    if "cat_idxs" in raw_feat_dict:
        cat_col2dim = {
            feat_cols[raw_feat_dict["cat_idxs"][ind]]: raw_feat_dict["cat_dims"][
                ind
            ]
            for ind in range(len(raw_feat_dict["cat_dims"]))
        }

        # reindex categorical variables
        cat_cols = [
            col
            for col in cols_to_keep
            if (feat_cols.index(col) in raw_feat_dict["cat_idxs"])
        ]
        cat_idxs = [cols_to_keep.index(col) for col in cat_cols]
        cat_dims = [cat_col2dim[col] for col in cat_cols]
    else:
        cat_idxs = []
        cat_dims = []

    print("Node features shape:", node_features.shape)
    assert np.all(node_features != -1)
    assert node_features.shape[1] == max_seq_len_ehr
    del feat_dict

    # standardize
    if standardize:
        print("Standardizing features...")

        num_features = node_features.shape[-1]
        train_feat = node_features[train_idxs].reshape((-1, num_features))

        scaler = StandardScaler()
        scaler.fit(train_feat)

        node_features = scaler.transform(node_features.reshape((-1, num_features)))
        # reshape back
        node_features = node_features.reshape((-1, max_seq_len_ehr, num_features))

    if not is_graph:
        print("Not constructing graph...")
        dummy_graph = dgl.graph(([], []), num_nodes=len(node_names))
        dummy_graph.ndata["train_mask"] = train_masks
        dummy_graph.ndata["val_mask"] = val_masks
        dummy_graph.ndata["test_mask"] = test_masks
        dummy_graph.ndata["label"] = torch.from_numpy(labels)
        dummy_graph.ndata["feat"] = torch.from_numpy(node_features)
        return node2idx, dummy_graph, labels, splits

    if (
            ("demo" in edge_modality)
            or ("cpt" in edge_modality)
            or ("icd" in edge_modality)
            or ("lab" in edge_modality)
            or ("imaging" in edge_modality)
            or ("med" in edge_modality)
    ):
        assert edge_ehr_file is not None
        with open(edge_ehr_file, "rb") as pf:
            raw_ehr_dict = pickle.load(pf)
        feat_cols = raw_ehr_dict["feature_cols"]
        node_edge_dict = {}

        if "demo" in edge_modality:
            demo_col_idxs = np.array(
                [feat_cols.index(col) for col in raw_ehr_dict["demo_cols"]]
            )  # wrt original cols
            demo_dict = {
                name: raw_ehr_dict["feat_dict"][name][0, demo_col_idxs]
                for name in node_names
            }
            for name in node_names:
                if name not in node_edge_dict:
                    node_edge_dict[name] = []
                node_edge_dict[name] = np.concatenate(
                    [node_edge_dict[name], demo_dict[name]], axis=-1
                )

        if "med" in edge_modality:
            med_col_idxs = np.array(
                [feat_cols.index(col) for col in raw_ehr_dict["med_cols"]]
            )  # wrt original cols
            med_dict = {
                name: raw_ehr_dict["feat_dict"][name][0, med_col_idxs]
                for name in node_names
            }
            for name in node_names:
                if name not in node_edge_dict:
                    node_edge_dict[name] = []
                node_edge_dict[name] = np.concatenate(
                    [node_edge_dict[name], med_dict[name]], axis=-1
                )

        # time varying edges
        if (
                ("cpt" in edge_modality)
                or ("icd" in edge_modality)
                or ("lab" in edge_modality)
                or ("imaging" in edge_modality)
                or ("med" in edge_modality)
        ):
            if "cpt" in edge_modality:
                cpt_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["cpt_cols"]]
                )  # wrt original cols
                cpt_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, cpt_col_idxs]
                    for name in node_names
                }
            else:
                cpt_dict = None
            if "icd" in edge_modality:
                icd_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["icd_cols"]]
                )  # wrt original cols
                icd_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, icd_col_idxs]
                    for name in node_names
                }
            else:
                icd_dict = None
            if "lab" in edge_modality:
                lab_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["lab_cols"]]
                )  # wrt original cols
                lab_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, lab_col_idxs]
                    for name in node_names
                }
            else:
                lab_dict = None
            if "med" in edge_modality:
                med_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["med_cols"]]
                )  # wrt original cols
                med_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, med_col_idxs]
                    for name in node_names
                }
            else:
                med_dict = None
            ehr_edge_dict = get_time_varying_edges(
                node_names=node_names,
                edge_dict=node_edge_dict,
                edge_modality=edge_modality,
                hospital_stay=hospital_stays,
                cpt_dict=cpt_dict,
                icd_dict=icd_dict,
                lab_dict=lab_dict,
                med_dict=med_dict,
            )
        else:
            ehr_edge_dict = node_edge_dict
    else:
        ehr_edge_dict = {}

    print("Using 'euclidean distance' for similarity/distance measure...")
    ehr_dist_dict = compute_dist_mat(ehr_edge_dict, scale=True)

    # Construct graphs
    src_nodes, dst_nodes, weights = compute_edges(
        ehr_dist_dict,
        top_perc=top_perc,
        gauss_kernel=gauss_kernel,
    )
    del ehr_dist_dict

    src_nodes = torch.from_numpy(src_nodes)
    dst_nodes = torch.from_numpy(dst_nodes)
    g_directed = dgl.graph((src_nodes, dst_nodes), idtype=torch.int32)
    g_directed.edata["weight"] = torch.FloatTensor(weights)

    dgl_G = dgl.add_reverse_edges(g_directed, copy_ndata=True, copy_edata=True)
    dgl_G = dgl.to_simple(dgl_G, return_counts=None, copy_ndata=True, copy_edata=True)

    num_nodes = dgl_G.num_nodes()

    dgl_G.ndata["train_mask"] = train_masks
    dgl_G.ndata["val_mask"] = val_masks
    dgl_G.ndata["test_mask"] = test_masks

    dgl_G.ndata["label"] = torch.FloatTensor(labels)

    dgl_G.ndata["feat"] = torch.FloatTensor(node_features)

    return node2idx, dgl_G, cat_idxs, cat_dims


class ReadmissionDataset(DGLDataset):
    def __init__(
            self,
            demo_file,
            edge_ehr_file=None,
            ehr_feature_file=None,
            edge_modality=("demo",),
            feature_type="multimodal",
            img_feature_dir=None,
            top_perc=None,
            gauss_kernel=False,
            max_seq_len_img=6,
            max_seq_len_ehr=8,
            sim_measure="euclidean",
            standardize=True,
            ehr_types=("demo", "cpt", "icd", "lab", "med"),
            is_graph=True,
    ):
        """
        Args:
            demo_file: file of cohort with demographic and imaging information
            ehr_feature_file: file of preprocessed EHR feature
            edge_ehr_file: file of preprocesdded EHR feature for edges
            edge_modality: list of EHR sources for edge
            feature_type: "multimodal", "imaging" or "non-imaging"
            img_feature_dir: dir to extracted imaging features
            top_perc: top percentage edges to be kept for graph
            gauss_kernel: whether to use Gaussian kernel for edge weights
            standardize: whether to standardize node features
            max_seq_len_img: maximum sequence length of imaging features
            max_seq_len_ehr: maximum sequence length of EHR features
            sim_measure: metric to measure node similarity for edges
            ehr_types: list of EHR sources for node features
        """
        self.demo_file = demo_file
        self.edge_modality = edge_modality
        self.feature_type = feature_type
        self.img_feature_dir = img_feature_dir
        self.ehr_feature_file = ehr_feature_file
        self.edge_ehr_file = edge_ehr_file
        self.top_perc = top_perc
        self.gauss_kernel = gauss_kernel
        self.max_seq_len_img = max_seq_len_img
        self.max_seq_len_ehr = max_seq_len_ehr
        self.sim_measure = sim_measure
        self.standardize = standardize
        self.ehr_types = ehr_types
        self.is_graph = is_graph

        if sim_measure not in ["cosine", "euclidean"]:
            raise NotImplementedError

        print("Edge modality:", edge_modality)
        print("EHR types:", ehr_types)

        # get patients
        self.df_all = pd.read_csv(demo_file)
        # use only these admissions where patient was discharged
        self.df_all = self.df_all.loc[lambda _df: ~_df.died]

        super().__init__(name="readmission")

    def process(self):
        (
            self.node2idx,
            self.graph,
            self.cat_idxs,
            self.cat_dims,
        ) = construct_graph_readmission(
            df_demo=self.df_all,
            ehr_feature_file=self.ehr_feature_file,
            edge_ehr_file=self.edge_ehr_file,
            edge_modality=self.edge_modality,
            top_perc=self.top_perc,
            gauss_kernel=self.gauss_kernel,
            standardize=self.standardize,
            max_seq_len_ehr=self.max_seq_len_ehr,
            ehr_types=self.ehr_types,
            is_graph=self.is_graph,
        )

        self.targets = self.graph.ndata["label"].cpu().numpy()

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
