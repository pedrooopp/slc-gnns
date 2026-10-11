from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger("gnn_streamflow")

def build_windowed_graphs(
    x_all,                     # [num_nodes, T, num_features]
    edge_index,                # [2, num_edges]
    window,
    target_feature_idx,        # feature to predict
    target_node_idx,           # None → predict for all nodes
    edge_weight=None,          # [num_nodes, num_features]
    use_streamflow_as_input=False
):

    num_nodes, T, num_features = x_all.shape

    data_list = []

    if (window == 1):

        for t in range(T - window):
            # Input
            x_window = x_all[:,t,:].clone()  # shape: [num_nodes, window, num_features]

            if not use_streamflow_as_input:
                x_window[:, target_feature_idx] = 0 # Don't use target as input (set it to zero)

            # Target timestep
            y_full = x_all[:, t + window, target_feature_idx]  # shape: [num_nodes] 
            # NOTA: Antes habia un bug que consideraba para la predicción cuando window=1 el caudal actual

            # If: 
            #   target_node_idx = None --> Predict all variables
            #   target_node_idx = idx --> Predict 'idx' variable
            if target_node_idx is not None:
                y = y_full[target_node_idx]  # shape: scalar
            else:
                y = y_full  # shape: [num_nodes]

            data = Data(
                x=x_window,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=y
            )

            data_list.append(data)

    else:
        for t in range(T - window):

            # Input window
            x_window = x_all[:, t:t+window, :].clone()  # shape: [num_nodes, window, num_features]

            if not use_streamflow_as_input:
                x_window[:, :, target_feature_idx] = 0 # Don't use target as input (set it to zero)

            # Target timestep
            y_full = x_all[:, t+window, target_feature_idx]  # shape: [num_nodes]
            
            # If: 
            #   target_node_idx = None --> Predict all variables
            #   target_node_idx = idx --> Predict 'idx' variable
            if target_node_idx is not None:
                y = y_full[target_node_idx]  # shape: scalar
            else:
                y = y_full  # shape: [num_nodes]

            data = Data(
                x=x_window,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=y
            )

            data_list.append(data)

    return data_list

def IDW_Calc(d, p=2):
    d = np.array(d, dtype=float)
    
    weights = 1 / (d ** p)
    weights /= weights.sum()
    
    return weights

def calc_weights(func, edge_index, coords_list):
    row, col = edge_index

    coords = torch.tensor(coords_list, dtype=torch.float)

    distances = torch.linalg.norm(
        coords[row] - coords[col],
        dim=1
    )

    # ---- Prevent extremely small distances ----
    min_dist = distances.quantile(0.1)  # 10th percentile
    distances = torch.clamp(distances, min=min_dist)

    if func=="idw":
        # ---- Gaussian weighting ----
        weights = torch.tensor(IDW_Calc(distances), dtype=torch.float32)

    elif func=="gauss":

        # ---- Gaussian weighting ----
        sigma = distances.mean()

        weights = torch.exp(
            - (distances ** 2) / (2 * sigma ** 2)
        )

        # Normalize to [0,1]
        weights = (weights - weights.min()) / (
            weights.max() - weights.min()
        )

    else:
        raise ValueError(f"Wrong weight func '{func}', func must be 'idw', 'dist' or 'gauss'")

    dist_weight = weights.view(-1, 1)


    return dist_weight

# Graph variants >= 5 and <= 10 use MAP
GRAPH_DEFINITIONS = {
        1: [[1, 2, 3, 4, 5, 6],
            [0, 0, 0, 0, 0, 0]],

        2: [[0, 1, 2, 3, 4, 5, 6],
            [0, 0, 0, 0, 0, 0, 0]],

        3: [[2, 3, 4, 5, 6, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 2, 3, 4, 5, 6]],

        4: [[0, 2, 3, 4, 5, 6, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6]],
                
        5: [[1, 2],
            [0, 0]],

        6: [[0, 1, 2],
            [0, 0, 0]],
        
        7: [[1, 2],
            [2, 0]],
        
        8: [[0, 1, 2],
            [0, 2, 0]],

        9: [[2],
            [0]],
        
        10: [[0, 2],
            [0, 0]],
        
        11: [[2, 3, 4, 5, 6],
            [0, 0, 0, 0, 0]],

        12: [[0, 2, 3, 4, 5, 6],
            [0, 0, 0, 0, 0, 0]],

        13: [[0],
            [0]]
    }

def define_graph(graph_variant: int, coords_list: list, weight_type: str="IDW"):

    if graph_variant not in GRAPH_DEFINITIONS:
        raise ValueError(
            f"Unknown graph_variant={graph_variant}. "
            f"Available variants: {sorted(GRAPH_DEFINITIONS)}"
        )

    edge_index = torch.tensor(
        GRAPH_DEFINITIONS[graph_variant],
        dtype=torch.long,
    )

    # if use_weights:
    #     dist_weight = calc_weights(weight_type, edge_index, coords_list)
    #     edge_weight = dist_weight/dist_weight.sum() # Distance normalization --> edge weights
    # else: 
    #     edge_weight = None

    # NOTA: Cambie que siempre se calcule los pesos así puedo despues dentro de optuna optimizar use_weigths. 
    # Ahora siempre se calculan los pesos y dentro de los modelos se define si se usan o no.
    dist_weight = calc_weights(weight_type, edge_index, coords_list)
    edge_weight = dist_weight/dist_weight.sum() # Distance normalization --> edge weights

    return edge_index, edge_weight


def load_data(data_dir, graph_variant, start_date, end_date):

    # TODO: Hardcodeado si usa MAP o no. Arreglar
    if graph_variant <= 4 or 11 <= graph_variant <= 12:
        logger.debug("Using normal precipitation stations.")
        use_map = False
    elif 5 <= graph_variant <= 10:
        logger.debug("Using MAP data.")
        use_map = True
    elif graph_variant==13:
        df_data_all = pd.read_parquet(data_dir/"SLC_DINAGUA_2011_2023.parquet")
        stations_coords = pd.read_csv(data_dir/"other_data/SL_stations.csv")
        logger.debug(f"Using graph number {graph_variant}, which only has streamflow as input.")
        df_data_all = df_data_all.drop(columns=["[Florida (Puente Ruta 5)] Nivel de agua (h) [m]"]) # Drop nivel (= Caudal)

        return df_data_all, stations_coords
    else:
        raise ValueError(f"Unexpected graph variant, must be greater than 0 and less than 13: {graph_variant}.")
    
    if use_map:
        df_data_all = pd.read_parquet(data_dir/"SLC_MAP_all_data_2011_2022.parquet")
    else:
        df_data_all = pd.read_parquet(data_dir/"SLC_all_data_2011_2022.parquet")
    
    df_data_all = df_data_all.drop(columns=["[Florida (Puente Ruta 5)] Nivel de agua (h) [m]"]) # Drop nivel (= Caudal)
    df_data_all = df_data_all.loc[start_date:end_date]

    stations_coords = pd.read_csv(data_dir/"other_data/SL_stations.csv")

    return df_data_all, stations_coords