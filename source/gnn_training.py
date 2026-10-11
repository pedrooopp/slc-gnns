from source.models import train, test, run_study, build_model
from source.graphs import build_windowed_graphs, define_graph
from source.utils import has_self_loop, extract_station
from source.training_config import NUM_EPOCHS, TRAIN_RATIO, TARGET_STATION, TARGET_VARIABLE, VAL_RATIO

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from collections import defaultdict
import re
import numpy as np
import json

import torch
from torch.utils.data import ConcatDataset

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("gnn_streamflow")

# TODO: Dividir esta función en sub-funciones (ver script forecasting)
def streamflow_model_search(df_data_all, stations_coords, graph_variant, model_type, weight_type, save_model_path, window_size, use_weights, concat):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_cols = list(df_data_all.columns)
    all_stations = [
        re.search(r"\[(.*?)\]", e).group(1)
        for e in all_cols
    ]
    station_order = list(dict.fromkeys(all_stations)) # Remove duplicates from list while preserving order of elements


    stations_coords = stations_coords[stations_coords["station"].isin(station_order)] # Keep only stations inside Florida's basin
    stations_coords = stations_coords.set_index('station') # Define new index

    # --- (1) Se definen las distancias ---

    coords_list = [
            (stations_coords.loc[s, 'longitud'],
            stations_coords.loc[s, 'latitud'])
        for s in station_order
    ]

    variable_order = [c.split("] ", 1)[1] for c in df_data_all.columns]
    variable_order = list(dict.fromkeys(variable_order))

    df_train, df_test = train_test_split(df_data_all,  train_size=TRAIN_RATIO, shuffle=False)
    df_train, df_val = train_test_split(df_train, train_size=VAL_RATIO, shuffle=False)  # e.g. 0.85/0.15 of df_train

    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()

    target_col = f"[{TARGET_STATION}] {TARGET_VARIABLE}"

    # -- TRAIN --
    df_train_scaled = df_train.copy()

    X_train = df_train_scaled.drop(columns=[target_col])
    y_train = df_train_scaled[[target_col]] 

    # Fit scalers
    if not X_train.empty: # When model is just recursive --> Only imput data is streamflow (y_train/y_test)
        X_train_scaled = scalerX.fit_transform(X_train)
    else:
        logger.warning("X_train dataframe is empty, this may be an error. If model is only recursive (only imput data is streamflow) this behaviour is expected.")
    y_train_scaled = scalerY.fit_transform(y_train)

    # Assign back
    if not X_train.empty: # When model is just recursive --> Only imput data is streamflow (y_train/y_test)
        df_train_scaled[X_train.columns] = X_train_scaled
    df_train_scaled[target_col] = y_train_scaled

    # -- TEST --
    df_test_scaled = df_test.copy()

    X_test = df_test_scaled.drop(columns=[target_col])
    y_test = df_test_scaled[[target_col]]

    # Transform only
    if not X_train.empty: # When model is just recursive --> Only imput data is streamflow (y_train/y_test)
        # If X_train is empty, then the scalerX isn't fitted!
        X_test_scaled = scalerX.transform(X_test)
    y_test_scaled = scalerY.transform(y_test)

    # Assign back
    if not X_train.empty: # When model is just recursive --> Only imput data is streamflow (y_train/y_test)
        df_test_scaled[X_test.columns] = X_test_scaled
    df_test_scaled[target_col] = y_test_scaled  

    # -- VAL --
    df_val_scaled = df_val.copy()

    X_val = df_val_scaled.drop(columns=[target_col])
    y_val = df_val_scaled[[target_col]]

    # Transform only
    if not X_train.empty:
        # If X_train is empty, then the scalerX isn't fitted!
        X_val_scaled = scalerX.transform(X_val)
    y_val_scaled = scalerY.transform(y_val)

    # Assign back
    if not X_train.empty: 
        df_val_scaled[X_val.columns] = X_val_scaled
    df_val_scaled[target_col] = y_val_scaled  


    num_graph_nodes = len(station_order)
    num_features = len(variable_order)
    num_time_train = len(df_train_scaled)
    num_time_test = len(df_test_scaled)
    num_time_val = len(df_val_scaled)

    X_all_train = torch.zeros(
        (num_graph_nodes, num_time_train, num_features),
        dtype=torch.float32
    )

    X_all_test = torch.zeros(
        (num_graph_nodes, num_time_test, num_features),
        dtype=torch.float32
    )

    X_all_val = torch.zeros(
        (num_graph_nodes, num_time_val, num_features),
        dtype=torch.float32
    )

    for i, station in enumerate(station_order):

        for j, variable in enumerate(variable_order):

            col_name = f"[{station}] {variable}"

            if col_name in df_train_scaled.columns:

                X_all_train[i, :, j] = torch.tensor(
                    df_train_scaled[col_name].values,
                    dtype=torch.float32
                )

                X_all_test[i, :, j] = torch.tensor(
                    df_test_scaled[col_name].values,
                    dtype=torch.float32
                )

                X_all_val[i, :, j] = torch.tensor(
                    df_val_scaled[col_name].values,
                    dtype=torch.float32
                )

            else:
                # missing variable at station
                X_all_train[i, :, j] = 0.0
                X_all_test[i, :, j] = 0.0   
                X_all_val[i, :, j] = 0.0   


    target_feature_idx = variable_order.index(TARGET_VARIABLE) # Variable --> Feature
    target_node_idx = station_order.index(TARGET_STATION) # Station --> Node

    edge_index, edge_weight = define_graph(graph_variant, coords_list, weight_type)

    # Check is use streamflow as input
    use_streamflow_as_input = has_self_loop(edge_index, target_node_idx)

    if use_streamflow_as_input:
        logger.debug("Self loop found in graph!")
    logger.debug(f"use_streamflow_as_input: {use_streamflow_as_input}")
    logger.debug(f"target_node_idx: {target_node_idx}")
    logger.debug(f"device: {device}")

    # Muevo datos a la GPU
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    X_all_train = X_all_train.to(device)
    X_all_test = X_all_test.to(device)
    X_all_val = X_all_val.to(device)

    dataset_train = build_windowed_graphs(
        X_all_train,
        edge_index,
        window=window_size,
        target_feature_idx=target_feature_idx,
        target_node_idx=target_node_idx,
        edge_weight=edge_weight,
        use_streamflow_as_input=use_streamflow_as_input
    )

    dataset_test = build_windowed_graphs(
        X_all_test,
        edge_index,
        window=window_size,
        target_feature_idx=target_feature_idx,
        target_node_idx=target_node_idx,
        edge_weight=edge_weight,
        use_streamflow_as_input=use_streamflow_as_input
    )

    dataset_val = build_windowed_graphs(
        X_all_val,
        edge_index,
        window=window_size,
        target_feature_idx=target_feature_idx,
        target_node_idx=target_node_idx,
        edge_weight=edge_weight,
        use_streamflow_as_input=use_streamflow_as_input
    )

    batch_size = 64

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    
    model_config = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": val_loader,
        "num_features": num_features,
        "target_node_idx": target_node_idx,
        "num_graph_nodes": num_graph_nodes,
        "scalerY": scalerY,
        "concat": concat,
        "use_weights": use_weights,
        "device": device,
    }

    study = run_study(model_type, model_config, n_trials=150)
 
    print(f"Best trial: #{study.best_trial.number} (Val NSE: {study.best_trial.value:.6f})")
    print("Best params:", study.best_trial.params)
 
    # 2) Rebuild the model with the best params and train it fully
    best_params = study.best_trial.params # Retorna parametros del mejor trial que tuvo mejor NSE en todas las epochs
    best_model = build_model(model_type, best_params, model_config)
    best_lr = best_params.get("lr", model_config.get("default_lr", 1e-3))

    optimizer_name = best_params["optimizer"]

    if optimizer_name=="Adam":
        best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)
    elif optimizer_name=="RMSprop":
        best_optimizer = torch.optim.RMSprop(best_model.parameters(), lr=best_lr)
    else:
        raise ValueError(f"Unexpected optimizer name (must be 'Adam' or 'RMSprop'), got '{optimizer_name}'")
    
    best_criterion = torch.nn.MSELoss()

    data_config = {
        "X_all_train": X_all_train,
        "X_all_val": X_all_val,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "window_size": window_size,
        "target_feature_idx": target_feature_idx,
        "use_streamflow_as_input": use_streamflow_as_input,
    }
    
    (
        model, train_loss_list, test_loss_list,
        train_nse_list, test_nse_list, y_pred_train, y_pred_test
    ) = train_full_model(
        best_model, best_criterion, best_optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        model_config=model_config,
        save_model_path=save_model_path,
        data_config=data_config,
    )

    # NOTE: Si cambio el ploteo de todo el Caudal y como lo hago, sacar el y_pred del retorno de la función 
    return model, train_loss_list, test_loss_list, train_nse_list, test_nse_list, y_pred_train, y_pred_test, target_col 

def train_full_model(model, criterion, optimizer, dataset_train, dataset_test, model_config, save_model_path, data_config):
    train_loader = model_config["train_loader"]
    val_loader = model_config["val_loader"]
    test_loader =  model_config["test_loader"]
    target_node_idx = model_config["target_node_idx"]
    num_graph_nodes = model_config["num_graph_nodes"]
    scalerY = model_config["scalerY"]
    use_weights = model_config["use_weights"]

    train_loss_list, test_loss_list = [], []
    train_nse_list, test_nse_list = [], []

    # ----- Combine train + val for final training -----
    X_all_train_full = torch.cat([data_config["X_all_train"], data_config["X_all_val"]], dim=1)  # concat along time axis

    full_train_dataset = build_windowed_graphs(
        X_all_train_full,
        data_config["edge_index"],
        window=data_config["window_size"],
        target_feature_idx=data_config["target_feature_idx"],
        target_node_idx=target_node_idx,
        edge_weight=data_config["edge_weight"],
        use_streamflow_as_input=data_config["use_streamflow_as_input"]
    )
    full_train_loader = DataLoader(full_train_dataset, batch_size=train_loader.batch_size, shuffle=False)
    
    for epoch in range(NUM_EPOCHS):
        loss, train_nse = train(model, criterion, optimizer, full_train_loader, use_weights, target_node_idx, num_graph_nodes, scalerY)
        test_loss, test_nse = test(model, criterion, test_loader, use_weights, target_node_idx, num_graph_nodes, scalerY)
 
        train_loss_list.append(loss)
        test_loss_list.append(test_loss)
        train_nse_list.append(train_nse)
        test_nse_list.append(test_nse)
 
        print(
            f"Epoch [{1 + epoch:03d}/{NUM_EPOCHS}] | "
            f"Train Loss: {loss:.6f} | "
            f"Train NSE: {train_nse:.2f} | "
            f"Test Loss: {test_loss:.6f} | "
            f"Test NSE: {test_nse:.2f}"
        )
 
    torch.save(model, save_model_path)
 
    # ----- Get predictions -----
    model.eval()
    y_pred_train = []
    y_pred_test = []
 
    with torch.no_grad():
        for data in full_train_dataset:
            if use_weights:
                out = model(data.x, data.edge_index, data.edge_weight)
            else:
                out = model(data.x, data.edge_index)
            pred = out[target_node_idx::num_graph_nodes]
            pred = pred.detach().cpu().numpy()
            pred_inv = scalerY.inverse_transform(pred.reshape(-1, 1))
            pred_inv[pred_inv < 0] = 0
            y_pred_train.append(pred_inv.squeeze())
 
        for data in dataset_test:
            if use_weights:
                out = model(data.x, data.edge_index, data.edge_weight)
            else:
                out = model(data.x, data.edge_index)
            pred = out[target_node_idx::num_graph_nodes]
            pred = pred.detach().cpu().numpy()
            pred_inv = scalerY.inverse_transform(pred.reshape(-1, 1))
            pred_inv[pred_inv < 0] = 0
            y_pred_test.append(pred_inv.squeeze())
     
    return model, train_loss_list, test_loss_list, train_nse_list, test_nse_list, y_pred_train, y_pred_test