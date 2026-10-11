import logging
import os
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
#import gc

from cluster_scripts.scripts_utils.config_args import get_gnn_training_args
from cluster_scripts.scripts_utils.visualization import plot_loss, plot_nse, plot_streamflow

from source.training_config import NUM_EPOCHS, TRAIN_RATIO
from source.utils import setup_logger, get_variant_config, get_model_variant
from source.graphs import GRAPH_DEFINITIONS, load_data
from source.gnn_training import streamflow_model_search

DEFAULT_WINDOW_SIZE = 36

if __name__ == "__main__":

    logging.captureWarnings(True)

    logger = setup_logger('gnn_streamflow', level=logging.DEBUG)

    parser = get_gnn_training_args(argparse.ArgumentParser(description="Search best GNN models for streamflow at Florida station."))

    args = parser.parse_args()

    experiment_mode = args.experiment_mode
    
    model_type = args.model
    weight_type = args.weight_type

    output_dir = args.results_dir
    data_dir = Path(args.data_dir)

    run_all_graphs = args.run_all_graphs
    graph_variant_pairs = args.graph_variant_pairs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running training using: '{device}'")
    
    if graph_variant_pairs:
        # Graphs are implied by the pairs themselves, the run-all/run-graph-number args are irrelevant.
        graph_variant_list = sorted({g for g, _ in graph_variant_pairs})
    elif not run_all_graphs:
        graph_variant_list = args.run_graph_number

        if not graph_variant_list:
            raise ValueError("If run_all_graphs=False then 'graph_variant' must NOT be None.")
    else:
        graph_variant_list = list(GRAPH_DEFINITIONS.keys())

    date_format = "%Y-%m-%d" 
    start_date = args.start_date
    end_date = args.end_date
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    logger.info(f'Running streamflow model search (model type {model_type}) for graphs {graph_variant_list}. '
                f'Dates: [{start_date} to {end_date}]')
    
    results_df = pd.DataFrame(
        columns=[
            "model_type",
            "graph_variant",
            "model_variant",
            "loss_train",
            "loss_test",
            "nse_train",
            "nse_test",
        ]
    )

    # Makes sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    #input_window_list = [12, 24, 36]
    input_window_list = [DEFAULT_WINDOW_SIZE] # Por ahora optimizo con este tamaño solamente (muy costoso sino)

    if DEFAULT_WINDOW_SIZE not in input_window_list:
        logger.warning(f"Default window size ({DEFAULT_WINDOW_SIZE}) isn't in input_window_list")

    # Algunas consideraciones especiales segun el modelo a entrenar
    # TODO: Ver si se puede organizar mas lindo esto
    if model_type=="conv":
        # Modelo solamente convolutivo --> Ventana temporal = 1 (no tiene aspecto temporal)
        input_window_list = [1]
        logger.info(
            f"For model type '{model_type}' window size MUST be 1, "
            f"thus default window size of {DEFAULT_WINDOW_SIZE} was changed to "
            f"{input_window_list[0]}"
        )
    concat_model_list = ["attention", "attentionConv", "temporalAttention", "temporalAttentionConv", "recursiveAttention"]


    if experiment_mode=="auto":

        logger.info("Running experiments in auto mode (trying all model variants).")
        if model_type in concat_model_list:
            logger.info(
                f"For model type '{model_type}' output of attention cell can be concatenated, "
                f"thus models with concatenated and averaged attention output are trained")
            concat_list = [False, True]
        else:
            logger.debug("MODEL DOESN'T SUPPORT CONCAT ARG. IF THIS IS WRONG CHECK concat_model_list!!")
            concat_list = [False]

        for graph_variant in graph_variant_list:
            df_data_all, stations_coords = load_data(data_dir, graph_variant, start_date, end_date)
            for window_size in input_window_list: 
                for concat in concat_list:
                    for use_weights in [False, True]:
                        model_variant = get_model_variant(use_weights, concat)
                        logger.info(f"Now training graph variant {graph_variant} with use_weights={use_weights} and concat={concat} (model variant {model_variant}), and window_size={window_size}.")
                        logger.debug(f"use_weights={use_weights} and concat={concat}")

                        graph_dir = os.path.join(output_dir, f"{model_type}", f"graph_{graph_variant}") # One folder for each graph
                        os.makedirs(graph_dir, exist_ok=True)

                        save_model_path = os.path.join(graph_dir, f"{model_type}_graph{graph_variant}_window_size{window_size}_model_variant_{model_variant}.pth")

                        training_output = streamflow_model_search(df_data_all, stations_coords, graph_variant, model_type, weight_type, save_model_path, 
                                                                    window_size=window_size, use_weights=use_weights, concat=concat)
                        model, train_loss_list, test_loss_list, train_nse_list, test_nse_list, y_pred_train, y_pred_test, target_col = training_output
                        
                        if len(train_loss_list) == 0:
                            raise ValueError("Results list is empty, there was an error saving the results.")

                        print(f"Training loss: {train_loss_list[-1]:.4f}, NSE: {train_nse_list[-1]:.4f}")
                        print(f"Testing loss: {test_loss_list[-1]:.4f}, NSE: {test_nse_list[-1]:.4f}")       

                        # Store results
                        results_df.loc[len(results_df)] = [
                            model_type,
                            graph_variant,
                            model_variant,
                            train_loss_list[-1],
                            test_loss_list[-1],
                            round(train_nse_list[-1], 4),
                            round(test_nse_list[-1], 4),
                        ] 

                        # Plots
                        plot_loss(train_loss_list, test_loss_list, NUM_EPOCHS, graph_variant,
                                os.path.join(graph_dir, f"plot_loss_{model_type}_{graph_variant}_model_variant_{model_variant}.png"))
                        plot_nse(train_nse_list, test_nse_list, NUM_EPOCHS, graph_variant,
                                os.path.join(graph_dir, f"plot_nse_{model_type}_{graph_variant}_model_variant_{model_variant}.png"))
                        plot_streamflow(df_data_all, y_pred_train, y_pred_test, target_col, window_size, TRAIN_RATIO, graph_variant,
                                        os.path.join(graph_dir, f"plot_streamflow_{model_type}_{graph_variant}_model_variant_{model_variant}.png"))

                        logger.info(f"Training completed for graph variant {graph_variant} with window_size={window_size}.")
                    
                        logger.info(f"Training ended for graph {graph_variant} and model type {model_type}.")
                        #del df_data_all, training_output, model, y_pred_train, y_pred_test
                        #gc.collect()

    elif experiment_mode=="selective":

        logger.info(f"Running training for graph variant pairs: {graph_variant_pairs}")

        if graph_variant_pairs:
        
            for graph_variant, selected_variant in graph_variant_pairs:

                use_weights, concat = get_variant_config(selected_variant)

                if model_type in concat_model_list:
                    logger.info(
                        f"For model type '{model_type}' output of attention cell can be concatenated, "
                        f"thus models with concatenated and averaged attention output are trained")
                else:
                    logger.debug("MODEL DOESN'T SUPPORT CONCAT ARG. IF THIS IS WRONG CHECK concat_model_list!!")
                    concat = False

                logger.info(f"Running experiments in selective mode (variant {selected_variant} and graph {graph_variant}).")
                logger.debug(f"use_weights={use_weights} and concat={concat}")

                df_data_all, stations_coords = load_data(data_dir, graph_variant, start_date, end_date)

                for window_size in input_window_list: 
                    logger.info(f"Now training graph {graph_variant} and window_size={window_size}.")
                    

                    graph_dir = os.path.join(output_dir, f"{model_type}", f"graph_{graph_variant}") # One folder for each graph
                    os.makedirs(graph_dir, exist_ok=True)

                    save_model_path = os.path.join(graph_dir, f"{model_type}_graph{graph_variant}_window_size{window_size}_model_variant_{selected_variant}.pth")

                    training_output = streamflow_model_search(df_data_all, stations_coords, graph_variant, model_type, weight_type, save_model_path, 
                                                                window_size=window_size, use_weights=use_weights, concat=concat)
                    model, train_loss_list, test_loss_list, train_nse_list, test_nse_list, y_pred_train, y_pred_test, target_col = training_output
                    
                    if len(train_loss_list) == 0:
                        raise ValueError("Results list is empty, there was an error saving the results.")

                    print(f"Training loss: {train_loss_list[-1]:.4f}, NSE: {train_nse_list[-1]:.4f}")
                    print(f"Testing loss: {test_loss_list[-1]:.4f}, NSE: {test_nse_list[-1]:.4f}")       

                    # Store results
                    results_df.loc[len(results_df)] = [
                        model_type,
                        graph_variant,
                        selected_variant,
                        train_loss_list[-1],
                        test_loss_list[-1],
                        round(train_nse_list[-1], 4),
                        round(test_nse_list[-1], 4),
                    ] 

                    # Plots
                    plot_loss(train_loss_list, test_loss_list, NUM_EPOCHS, graph_variant,
                            os.path.join(graph_dir, f"plot_loss_{model_type}_{graph_variant}_model_variant_{selected_variant}.png"))
                    plot_nse(train_nse_list, test_nse_list, NUM_EPOCHS, graph_variant,
                            os.path.join(graph_dir, f"plot_nse_{model_type}_{graph_variant}_model_variant_{selected_variant}.png"))
                    plot_streamflow(df_data_all, y_pred_train, y_pred_test, target_col, window_size, TRAIN_RATIO, graph_variant,
                                    os.path.join(graph_dir, f"plot_streamflow_{model_type}_{graph_variant}_model_variant_{selected_variant}.png"))

                    logger.info(f"Training completed for graph variant {graph_variant} with window_size={window_size}.")
                
                    logger.info(f"Training ended for graph {graph_variant} and model type {model_type}.")
                    #del df_data_all, training_output, model, y_pred_train, y_pred_test
                    #gc.collect()

        else:
            raise ValueError("Must pass graph and variant!")
    
    else:
        raise ValueError(f"Unexpected experiment mode given in args: {experiment_mode}")

    # Export summary CSV
    results_csv_path = os.path.join(output_dir, f"{model_type}_results.csv")
    results_df = results_df.sort_values(by='nse_test', ascending=False)
    results_df.to_csv(
        results_csv_path,
        index=False,
        sep=";",
        decimal=","
    )

    logger.info("Script ended.")