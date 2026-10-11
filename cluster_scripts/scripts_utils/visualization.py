import logging
import matplotlib.pyplot as plt

logger = logging.getLogger("gnn_streamflow")

def plot_loss(train_loss_list, test_loss_list, num_epochs, graph_variant, save_path):

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(8,6))

    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, test_loss_list, label="Test Loss")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)

    title_suffix = f"graph {graph_variant}"

    plt.title(
        f"Train vs Test loss ({title_suffix})",
        fontsize=14
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_nse(train_nse_list, test_nse_list, num_epochs, graph_variant, save_path):

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(8,6))

    plt.plot(epochs, train_nse_list, label="Train NSE")
    plt.plot(epochs, test_nse_list, label="Test NSE")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("NSE", fontsize=12)

    title_suffix = f"graph {graph_variant}"

    plt.title(
        f"Train vs Test NSE ({title_suffix})",
        fontsize=14
    )
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_streamflow(df_data_all, y_pred_train, y_pred_test, target_col, window_size, train_ratio, graph_variant, save_path):

    y_true_all = df_data_all.loc[:, target_col].values

    # True values (NO SHUFFLE — keep temporal order)
    time = df_data_all.index

    split_index = int(len(y_true_all) * train_ratio)

    y_true_train = y_true_all[:split_index]
    y_true_test = y_true_all[split_index:]

    # Remove window offset (VERY IMPORTANT)
    samples_train = len(y_pred_train) # trust the real predictions, not manual math
    y_true_train = y_true_train[-samples_train:] # align truth to match

    samples_test = len(y_pred_test)
    y_true_test = y_true_test[-samples_test:]

    # --- Style settings
    plt.figure(figsize=(12, 6))

    # Colorblind-friendly colors
    color_true = "#1f77b4"   # blue
    color_pred = "#ff7f0e"   # orange

    lw_true = 2.2
    lw_pred = 2.0

    # --- Training data (solid lines)
    plt.plot(
        time[:samples_train],
        y_true_train,
        label="Train True",
        color=color_true,
        linewidth=lw_true,
        linestyle="-"
    )

    plt.plot(
        time[:samples_train],
        y_pred_train,
        label="Train Predicted",
        color=color_pred,
        linewidth=lw_pred,
        linestyle="-",
        alpha=0.75   # transparency helps overlap visibility
    )

    # --- Testing data (dashed lines)
    plt.plot(
        time[-samples_test:],
        y_true_test,
        color=color_true,
        linewidth=lw_true,
        linestyle="-"
    )

    plt.plot(
        time[-samples_test:],
        y_pred_test,
        color=color_pred,
        linewidth=lw_pred,
        linestyle="-",
        alpha=0.75
    )

    # --- Divider line
    plt.axvline(
        x=time[samples_train],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Train/Test Split"
    )

    # --- Labels and formatting
    plt.xlabel("Time", fontsize=13)
    plt.ylabel("Caudal (m³/s)", fontsize=13)

    title_suffix = f"graph {graph_variant}"

    plt.title(
        f"Caudal Time Series — Model Predictions ({title_suffix})",
        fontsize=15,
        pad=12
    )

    plt.legend(fontsize=11)
    plt.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        alpha=0.7
    )
    plt.margins(x=0.01)
    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()

    # --- Test-only figure
    plt.figure(figsize=(12, 5))

    # Time for test segment
    time_test = time[-samples_test:]

    # True and predicted
    plt.plot(
        time_test,
        y_true_test,
        label="Test True",
        color=color_true,
        linewidth=2.2
    )

    plt.plot(
        time_test,
        y_pred_test,
        label="Test Predicted",
        color=color_pred,
        linewidth=2.0,
        alpha=0.8
    )

    plt.title(
        f"Caudal Time Series — Test Set Only ({title_suffix})",
        fontsize=15,
        pad=12
    )

    plt.xlabel("Time", fontsize=13)
    plt.ylabel("Caudal (m³/s)", fontsize=13)

    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=11)

    plt.margins(x=0.01)
    plt.tight_layout()
    if save_path.endswith(".png"):
        save_path_test = save_path[:-4] + "_only_test.png"
        plt.savefig(save_path_test)
    else:
        logger.error(f"Path no termina en '.png': {save_path}")

    plt.close()