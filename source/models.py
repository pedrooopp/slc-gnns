import numpy as np
import torch
import optuna
import logging

import torch.nn as nn
import torch.nn.functional as actF
from torch_geometric.nn import GraphConv, GATv2Conv
from torch.nn import GRU
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score as nse_metric
from source.training_config import NUM_EPOCHS


logger = logging.getLogger("gnn_streamflow")
logging.basicConfig()

def expand_edge_index_for_time(edge_index, edge_weight, T, N, device):
    """
    Replicate a graph's edges T times with node-index offsets, so that
    T independent timesteps can be processed as ONE big graph in a single
    conv call instead of T sequential calls.
    """
    offsets = torch.arange(T, device=device).repeat_interleave(edge_index.shape[1]) * N
    edge_index_big = edge_index.repeat(1, T) + offsets.unsqueeze(0)

    edge_weight_big = None
    if edge_weight is not None:
        if edge_weight.dim() == 1:
            # shape [num_edges] -> [num_edges * T]
            edge_weight_big = edge_weight.repeat(T)
        else:
            # shape [num_edges, edge_dim] -> [num_edges * T, edge_dim]
            repeat_dims = (T,) + (1,) * (edge_weight.dim() - 1)
            edge_weight_big = edge_weight.repeat(*repeat_dims)

    return edge_index_big, edge_weight_big

class RecursiveGNN(nn.Module):

    def __init__(self, num_features, gnn_hidden, rnn_hidden, out_channels, use_weights, dropout=0.0):
        super().__init__()

        self.gnn_hidden = gnn_hidden
        self.num_features = num_features
        self.use_weights = use_weights

        # NOTA: Con dropout (agrego a atención, si no tengo ninguno de los dos le agrego a conv)
        # Agrego capa dropout para usarla junto con la conv
        self.dropout = torch.nn.Dropout(dropout)

        self.gconv1 = GraphConv(num_features, gnn_hidden)
        #self.gconv2 = GraphConv(gnn_hidden, gnn_hidden)

        self.rnn = nn.GRU(
            input_size=gnn_hidden,
            hidden_size=rnn_hidden,
            batch_first=True,
        )

        self.fc = nn.Linear(
            rnn_hidden,
            out_channels,
        )

    def forward(self, x, edge_index, edge_weight=None):

        # x:
        # [num_nodes, time_window, features]
        N, T, Fdim = x.shape

        use_weights = self.use_weights

        if use_weights and edge_weight is None:
            raise ValueError("Model was configured with use_weights=True, but edge_weight=None was passed to the forward pass.")

        if not use_weights and edge_weight is not None:
            raise ValueError("Model was configured with use_weights=False, but edge_weight was provided to the forward pass.")

        edge_index_big, edge_weight_big = expand_edge_index_for_time(
            edge_index, edge_weight, T, N, x.device
        )

        # [N, T, F] -> [T, N, F] -> [T*N, F]  (t-major to match edge index construction)
        x_flat = x.transpose(0, 1).reshape(T * N, Fdim)
        
        h = self.gconv1(x_flat, edge_index_big, edge_weight=edge_weight_big)
        h = actF.relu(h)
        h = self.dropout(h)

        # back to [N, T, hidden]
        h = h.view(T, N, self.gnn_hidden).transpose(0, 1)

        h, _ = self.rnn(h)
        h_last = h[:, -1, :]
        out = self.fc(h_last)
        return out.squeeze(-1)

class ConvGNN(nn.Module):

    def __init__(self, num_features, gnn_hidden, out_channels, use_weights, dropout=0.0):
        super().__init__()

        self.gnn_hidden = gnn_hidden
        self.num_features = num_features
        self.use_weights = use_weights

        self.gconv1 = GraphConv(num_features, gnn_hidden)
        #self.gconv2 = GraphConv(gnn_hidden, gnn_hidden)

        # Agrego capa dropout para usarla junto con la conv
        self.dropout = torch.nn.Dropout(dropout)

        self.fc = nn.Linear(
            gnn_hidden,
            out_channels
        )

    def forward(self, x, edge_index, edge_weight=None):

        # x:
        # [num_nodes, time_window, features]
        #N, T, F = x.shape

        use_weights = self.use_weights

        if use_weights and edge_weight is None:
            raise ValueError("Model was configured with use_weights=True, but edge_weight=None was passed to the forward pass.")

        if not use_weights and edge_weight is not None:
            raise ValueError("Model was configured with use_weights=False, but edge_weight was provided to the forward pass.")


        x_t = x[:, :]  # [num_nodes, features]

        h_t = self.gconv1(
            x_t,
            edge_index=edge_index,
            edge_weight=edge_weight
        )  # [num_nodes, gnn_hidden]
        h_t = actF.relu(h_t)
        h_t = self.dropout(h_t) 
        
        #print(f"h stack: {h.shape}")

        # Take last timestep
        out = self.fc(h_t)  # Revisar aca, raro dimensiones

        return out.squeeze(-1)
    
class AttentionGNN(nn.Module):

    def __init__(self, num_features, gnn_hidden, out_channels, concat, use_weights, heads=4, dropout=0.0):
        super().__init__()

        self.use_weights = use_weights

        if use_weights:
            self.gat = GATv2Conv(
                in_channels=num_features,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout,
                edge_dim=1
            )
        else:
            self.gat = GATv2Conv(
                in_channels=num_features,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout
            )

        if concat:
            gat_dim = gnn_hidden * heads
        else:
            gat_dim = gnn_hidden

        self.fc = nn.Linear(
            gat_dim,
            out_channels,
        )

    def forward(self, x, edge_index, edge_weight=None):

        use_weights = self.use_weights

        if use_weights and edge_weight is None:
            raise ValueError("Model was configured with use_weights=True, but edge_weight=None was passed to the forward pass.")

        if not use_weights and edge_weight is not None:
            raise ValueError("Model was configured with use_weights=False, but edge_weight was provided to the forward pass.")

        # x: [num_nodes, time_window, features]
        N, T, Fdim = x.shape

        edge_index_big, edge_weight_big = expand_edge_index_for_time(
            edge_index, edge_weight, T, N, x.device
        )

        # [N, T, F] -> [T, N, F] -> [T*N, F]  (t-major to match edge index construction)
        x_flat = x.transpose(0, 1).reshape(T * N, Fdim)
        
        h_flat = self.gat(x_flat, edge_index_big, edge_attr=edge_weight_big)
        h_flat = actF.relu(h_flat)

        gat_dim = h_flat.shape[-1]
        h = h_flat.view(T, N, gat_dim).transpose(0, 1)

        # Temporal aggregation
        h = h.mean(dim=1)

        out = self.fc(h)

        return out.squeeze(-1)

class AttentionConvGNN(nn.Module):

    def __init__(self, num_features, gnn_hidden, out_channels, concat, use_weights, heads=4, dropout=0.0):
        super().__init__()

        self.use_weights = use_weights

        if use_weights:
            self.gat = GATv2Conv(
                in_channels=gnn_hidden,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout,
                edge_dim=1
            )
        else:
            self.gat = GATv2Conv(
                in_channels=gnn_hidden,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout
            )

        if concat:
            gat_dim = gnn_hidden * heads
        else:
            gat_dim = gnn_hidden

        self.gconv1 = GraphConv(
            num_features,
            gnn_hidden,
        )

        self.fc = nn.Linear(
            gat_dim,
            out_channels,
        )

    def forward(self, x, edge_index, edge_weight=None):

        use_weights = self.use_weights

        if use_weights and edge_weight is None:
            raise ValueError("Model was configured with use_weights=True, but edge_weight=None was passed to the forward pass.")

        if not use_weights and edge_weight is not None:
            raise ValueError("Model was configured with use_weights=False, but edge_weight was provided to the forward pass.")

        # x: [num_nodes, time_window, features]
        N, T, Fdim = x.shape

        edge_index_big, edge_weight_big = expand_edge_index_for_time(
            edge_index, edge_weight, T, N, x.device
        )

        # [N, T, F] -> [T, N, F] -> [T*N, F]  (t-major, matches edge index construction)
        x_flat = x.transpose(0, 1).reshape(T * N, Fdim)

        h_flat = self.gconv1(
            x_flat,
            edge_index_big,
            edge_weight_big,
        )
        h_flat = actF.relu(h_flat)

        # GATv2Conv
        h_flat = self.gat(
            h_flat,
            edge_index_big,
            edge_attr=edge_weight_big
        )
        h_flat = actF.relu(h_flat)

        gat_dim = h_flat.shape[-1]
        h = h_flat.view(T, N, gat_dim).transpose(0, 1)

        # Temporal aggregation
        h = h.mean(dim=1)

        out = self.fc(h)

        return out.squeeze(-1)

class TemporalAttentionGNN(nn.Module):

    def __init__(self, num_features, gnn_hidden, out_channels, concat, use_weights, heads=4, dropout=0.0):
        super().__init__()

        self.use_weights = use_weights

        if use_weights:
            self.gat = GATv2Conv(
                in_channels=num_features,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout,
                edge_dim=1
            )
        else:
            self.gat = GATv2Conv(
                in_channels=num_features,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout
            )

        if concat:
            gat_dim = gnn_hidden * heads # Con concat alcanzo 0.8 pero bajo, mas ruidoso
        else:
            gat_dim = gnn_hidden

        self.fc = nn.Linear(
            gnn_hidden,
            out_channels
        )

        self.temporal_conv = nn.Conv1d(
            in_channels=gat_dim,
            out_channels=gnn_hidden,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, edge_index, edge_weight=None):

        use_weights = self.use_weights

        if use_weights and edge_weight is None:
            raise ValueError("Model was configured with use_weights=True, but edge_weight=None was passed to the forward pass.")

        if not use_weights and edge_weight is not None:
            raise ValueError("Model was configured with use_weights=False, but edge_weight was provided to the forward pass.")

        # x: [num_nodes, time_window, features]
        N, T, Fdim = x.shape

        # Build one big block-diagonal graph covering all T timesteps at once,
        # instead of looping over t and calling self.gat T separate times.
        edge_index_big, edge_weight_big = expand_edge_index_for_time(
            edge_index, edge_weight, T, N, x.device
        )

        # [N, T, F] -> [T, N, F] -> [T*N, F]  (t-major, matches edge index construction)
        x_flat = x.transpose(0, 1).reshape(T * N, Fdim)

        h_flat = self.gat(
            x_flat,
            edge_index_big,
            edge_attr=edge_weight_big
        )
        h_flat = actF.relu(h_flat)

        gat_dim = h_flat.shape[-1]
        h = h_flat.view(T, N, gat_dim).transpose(0, 1)

        # [N, H, T]
        h = h.transpose(1, 2)

        # Temporal aggregation
        h = self.temporal_conv(h)
        h = actF.relu(h)
        h_last = h[:, :, -1] # Last timestep! (remember transpose of 'h')
        #h_last = h.mean(dim=-1)

        out = self.fc(h_last)

        return out.squeeze(-1)
    
class TemporalAttentionConvGNN(nn.Module):

    def __init__(self, num_features, gnn_hidden, out_channels, concat, use_weights, heads=4, dropout=0.0):
        super().__init__()

        self.use_weights = use_weights

        if use_weights:
            self.gat = GATv2Conv(
                in_channels=gnn_hidden,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout,
                edge_dim=1
            )
        else:
            self.gat = GATv2Conv(
                in_channels=gnn_hidden,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout
            )

        if concat:
            gat_dim = gnn_hidden * heads # Con concat alcanzo 0.8 pero bajo, mas ruidoso
        else:
            gat_dim = gnn_hidden


        self.gconv1 = GraphConv(
            num_features,
            gnn_hidden,
        )

        self.fc = nn.Linear(
            gnn_hidden,
            out_channels,
        )

        self.temporal_conv = nn.Conv1d(
            in_channels=gat_dim,
            out_channels=gnn_hidden,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, edge_index, edge_weight=None): # Por ahora edge_weight no hace nada... (CREO QUE ESTE COMENTARIO ES INCORRECTO Y ES VIEJO)

        use_weights = self.use_weights

        if use_weights and edge_weight is None:
            raise ValueError("Model was configured with use_weights=True, but edge_weight=None was passed to the forward pass.")

        if not use_weights and edge_weight is not None:
            raise ValueError("Model was configured with use_weights=False, but edge_weight was provided to the forward pass.")

        # x: [num_nodes, time_window, features]
        N, T, Fdim = x.shape

        edge_index_big, edge_weight_big = expand_edge_index_for_time(
            edge_index, edge_weight, T, N, x.device
        )

        # [N, T, F] -> [T, N, F] -> [T*N, F]  (t-major, matches edge index construction)
        x_flat = x.transpose(0, 1).reshape(T * N, Fdim)

        h_flat = self.gconv1(
            x_flat,
            edge_index_big,
            edge_weight_big,
        )
        h_flat = actF.relu(h_flat)

        # GATv2Conv
        h_flat = self.gat(
            h_flat,
            edge_index_big,
            edge_attr=edge_weight_big
        )
        h_flat = actF.relu(h_flat)

        gat_dim = h_flat.shape[-1]
        h = h_flat.view(T, N, gat_dim).transpose(0, 1)

        # [N, H, T]
        h = h.transpose(1, 2)

        # Temporal aggregation
        h = self.temporal_conv(h)
        h = actF.relu(h)
        h_last = h[:, :, -1] # Last timestep! (remember transpose of 'h')
        #h_last = h.mean(dim=-1)

        out = self.fc(h_last)

        return out.squeeze(-1)

class RecursiveAttentionGNN(nn.Module):

    def __init__(self, num_features, gnn_hidden, rnn_hidden, out_channels, concat, use_weights, heads=4, dropout=0.0):
        super().__init__()

        self.use_weights = use_weights

        if use_weights:
            self.gat = GATv2Conv(
                in_channels=num_features,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout,
                edge_dim=1
            )
        else:
            self.gat = GATv2Conv(
                in_channels=num_features,
                out_channels=gnn_hidden,
                heads=heads,
                concat=concat,  # output size = gnn_hidden
                dropout=dropout
            )


        if concat:
            gat_dim = gnn_hidden * heads # Con concat alcanzo 0.8 pero bajo, mas ruidoso
        else:
            gat_dim = gnn_hidden

        self.rnn = nn.GRU(
            input_size=gat_dim,
            hidden_size=rnn_hidden,
            batch_first=True,
        )

        self.fc = nn.Linear(
            rnn_hidden,
            out_channels,
        )

    def forward(self, x, edge_index, edge_weight=None):

        use_weights = self.use_weights

        if use_weights and edge_weight is None:
            raise ValueError("Model was configured with use_weights=True, but edge_weight=None was passed to the forward pass.")

        if not use_weights and edge_weight is not None:
            raise ValueError("Model was configured with use_weights=False, but edge_weight was provided to the forward pass.")

        # x: [num_nodes, time_window, features]
        N, T, Fdim = x.shape

        # Build one big block-diagonal graph covering all T timesteps at once,
        # instead of looping over t and calling self.gat T separate times.
        edge_index_big, edge_weight_big = expand_edge_index_for_time(
            edge_index, edge_weight, T, N, x.device
        )

        # [N, T, F] -> [T, N, F] -> [T*N, F]  (t-major, matches edge index construction)
        x_flat = x.transpose(0, 1).reshape(T * N, Fdim)

        h_flat = self.gat(
            x_flat,
            edge_index_big,
            edge_attr=edge_weight_big
        )
        h_flat = actF.relu(h_flat)

        # back to [T, N, gat_dim] -> [N, T, gat_dim]
        gat_dim = h_flat.shape[-1]
        h = h_flat.view(T, N, gat_dim).transpose(0, 1)

        # Temporal aggregation
        #h = h.mean(dim=1) Este no va para tener bien las dimensiones

        h, _ = self.rnn(h)

        # Take last timestep
        h_last = h[:, -1, :]

        # Prediction
        out = self.fc(h_last)

        return out.squeeze(-1)

class OnlyRecursiveGNN(nn.Module):

    def __init__(self, num_features, rnn_hidden, out_channels):
        super().__init__()

        self.num_features = num_features

        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=rnn_hidden,
            batch_first=True
        )

        self.fc = nn.Linear(
            rnn_hidden,
            out_channels
        )

    def forward(self, x, edge_index, edge_weight=None):

        # x:
        # [num_nodes, time_window, features]
        N, T, F = x.shape

        # RNN
        h, _ = self.rnn(x)

        # Take last timestep
        h_last = h[:, -1, :]

        # Prediction
        out = self.fc(h_last)

        return out.squeeze(-1)
        
    
# Recursive GNN using pytorch Geometric Temporal
# class GeometricRecursiveGNN(nn.Module):

#     def __init__(
#         self,
#         num_features,
#         gnn_hidden,
#         out_channels,
#         K=2,
#     ):
#         super().__init__()

#         self.recurrent = GConvGRU(
#             in_channels=num_features,
#             out_channels=gnn_hidden,
#             K=K,
#         )

#         self.fc = nn.Linear(
#             gnn_hidden,
#             out_channels
#         )

#     def forward(self, x, edge_index, edge_weight=None):

#         # x:
#         # [num_nodes, time_window, features]
#         N, T, F = x.shape

#         H = None

#         for t in range(T):

#             x_t = x[:, t, :]

#             H = self.recurrent(
#                 X=x_t,
#                 edge_index=edge_index,
#                 edge_weight=edge_weight,
#                 H=H,
#             )

#             H = actF.relu(H)

#         # H contains the final hidden state
#         out = self.fc(H)

#         return out.squeeze(-1)


# Recursive GNN using pytorch Geometric Temporal
# class GeometricRecursiveConvGNN(nn.Module):

#     def __init__(
#         self,
#         num_features,
#         gnn_hidden,
#         out_channels,
#         K=2,
#     ):
#         super().__init__()

#         self.recurrent = GConvGRU(
#             in_channels=num_features,
#             out_channels=gnn_hidden,
#             K=K,
#         )

#         self.fc = nn.Linear(
#             gnn_hidden,
#             out_channels
#         )

#         self.gconv1 = GraphConv(gnn_hidden, gnn_hidden)


#     def forward(self, x, edge_index, edge_weight=None):

#         # x:
#         # [num_nodes, time_window, features]
#         N, T, F = x.shape

#         H = None

#         for t in range(T):

#             x_t = x[:, t, :]

#             H = self.recurrent(
#                 X=x_t,
#                 edge_index=edge_index,
#                 edge_weight=edge_weight,
#                 H=H,
#             )

#             H = actF.relu(H)

#         H = self.gconv1(H, edge_index, edge_weight)
    
#         H = actF.relu(H)

#         # H contains the final hidden state
#         out = self.fc(H)

#         return out.squeeze(-1)

def build_model(model_name, params, model_config):

    model_definitions = {
        "conv": lambda: ConvGNN(
            num_features=model_config["num_features"],
            use_weights=model_config["use_weights"],
            gnn_hidden=params["num_nodes_layer"],
            dropout=params["dropout"],
            out_channels=1,
        ),
    
        "recursive": lambda: RecursiveGNN(
            num_features=model_config["num_features"],
            use_weights=model_config["use_weights"],
            gnn_hidden=params["num_nodes_layer"],
            rnn_hidden=params["num_nodes_layer"],
            dropout=params["dropout"],
            out_channels=1,
        ),

        "attention": lambda: AttentionGNN(
            num_features=model_config["num_features"],
            concat=model_config["concat"],
            use_weights=model_config["use_weights"],
            gnn_hidden=params["num_nodes_layer"],
            dropout=params["dropout"],
            out_channels=1,
        ), 

        "attentionConv": lambda: AttentionConvGNN(
            num_features=model_config["num_features"],
            concat=model_config["concat"],
            use_weights=model_config["use_weights"],
            gnn_hidden=params["num_nodes_layer"],
            dropout=params["dropout"],
            out_channels=1,
        ),

        "temporalAttention": lambda: TemporalAttentionGNN(
            num_features=model_config["num_features"],
            concat=model_config["concat"],
            use_weights=model_config["use_weights"],
            gnn_hidden=params["num_nodes_layer"],
            dropout=params["dropout"],
            out_channels=1,
        ),

        "temporalAttentionConv": lambda: TemporalAttentionConvGNN(
            num_features=model_config["num_features"],
            concat=model_config["concat"],
            use_weights=model_config["use_weights"],
            gnn_hidden=params["num_nodes_layer"],
            dropout=params["dropout"],
            out_channels=1,
        ),

        "recursiveAttention": lambda: RecursiveAttentionGNN(
            num_features=model_config["num_features"],
            concat=model_config["concat"],
            use_weights=model_config["use_weights"],
            gnn_hidden=params["num_nodes_layer"],
            rnn_hidden=params["num_nodes_layer"],
            dropout=params["dropout"],
            out_channels=1,
        ),

        "onlyRecursive": lambda: OnlyRecursiveGNN( # NO SOPORTA model_weights!
            num_features=model_config["num_features"],
            rnn_hidden=params["num_nodes_layer"],
            out_channels=1
        )
    }
    if model_name not in model_definitions:
        raise ValueError(
            f"Unknown model_name={model_name}."
            f"Available variants: {sorted(model_definitions)}"
        )
    
    model = model_definitions[model_name]()
    model = model.to(model_config["device"]) 

    return model

# Por ahora optimizo solo los parametros y dejo
def _base_space(trial):
    """Common hyperparameters shared across all model variants."""
    return {
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "num_nodes_layer": trial.suggest_int("num_nodes_layer", 8, 32, step=8),
    }


# NOTA: Si es necesario expandir los parametros segun el modelo se hace aca...
def get_hyperparams(trial, model_name):
    """Suggests hyperparameters for a given model name."""

    param_space = _base_space(trial)

    # Solamente 'onlyRecursive' no tiene dropout
    if model_name!="onlyRecursive":
        param_space["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

    return param_space

def train(model, criterion, optimizer, train_loader, use_weights, target_node_idx, num_graph_nodes, scalerY):
    model.train()
    total_train_loss = torch.zeros((), device=next(model.parameters()).device)
    all_preds = []
    all_targets = []

    for data in train_loader:
        optimizer.zero_grad()  # Clear gradients.

        # Si use_weights = True uso los pesos, sino no!
        if use_weights:
            out = model(data.x, data.edge_index, data.edge_weight)
        else:
            out = model(data.x, data.edge_index)

        # out: [num_graph_nodes x batch_size, num_features] ---> We want 'Caudal' for Node 'Florida'
        target_prediction = out[target_node_idx::num_graph_nodes]

        train_loss = criterion(target_prediction, data.y)
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.detach()
        all_preds.append(target_prediction.detach())
        all_targets.append(data.y.detach())

    all_preds = torch.cat(all_preds).reshape(-1, 1).cpu().numpy()
    all_targets = torch.cat(all_targets).reshape(-1, 1).cpu().numpy()

    all_preds_inv = scalerY.inverse_transform(all_preds)
    all_targets_inv = scalerY.inverse_transform(all_targets)
    all_preds_inv[all_preds_inv < 0] = 0

    nse = nse_metric(all_targets_inv, all_preds_inv)
    avg_loss = (total_train_loss / len(train_loader)).item()

    return avg_loss, nse


def test(model, criterion, test_loader, use_weights, target_node_idx, num_graph_nodes, scalerY):
    model.eval()
    total_test_loss = torch.zeros((), device=next(model.parameters()).device)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in test_loader:
            if use_weights:
                out = model(data.x, data.edge_index, data.edge_weight)
            else:
                out = model(data.x, data.edge_index)

            target_prediction = out[target_node_idx::num_graph_nodes]

            loss = criterion(target_prediction, data.y)
            total_test_loss += loss.detach()

            all_preds.append(target_prediction.detach())
            all_targets.append(data.y.detach())

    all_preds = torch.cat(all_preds).reshape(-1, 1).cpu().numpy()
    all_targets = torch.cat(all_targets).reshape(-1, 1).cpu().numpy()

    all_preds_inv = scalerY.inverse_transform(all_preds)
    all_targets_inv = scalerY.inverse_transform(all_targets)
    all_preds_inv[all_preds_inv < 0] = 0

    nse = nse_metric(all_targets_inv, all_preds_inv)
    avg_loss = (total_test_loss / len(test_loader)).item()

    return avg_loss, nse

def objective(trial, model_name, model_config):
    """
    model_config must contain:
        train_loader, test_loader
        use_weights, target_node_idx, num_nodes
        scalerY
        NUM_EPOCHS
        device (optional, used by build_model)
    Optional:
        default_lr (used if _base_space doesn't suggest "lr")
    """
    params = get_hyperparams(trial, model_name)
 
    model = build_model(model_name, params, model_config)

    optimizer_name = params["optimizer"]

    lr = params.get("lr", model_config.get("default_lr", 1e-3)) # Si no paso lr en optuna uso el defaul de model_config o 1e-3

    if optimizer_name=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name=="RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unexpected optimizer name (must be 'Adam' or 'RMSprop'), got '{optimizer_name}'")
 
    criterion = torch.nn.MSELoss()
 
    val_nse = -np.inf
 
    for epoch in range(NUM_EPOCHS):
        _, train_nse = train(
            model, criterion, optimizer,
            model_config["train_loader"], model_config["use_weights"],
            model_config["target_node_idx"], model_config["num_graph_nodes"], model_config["scalerY"],
        )
        val_loss, val_nse = test(
            model, criterion,
            model_config["val_loader"], model_config["use_weights"],
            model_config["target_node_idx"], model_config["num_graph_nodes"], model_config["scalerY"],
        )
 
        # Let Optuna prune unpromising trials early
        trial.report(val_nse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
 
    return val_nse


def run_study(model_name, config, n_trials=300, random_state=42):
    """
    Example:
        config = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "use_weights": True,
            "target_node_idx": target_node_idx,
            "num_nodes": num_nodes,
            "scalerY": scalerY,
            "NUM_EPOCHS": 500,
            "device": device,
        }
        study = run_study("RecursiveGNN", config, n_trials=250)
        print("Best trial:", study.best_trial.params)
    """
    sampler = TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    ) # TODO: Capaz conviene agregar un timeout por las dudas (por el tiempo)
    study.optimize(lambda trial: objective(trial, model_name, config), n_trials=n_trials)
    return study