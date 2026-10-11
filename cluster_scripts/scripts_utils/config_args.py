import argparse
import os
import re

def valid_sqlite_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Invalid path: '{path}' does not exist or is not a file.")
    if not path.endswith('.sqlite'):
        raise argparse.ArgumentTypeError(f"Invalid file: '{path}' does not have a '.sqlite' extension.")
    return path

def parse_graph_variant_pairs(value):
    """
    Parses a string like:
        "Grafo 6 v2, Grafo 5, Grafo 12 v2, Grafo 2 v2, Grafo 10, Grafo 12, Grafo 2"
    into a list of (graph_number, variant) tuples:
        [(6, 'v2'), (5, 'v1'), (12, 'v2'), (2, 'v2'), (10, 'v1'), (12, 'v1'), (2, 'v1')]

    If no version is given after the graph number, it defaults to 'v1'
    (use_weights=False, concat=False).
    """
    pairs = []
    entries = [e.strip() for e in value.split(',') if e.strip()]
    pattern = re.compile(r'^Grafo\s*(\d+)\s*(v[1-4])?$', re.IGNORECASE)

    for entry in entries:
        match = pattern.match(entry)
        if not match:
            raise argparse.ArgumentTypeError(f"Invalid graph-variant entry: '{entry}'. Expected format like 'Grafo 6 v2' or 'Grafo 5' (defaults to v1).")
        graph_number = int(match.group(1))
        variant = match.group(2).lower() if match.group(2) else 'v1'
        pairs.append((graph_number, variant))

    return pairs

def get_gnn_training_args(parser):

    # Tipo de modelo empleado
    parser.add_argument('--model', '-m', required=True, help="<Required> Select model type: 'conv', 'recursive' or 'attention'")

    # Modo seleccionado
    parser.add_argument('--experiment-mode', '-em', default="auto", help="<Required> Experiment mode: 'auto' (evaluate all available versions) or 'selective' (evaluate only the specified version(s)).")

    # Selected variant(s) (only used when --experiment-mode=selected)
    parser.add_argument('--graph-variant-pairs', '-gvp', type=parse_graph_variant_pairs,
                         help="<Optional> Specific graph+variant combinations to run in 'selective' mode, e.g. "
                              "\"Grafo 6 v2, Grafo 5, Grafo 12 v2\". "
                              "No version after the number defaults to v1. When provided, this overrides "
                              "--run-all-graphs and --run-graph-number.")
    
    parser.add_argument('--weight-type', '-wt', required=True, help="<Required> Select weight type: 'dist' or 'idw'")

    # Ubicación de directorios
    parser.add_argument('--results-dir', default='./data/output_results', help="Directory to store results. Default is './data/output_results'.")
    parser.add_argument('--data-dir', default='./data/input_data', help="Directory to read input data from. Default is './data/input_data'.")

    parser.add_argument('--run-all-graphs', '-rag', action='store_true', help="Run all graph variants for the given model.")

    parser.add_argument('--run-graph-number', '-rgn', nargs='+', type=int, help="Run selected graph variants.")

    # Fecha de inicio y fin
    parser.add_argument('--start-date', default='2011-1-1', help="Start date for the data range in YYYY-MM-DD format. Default is '2011-1-1'.")
    parser.add_argument('--end-date', default='2022-03-28', help="End date for the data range in YYYY-MM-DD format. Default is '2022-03-28'.")

    return parser
