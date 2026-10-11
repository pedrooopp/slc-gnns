import re
import logging

def extract_station(col):
    return re.match(r"\[(.*?)\]", col).group(1)

def has_self_loop(edge_index, target_node_idx):
    return ((edge_index[0] == target_node_idx) &
            (edge_index[1] == target_node_idx)).any().item()

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name :str, level=logging.INFO):
    "Prepara el nivel de log y lo formatea"

    # Se define nivel de log y se le cambia el formato
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

        fh = logging.FileHandler(f"{name}.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_variant_config(variant):
    """
    Return the configuration associated with a model variant.

    Parameters:
    variant : str
        One of: 'v1', 'v2', 'v3', 'v4'.

    Returns:
    tuple[bool, bool]
        (use_weights, concat)
    """
    configs = {
        "v1": (False, False),
        "v2": (True,  False),
        "v3": (False, True),
        "v4": (True,  True),
    }

    try:
        return configs[variant]
    except KeyError:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of {list(configs.keys())}.")
    
def get_model_variant(use_weights, concat):
    """
    Return the model variant associated with a configuration.

    Parameters:
    use_weights : bool
        Whether edge weights are used.
    concat : bool
        Whether concatenation is enabled.

    Returns:
    str
        One of: 'v1', 'v2', 'v3', 'v4'.
    """
    variants = {
        (False, False): "v1",
        (True,  False): "v2",
        (False, True):  "v3",
        (True,  True):  "v4",
    }

    try:
        return variants[(use_weights, concat)]
    except KeyError:
        raise ValueError(
            f"Invalid configuration: use_weights={use_weights}, concat={concat}."
        )