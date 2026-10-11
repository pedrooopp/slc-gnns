import re
from typing import List, Union, Optional

def replace_string_in_keys(input_dict, old_string, new_string):
    """
    Replaces all occurrences of a string in dictionary keys with another string.

    Args:
        input_dict (dict): The dictionary to modify.
        old_string (str): The string to replace.
        new_string (str): The string to replace with.

    Returns:
        dict: A new dictionary with the replaced keys.
    """
    return {key.replace(old_string, new_string): value for key, value in input_dict.items()}

def find_matching_name(full_names: List[str], clean_name: Union[str, List[str]]) -> Union[Optional[str], List[Optional[str]]]:
    """
    Finds the full name(s) from a list of names that match a given base name or list of base names.

    Parameters:
        full_names (List[str]): A list of full names, potentially including parenthetical descriptors.
        clean_name (Union[str, List[str]]): A single base name or a list of base names to match.

    Returns:
        Union[Optional[str], List[Optional[str]]]: The matched full name(s) from the list.
            - If `clean_name` is a string, returns a single matching full name or None if not found.
            - If `clean_name` is a list, returns a list of matches, preserving the order. Unmatched names return None.

    Example:
        full_names = ['Sensor A (calibrated)', 'Sensor B (raw)', 'Sensor C']
        find_matching_name(full_names, 'Sensor A')
        # Returns: 'Sensor A (calibrated)'

        find_matching_name(full_names, ['Sensor A', 'Sensor C'])
        # Returns: ['Sensor A (calibrated)', 'Sensor C']
    """
    def match_name(target):
        for name in full_names:
            base_name = re.sub(r'\s*\([^)]*\)', '', name)
            if base_name == target:
                return name
        return None

    if isinstance(clean_name, list):
        return [match_name(name) for name in clean_name]
    else:
        return match_name(clean_name)

def make_model_name(algorithm, metric, variables=None):
    return f"{algorithm} [{metric}] {tuple(sorted(variables))}"

def get_sklearn_metric_name(original_str):
    return original_str.replace('MSE', "squared_error").replace('MAE', "absolute_error")

def get_algorithm_full_name(original_str):
    return original_str.replace("RF", "Random Forest Regressor").replace("ET", "Extra Trees Regressor").replace("HGB", "Histogram-based Gradient Boosting")
