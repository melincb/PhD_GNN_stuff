import torch

def clean_state_dict(state_dict, prefixes=None):
    """
    Remove unwanted prefixes (like 'module.' or '_orig_mod.') from keys
    in a PyTorch state_dict so it matches the current model definition.

    Args:
        state_dict (dict): The original state_dict from checkpoint
        prefixes (list[str], optional): List of prefixes to strip. 
                                        Defaults to common wrappers.

    Returns:
        dict: A new state_dict with cleaned keys.
    """
    if prefixes is None:
        prefixes = ["_orig_mod."]  #["module.", "_orig_mod.", "_fsdp_wrapped_module."]

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for p in prefixes:
            if new_key.startswith(p):
                new_key = new_key[len(p):]
        new_state_dict[new_key] = v

    return new_state_dict


def coalesce_tensor_list(tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Coalesces each sparse tensor in a list.

    Args:
        tensor_list (list[torch.Tensor]): A list of sparse PyTorch tensors.

    Returns:
        list[torch.Tensor]: A new list containing the coalesced tensors.
    """
    coalesced_list = [t.coalesce() for t in tensor_list]
    return coalesced_list


def clean_level_connections(level_connections):
    level_connections_clean = []
    for level_dict in level_connections:
        clean_dict = {int(k): [int(v) for v in vals] for k, vals in level_dict.items()}
        level_connections_clean.append(clean_dict)
    return level_connections_clean


def preprocess_level_connections(level_connections):
    """Convert level_connections to a more efficient format."""
    processed = []
    for connections in level_connections:
        # Sort once and create efficient lookup structures
        coarse_indices = sorted(connections.keys())
        sorted_connections = [(coarse_idx, connections[coarse_idx]) for coarse_idx in coarse_indices]
        processed.append(sorted_connections)
    return processed

# coarse_indices = sorted(fine_to_coarse_connections.keys())
#         coarse_idx_to_output_idx = {coarse_idx: i for i, coarse_idx in enumerate(coarse_indices)}