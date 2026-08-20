import numpy as np
import torch
import os # Import os for path joining
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ReallyLazyGraphDataset_v2(Dataset):
    """
    Releases CPU memory intensity by memory-mapping input data.
    Handles a variable number of input tensor arrays + static input tensors,
    and a separate set of reconstructed fields.
    Input files are inferred from variable names and a base data directory.
    """
    def __init__(self, indices, 
                 data_dir,                         # New: Base directory for data files
                 time_varying_variables,           # Changed: list of variable names
                 static_variables,                 # Changed: list of variable names
                 reconstructed_variables):
        
        self.indices = indices
        self.data_dir = data_dir # Store data directory
        
        # --- Construct file paths from variable names ---
        time_varying_files = [os.path.join(data_dir, f"{var}.npy") for var in time_varying_variables]
        reconstructed_files = [os.path.join(data_dir, f"{var}.npy") for var in reconstructed_variables]
        static_files = [os.path.join(data_dir, f"{var}.npy") for var in static_variables]

        # Memory-map time-varying input tensor arrays
        if not time_varying_files:
            raise ValueError("No time-varying input variables provided.")
        # Check if files exist for time_varying
        for f in time_varying_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Time-varying file not found: {f}")
        self.time_varying_memmaps = [np.load(path, mmap_mode='r') for path in time_varying_files]
        print(f"Loaded {len(self.time_varying_memmaps)} time-varying input tensor arrays.")

        # Memory-map reconstructed tensor arrays
        if not reconstructed_files:
            raise ValueError("No reconstructed variables provided.")
        # Check if files exist for reconstructed
        for f in reconstructed_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Reconstructed file not found: {f}")
        self.reconstructed_memmaps = [np.load(path, mmap_mode='r') for path in reconstructed_files]
        print(f"Loaded {len(self.reconstructed_memmaps)} reconstructed tensor arrays.")

        # Load static tensor arrays fully into memory (eager loading once)
        if not static_files:
            raise ValueError("No static variables provided.")
        # Check if files exist for static
        for f in static_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Static file not found: {f}")
        self.static_tensors = [torch.tensor(np.load(path), dtype=torch.float32) for path in static_files]
        print(f"Loaded {len(self.static_tensors)} static input tensor arrays (fully into RAM).")
        
 
        # Store the number of features for convenience
        self.num_time_varying_inputs = len(self.time_varying_memmaps)
        self.num_static_inputs = len(self.static_tensors)
        self.num_reconstructed_inputs = len(self.reconstructed_memmaps)

        
        # Total input features: (num of time-varying inputs) + (num of reconstructed inputs) + (num of static inputs)
        self.num_input_features = self.num_time_varying_inputs + self.num_static_inputs
        
        # The number of output features is determined by the reconstructed inputs
        self.num_output_features = self.num_reconstructed_inputs
        print(self.num_output_features)
       
        # Ensure number of nodes matches between time-varying, reconstructed, and static tensors
        if self.time_varying_memmaps:
            num_nodes_time_varying = self.time_varying_memmaps[0].shape[1]
            # Check static tensors
            for i, st in enumerate(self.static_tensors):
                if st.shape[0] != num_nodes_time_varying:
                    raise ValueError(f"Number of nodes in static input tensor '{static_variables[i]}' "
                                     f"({st.shape[0]}) does not match "
                                     f"time-varying input arrays ({num_nodes_time_varying}).")
            # Check reconstructed tensors
            if self.reconstructed_memmaps: # Only check if there are reconstructed inputs
                num_nodes_reconstructed = self.reconstructed_memmaps[0].shape[1]
                if num_nodes_reconstructed != num_nodes_time_varying:
                    raise ValueError(f"Number of nodes in reconstructed input arrays "
                                     f"({num_nodes_reconstructed}) does not match "
                                     f"time-varying input arrays ({num_nodes_time_varying}).")
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx] # 't' represents the index for a specific sample/time step

        # --- Time-varying Input Data ---
        time_varying_data_at_t = [torch.tensor(memmap[t], dtype=torch.float32) 
                                  for memmap in self.time_varying_memmaps]
        
        # --- Reconstructed Input Data ---
        reconstructed_data_at_t = [torch.tensor(memmap[t], dtype=torch.float32) 
                                   for memmap in self.reconstructed_memmaps]
        
        # --- Static Input Data ---
        static_data = self.static_tensors # Already loaded as PyTorch tensors

        # --- Construct Input Features (x) ---
        # x consists of time-varying inputs + reconstructed inputs + static components
        x_components = time_varying_data_at_t + static_data
        x = torch.stack(x_components, dim=1)
        
        # --- Construct Output Labels (y) ---
        # y consists ONLY of the reconstructed inputs (target for reconstruction or next-step prediction)
        y = torch.stack(reconstructed_data_at_t, dim=1)
        
        return Data(x=x, y=y)

