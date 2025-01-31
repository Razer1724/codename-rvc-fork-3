import os
import sys
import importlib.util
from typing import Type, Dict, Any, Union
from torch.optim import Optimizer
import torch.optim as optim

def load_optimizer(optimizer_name: str, parameters, **kwargs) -> Optimizer:
    """
    Dynamically loads and instantiates an optimizer either from torch.optim or from custom_optimizers folder.
    """
    # First check if it's a PyTorch built-in optimizer
    if hasattr(optim, optimizer_name):
        return getattr(optim, optimizer_name)(parameters, **kwargs)
    
    # Get the absolute path to the script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths with proper case handling
    optimizer_folder = os.path.join(current_dir, "custom_optimizers", optimizer_name)
    possible_paths = [
        # Try original casing
        os.path.join(optimizer_folder, f"{optimizer_name.lower()}.py"),
        # Try all lowercase
        os.path.join(optimizer_folder.lower(), f"{optimizer_name.lower()}.py"),
        # Try with original optimizer name casing
        os.path.join(optimizer_folder, f"{optimizer_name}.py"),
    ]
    
    # Try each possible path
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if found_path:
        print(f"Found optimizer at: {found_path}")
        
        # Load the custom optimizer module
        spec = importlib.util.spec_from_file_location(
            optimizer_name.lower(), found_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load optimizer spec from {found_path}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[optimizer_name.lower()] = module
        spec.loader.exec_module(module)
        
        # Get the optimizer class from the module
        optimizer_class = getattr(module, optimizer_name)
        return optimizer_class(parameters, **kwargs)
    
    # If no path was found, print debug information
    print(f"Debug: Current directory is {current_dir}")
    print(f"Debug: Tried the following paths:")
    for path in possible_paths:
        print(f"  - {path} (exists: {os.path.exists(path)})")
    
    raise ValueError(
        f"Optimizer '{optimizer_name}' not found in torch.optim or custom_optimizers folder"
    )




def create_optimizer_config(name: str) -> Dict[str, Any]:
    """
    Creates optimizer configuration based on optimizer name.
    Add new optimizer configs here.
    
    Args:
        name (str): Name of the optimizer
    
    Returns:
        Dict[str, Any]: Configuration dictionary for the optimizer
    """
    configs = {
        "AdaBelief": {
            "lr": 1e-4,
            "betas": (0.8, 0.99),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "rectify": True,
            "degenerated_to_sgd": True
        },
        "SophiaG": {
            "lr": 1e-4,
            "betas": (0.8, 0.99),
            "rho": 0.04,
            "weight_decay": 1e-2,
            "eps": 1e-8
        },
        "RAdam":{
            "lr": 1e-4, 
            "betas": (0.8, 0.99),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "decoupled_weight_decay": True
        },
        # Add more optimizer configurations here
    }
    
    return configs.get(name, {})





