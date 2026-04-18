import yaml
from pathlib import Path
from src.frog_perch.nn_training.train import train
import configs.nn_config as config

def load_normalization_stats(yaml_path: str) -> dict:
    """Safely load the generated normalization stats."""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find {yaml_path}. Have you run the normalization script yet?"
        )

if __name__ == '__main__':
    # 1. Grab your static config
    experiment_config = {
        key: value for key, value in vars(config).items() 
        if key.isupper()
    }
    
    # 2. Load your dynamic stats
    # (Assuming normalization.yaml is in the same folder as config.py)
    config_dir = Path(config.__file__).parent
    norm_stats = load_normalization_stats(config_dir / "normalization.yaml")
    
    # 3. Assemble the final CONFIDENCE_PARAMS dictionary
    experiment_config["CONFIDENCE_PARAMS"] = {
        "duration_stats": norm_stats.get("duration", {}),
        "bandwidth_stats": norm_stats.get("bandwidth", {}),
        "logistic_params": experiment_config.get("CONFIDENCE_LOGISTIC_PARAMS", {})
    }
    
    # 4. Train!
    model, val_ds = train(experiment_config)