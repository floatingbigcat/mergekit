import yaml
import subprocess
import random

def build_config() -> dict:

    ja_parameters = {
                    "pre_density": 0.6,
                    "pre_weight": 0.6,
                    "post_density": 0.6,
                    "post_weight": 0.6,
                }
    
    math1_parameters = {
                    "pre_density": 0.6,
                    "pre_weight": 0.6,
                    "post_density": 0.6,
                    "post_weight": 0.6,
                }
    
    math2_parameters = {
                    "pre_density": 0.6,
                    "pre_weight": 0.6,
                    "post_density": 0.6,
                    "post_weight": 0.6,
                }
    for i in range(32):  # 32 iterations for layers 0 through 31
        ja_parameters[f"layers_{i}_density"] = random.uniform(0, 1)
        ja_parameters[f"layers_{i}_weight"] = random.uniform(0, 1)
        math1_parameters[f"layers_{i}_density"] = random.uniform(0, 1)
        math1_parameters[f"layers_{i}_weight"] = random.uniform(0, 1)
        math2_parameters[f"layers_{i}_density"] = random.uniform(0, 1)
        math2_parameters[f"layers_{i}_weight"] = random.uniform(0, 1)
    
    return {
        "models": [
            {
                "model": "mistralai/Mistral-7B-v0.1",
            },
            {
                "model": "augmxnt/shisa-gamma-7b-v1",
                "parameters": ja_parameters,
            },
            {
                "model": "WizardLM/WizardMath-7B-V1.1",
                "parameters": math1_parameters
            },
            {
                "model": "GAIR/Abel-7B-002",
                "parameters": math2_parameters
            },
        ],
        "merge_method": "dare_ties",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "parameters": {"int8_mask": True},
        "dtype": "bfloat16",
    }


work_dir = "/home/qisun/work-optuna/math-ja-dare-ties/trial_debug/"
model_dir = "/home/qisun/work-optuna/math-ja-dare-ties"
mergekit_config_path = work_dir + "mergekit_config.yaml"

with open(mergekit_config_path, "w") as f:
    yaml.safe_dump(build_config(), f)

subprocess.run(
    [
        "mergekit-yaml",
        "--cuda",
        "--random-seed",
        "42",
        str(mergekit_config_path),
        str(
            model_dir,
        ),
    ],
    check=True,
)