import yaml
import subprocess


def build_config() -> dict:
    return {
        "models": [
            {
                "model": "mistralai/Mistral-7B-v0.1",
            },
            {
                "model": "augmxnt/shisa-gamma-7b-v1",
                "parameters": {
                    "pre_density": 0.6,
                    "pre_weight": 0.6,
                    "layers_density": 0.5,
                    "layers_weight": 0.5,
                    "post_density": 0.6,
                    "post_weight": 0.6,
                },
            },
            {
                "model": "WizardLM/WizardMath-7B-V1.1",
                "parameters": {
                    "pre_density": 0.6,
                    "pre_weight": 0.6,
                    "layers_density": 0.5,
                    "layers_weight": 0.5,
                    "post_density": 0.6,
                    "post_weight": 0.6,
                },
            },
            {
                "model": "GAIR/Abel-7B-002",
                "parameters": {
                    "pre_density": 0.6,
                    "pre_weight": 0.6,
                    "layers_density": 0.5,
                    "layers_weight": 0.5,
                    "post_density": 0.6,
                    "post_weight": 0.6,
                },
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