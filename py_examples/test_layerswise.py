import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import yaml
from vllm import LLM
from mergekit.scripts.run_yaml import run
import torch

def set_weight_dict(models):
    weight_dict = {}
    for m in models:
        model_path = m['model']
        model = LLM(model=model_path, dtype="bfloat16")
        # BUG once init the vllm, the cuda environment is fixed
        model_layer = model.llm_engine.driver_worker.model_runner.model
        layer_dict = {}
        for name, param in model_layer.named_parameters():
            layer_dict[name] = param.data.clone().to("cpu")
        del model
        weight_dict[model_path] = layer_dict
    return weight_dict

def rewrite_parameters(update_weight, init_llm):
    init_model = init_llm.llm_engine.driver_worker.model_runner.model

    for name, param in init_model.named_parameters():
        param.data = update_weight[name]

    return init_llm

def baseline_params2config(params):
    ja_params, math_params = params[0:68], params[68:136]
    assert len(ja_params) == len(math_params)

    ja_parameters = {
                    "density": ja_params[0],
                    "weight": ja_params[1],
                }
    
    math_parameters = {
                    "density": math_params[0],
                    "weight": math_params[1],
                }
    
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
                "model": "GAIR/Abel-7B-002",
                "parameters": math_parameters
            },
        ],
        "merge_method": "dare_ties",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "parameters": {"int8_mask": True},
        "dtype": "bfloat16",
    }

def layerwise_params2config(params):
    ja_params, math_params = params[0:68], params[68:136]
    assert len(ja_params) == len(math_params)

    ja_parameters = {
                    "pre_density": ja_params[0],
                    "pre_weight": ja_params[1],
                    "post_density": ja_params[2],
                    "post_weight": ja_params[3],
                }
    
    math_parameters = {
                    "pre_density": math_params[0],
                    "pre_weight": math_params[1],
                    "post_density": math_params[2],
                    "post_weight": math_params[3],
                }

    
    for i in range(32):  # 32 iterations for layers 0 through 31
        ja_parameters[f"layers_{i}_density"] = ja_params[4+i]
        ja_parameters[f"layers_{i}_weight"] = ja_params[4+i+1]
        math_parameters[f"layers_{i}_density"] =  math_params[4+i] 
        math_parameters[f"layers_{i}_weight"] =  math_params[4+i+1] 

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
                "model": "GAIR/Abel-7B-002",
                "parameters": math_parameters
            },
        ],
        "merge_method": "dare_ties",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "parameters": {"int8_mask": True},
        "dtype": "bfloat16",
    }


def test_layerwise_merge():
    params = [0.1] * 136
    baseline_config = baseline_params2config(params)    
    layerwise_config = layerwise_params2config(params)    
    weight_dicts = set_weight_dict(baseline_config['models'])
    
    update_weight1 = run(config_source = yaml.safe_dump(baseline_config), weight_dict=weight_dicts, cuda = True, random_seed = 42, verbose = False)
    update_weight2 = run(config_source = yaml.safe_dump(layerwise_config), weight_dict=weight_dicts, cuda = True, random_seed = 42, verbose = False)
    
    for k in update_weight1.keys():
        assert torch.equal(update_weight1[k], update_weight2[k])

    print("Pass test!")

if __name__ == "__main__":
    test_layerwise_merge()