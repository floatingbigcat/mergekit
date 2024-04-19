# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import os
from typing import Optional

import tqdm
import transformers
import torch

from mergekit.architecture import ArchitectureInfo, get_architecture_info
from mergekit.card import generate_card
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner
from mergekit.tokenizer import TokenizerInfo


def run_merge(
    merge_config: MergeConfiguration,
    options: MergeOptions,
    weight_dict: dict[str, dict[str, torch.Tensor]],
):
    if options.random_seed is not None:
        transformers.trainer_utils.set_seed(options.random_seed)

    if not merge_config.models and not merge_config.slices:
        raise RuntimeError("No output requested")

    model_arch_info = [
        get_architecture_info(m.config(trust_remote_code=options.trust_remote_code))
        for m in merge_config.referenced_models()
    ]
    if not options.allow_crimes:
        if not all(a == model_arch_info[0] for a in model_arch_info[1:]):
            raise RuntimeError(
                "Must specify --allow-crimes to attempt to mix different architectures"
            )
    arch_info = model_arch_info[0]

    # create config for output model
    cfg_out = _model_out_config(
        merge_config, arch_info, trust_remote_code=options.trust_remote_code
    )

    logging.info("Planning operations")
    targets = MergePlanner(
        merge_config,
        arch_info,
        options=options,
        out_model_config=cfg_out,
    ).plan()

    exec = Executor(
        tasks=targets,
        math_device="cuda" if options.cuda else "cpu",
        storage_device="cuda" if options.low_cpu_memory else "cpu"
    )

    update_weight = exec.run(weight_dict)
    return update_weight


def _get_donor_tokenizer(
    merge_config: MergeConfiguration, trust_remote_code: bool = False
):
    try:
        donor_model = merge_config.base_model
        if not donor_model:
            donor_model = merge_config.referenced_models()[0]

        return transformers.AutoTokenizer.from_pretrained(
            donor_model.model.path,
            revision=donor_model.model.revision,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        logging.error(
            "Failed to copy tokenizer. The merge was still successful, just copy it from somewhere else.",
            exc_info=e,
        )
        return None


def _model_out_config(
    config: MergeConfiguration,
    arch_info: ArchitectureInfo,
    trust_remote_code: bool = False,
) -> transformers.PretrainedConfig:
    """Return a configuration for the resulting model."""
    if config.base_model:
        res = config.base_model.config(trust_remote_code=trust_remote_code)
    else:
        res = config.referenced_models()[0].config(trust_remote_code=trust_remote_code)
    if config.dtype:
        res.torch_dtype = config.dtype

    if config.slices:
        try:
            num_layers = sum(
                s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                for s in config.slices
            )
            setattr(res, arch_info.num_layers_config_key(), num_layers)
        except Exception as e:
            logging.warning(
                "Unable to set number of layers in output config - you may need to manually correct it.",
                exc_info=e,
            )

    return res


def _update_config_vocab(
    config: transformers.PretrainedConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
):
    try:
        config.vocab_size = len(tokenizer.get_vocab())
    except Exception as e:
        logging.warning(
            "Unable to set vocabulary size in output config - you may need to manually correct it.",
            exc_info=e,
        )


__all__ = ["MergeOptions", "run_merge"]
