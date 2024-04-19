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

import click
import yaml
import torch

from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, add_merge_options


def run(config_source: str,
        weight_dict: dict[str, dict[str, torch.Tensor]],
        cuda: bool, 
        random_seed: int, 
        verbose: bool):
    merge_options = MergeOptions(
        cuda=cuda,
        random_seed=random_seed,
    )

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    merge_config: MergeConfiguration = MergeConfiguration.model_validate(
        yaml.safe_load(config_source)
    )

    update_weight = run_merge(
        merge_config,
        options=merge_options,
        weight_dict=weight_dict,
    )

    return update_weight
