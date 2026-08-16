from typing import Optional

import torch
from torch.utils.data import random_split
from torchvision.transforms.v2 import Transform, ToDtype, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from ..pbdl_dataloader.dataset import Dataset as PBDLDataset

from .variable_dt_dataset import VariableDtDataset
from .ape_2d_splits import (
    SEPARATE_TEST_DATASETS,
    ape_2d_xxl_simulation_split,
)


'''
    Extended 2d datasets generated with exponax solver from Köhler et al.: APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs.
'''
seed = 46

def ape_2d_xxl_datasets(dataset_name: str,
                 dataset_directory: str,
                 unrolling_steps: int,
                 intermediate_time_steps: bool = True,
                 variable_dt_stride_maximum: int = 1,
                 test_variable_dt_stride_maximum: int = 1,
                 test_unrolling_steps: Optional[int] = None,
                 test_intermediate_time_steps: Optional[bool] = None,
                 test_sim_ids_override: Optional[list[int]] = None,
                 normalize_data: Optional[str] = None,
                 normalize_const: Optional[str] = None,
                 dataset_profile: str = "legacy_small",
                 **kwargs) -> tuple[PBDLDataset, PBDLDataset, PBDLDataset]:
    r'''
    Creates 2D extended APE train, val, and test dataset objects.

    Args:
        dataset_name: name of local dataset file
        dataset_directory: directory where the data set is located
        unrolling_steps: number of time steps between start and end of sequence
        intermediate_time_steps: determines if intermediate time steps are included in the data or not
        variable_dt_stride_maximum: maximum number of time steps between start and end of sequence for variable dt training
        test_variable_dt_stride_maximum: maximum number of time steps between start and end of sequence for variable dt testing
        test_unrolling_steps: number of time steps between start and end of sequence for the test dataset
        test_intermediate_time_steps: determines if intermediate time steps are included in the data
                                        or not for the test dataset
        test_sim_ids_override: explicit simulation IDs used for test-only datasets
        normalize_data: type of normalization to apply to the data (mean-std, std, zero-to-one, minus-one-to-one, None)
        normalize_const: type of normalization to apply to the constants (mean-std, std, zero-to-one, minus-one-to-one, None)

    Returns:
        tuple[PBDLDataset, PBDLDataset, PBDLDataset]: train, validation and test datasets
    '''

    if test_unrolling_steps is None:
        test_unrolling_steps = unrolling_steps
    if test_intermediate_time_steps is None:
        test_intermediate_time_steps = intermediate_time_steps
    if test_variable_dt_stride_maximum is None:
        test_variable_dt_stride_maximum = variable_dt_stride_maximum

    if test_sim_ids_override is not None:
        test_sim_ids = [int(sim_id) for sim_id in test_sim_ids_override]
        if not test_sim_ids or len(test_sim_ids) != len(set(test_sim_ids)):
            raise ValueError("test_sim_ids_override must contain unique simulation IDs.")

        normalize_const_for_dataset = normalize_const if "gs_" not in dataset_name else None
        params_train = {
            "dset_name": dataset_name,
            "local_datasets_dir": dataset_directory,
            "sel_sims": test_sim_ids,
            "time_steps": unrolling_steps,
            "intermediate_time_steps": intermediate_time_steps,
            "normalize_const": normalize_const_for_dataset,
            "normalize_data": normalize_data,
        }
        if variable_dt_stride_maximum <= 1:
            pbdl_all = PBDLDataset(**params_train)
        else:
            pbdl_all = VariableDtDataset(
                **params_train,
                maximum_dt=variable_dt_stride_maximum,
                seed=None,
            )

        train, val = random_split(
            pbdl_all, [0.85, 0.15], generator=torch.Generator().manual_seed(seed)
        )
        train.indices = sorted(train.indices)
        val.indices = sorted(val.indices)

        params_test = {
            "dset_name": dataset_name,
            "local_datasets_dir": dataset_directory,
            "sel_sims": test_sim_ids,
            "time_steps": test_unrolling_steps,
            "intermediate_time_steps": test_intermediate_time_steps,
            "normalize_const": normalize_const_for_dataset,
            "normalize_data": normalize_data,
        }
        if test_variable_dt_stride_maximum <= 1:
            test = PBDLDataset(**params_test)
        else:
            test = VariableDtDataset(
                **params_test,
                maximum_dt=test_variable_dt_stride_maximum,
                seed=seed,
            )
        return train, val, test

    # separate test sets with longer rollouts
    if dataset_name in SEPARATE_TEST_DATASETS:
        params_train = {
            "dset_name": dataset_name,
            "local_datasets_dir": dataset_directory,
            "time_steps": unrolling_steps,
            "intermediate_time_steps": intermediate_time_steps,
            "normalize_const": normalize_const if not "gs_" in dataset_name else None,
            "normalize_data": normalize_data
        }

        if variable_dt_stride_maximum <= 1:
            pbdl_all = PBDLDataset(**params_train)
        else:
            pbdl_all = VariableDtDataset(**params_train, maximum_dt=variable_dt_stride_maximum, seed=None)

        train, val = random_split(
            pbdl_all, [0.85, 0.15], generator=torch.Generator().manual_seed(seed)
        )
        # the data indices have to be sorted manually since random_split shuffles them
        train.indices = sorted(train.indices)
        val.indices = sorted(val.indices)

        if dataset_name in ["gs_alpha", "gs_beta", "gs_gamma", "gs_epsilon"]:
            trim_end = 100 - test_unrolling_steps - 1
        else:
            trim_end = 200 - test_unrolling_steps - 1

        params_test = {
            "dset_name": dataset_name + "_test",
            "local_datasets_dir": dataset_directory,
            "time_steps": test_unrolling_steps,
            "trim_end": trim_end,
            "intermediate_time_steps": test_intermediate_time_steps,
            "normalize_const": normalize_const if not "gs_" in dataset_name else None,
            "normalize_data": normalize_data
        }
        if test_variable_dt_stride_maximum <= 1:
            test = PBDLDataset(**params_test)
        else:
            test = VariableDtDataset(**params_test, maximum_dt=test_variable_dt_stride_maximum, seed=seed)




    # joint training and test data file, split by simulation
    else:
        
        train_sims, test_sims = ape_2d_xxl_simulation_split(
            dataset_name,
            dataset_profile,
        )
        if train_sims is None or test_sims is None:
            raise AssertionError("Joint-file dataset must define explicit simulation IDs.")

        params_train = {
            "dset_name": dataset_name,
            "local_datasets_dir": dataset_directory,
            "sel_sims": train_sims,
            "time_steps": unrolling_steps,
            "intermediate_time_steps": intermediate_time_steps,
            "normalize_const": normalize_const if not "gs_" in dataset_name else None,
            "normalize_data": normalize_data
        }

        if variable_dt_stride_maximum <= 1:
            pbdl_all = PBDLDataset(**params_train)
        else:
            pbdl_all = VariableDtDataset(**params_train, maximum_dt=variable_dt_stride_maximum, seed=None)

        train, val = random_split(
            pbdl_all, [0.85, 0.15], generator=torch.Generator().manual_seed(seed)
        )
        # the data indices have to be sorted manually since random_split shuffles them
        train.indices = sorted(train.indices)
        val.indices = sorted(val.indices)

        params_test = {
            "dset_name": dataset_name,
            "local_datasets_dir": dataset_directory,
            "sel_sims": test_sims,
            "time_steps": test_unrolling_steps,
            "intermediate_time_steps": test_intermediate_time_steps,
            "normalize_const": normalize_const if not "gs_" in dataset_name else None,
            "normalize_data": normalize_data
        }

        if test_variable_dt_stride_maximum <= 1:
            test = PBDLDataset(**params_test)
        else:
            test = VariableDtDataset(**params_test, maximum_dt=test_variable_dt_stride_maximum, seed=seed)

    return train, val, test



def ape_2d_xxl_transforms(dataset_name: str)-> tuple[Transform, Transform, Transform]:
    r'''
    Creates 2d extended APE train, val, and test transform objects.

    Args:
        dataset_name: name of local dataset

    Returns:
        tuple[Transform, Transform, Transform]: train, validation and test transforms
    '''

    transform_train = Compose(
        [
            ToDtype(torch.float32),
        ]
    )

    return transform_train, transform_train, transform_train


