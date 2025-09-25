import os
import sys

import torch
from torch.utils.data import DistributedSampler

sys.path.insert(0, os.path.dirname(__file__))

from datasets.audio_video_dataset import AudioVideoDataset


def prepare_training_data(config):
    """
    Prepare a training dataset class instance.
    Args:
        config (dict): Config dictionary.
            Required keys: train_dataset, train_batch_size, dataset related keys (e.g., mean, std, etc.)
            Optional keys: num_workers (default=4), pin_memory (default=True), persistent_workers (default=False).
    Returns:
         loader (torch.utils.data.DataLoader): Training data loader.
    """
    if config.get("dataset_type", None) is None:
        train_dataset = AudioVideoDataset(config, mode="train")
    else:
        raise NotImplementedError(f"Dataset {config['dataset_type']} is not implemented")

    sampler = DistributedSampler(train_dataset) if config.get("ddp", False) else None

    loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=(sampler is None),
        num_workers=config.get("num_workers", 4),
        drop_last=True,
        sampler=sampler,
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", False) if config.get("num_workers", 0) > 0 else False,
    )
    return loader


def prepare_validation_data(config):
    """
    Prepare validation dataset class instances. Construct a dict with "val_dataset_name" -> "val_dataloader" mappings.
    Args:
        config (dict): Config dictionary.
            Required keys: val_dataset, val_batch_size, dataset related keys (e.g., mean, std, etc.)
            Optional keys: num_workers (default=4), pin_memory (default=True), persistent_workers (default=False).
    Returns:
        val_data_loader_dict (dict): Validation data loader dict.
    """

    def get_val_data_loader(dataset_name):
        tmp_config = config.copy()
        tmp_config["val_dataset"] = dataset_name
        if config.get("dataset_type", None) is None:
            val_dataset = AudioVideoDataset(config=tmp_config, mode="val")
        else:
            raise NotImplementedError(f"Dataset {config['dataset_type']} is not implemented")
        sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if config.get("ddp", False) else None
        loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config["val_batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            drop_last=False,
            sampler=sampler,
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("persistent_workers", False) if config.get("num_workers", 0) > 0 else False,
        )
        return loader

    val_data_loader_dict = {}
    for val_dataset_name in config["val_dataset"]:
        val_data_loader_dict[val_dataset_name] = get_val_data_loader(val_dataset_name)
    return val_data_loader_dict


def prepare_testing_data(config):
    """
    Prepare testing dataset class instances. Construct a dict with "test_dataset_name" -> "test_dataloader" mappings.
    Args:
        config (dict): Config dictionary.
            Required keys: test_dataset, test_batch_size, dataset related keys (e.g., mean, std, etc.)
            Optional keys: num_workers (default=4), pin_memory (default=True), persistent_workers (default=False).
    Returns:
        test_data_loader_dict (dict): Testing data loader dict.
    """

    def get_test_data_loader(dataset_name):
        tmp_config = config.copy()
        tmp_config["test_dataset"] = dataset_name
        if config.get("dataset_type", None) is None:
            test_dataset = AudioVideoDataset(config=tmp_config, mode="test")
        else:
            raise NotImplementedError(f"Dataset {config['dataset_type']} is not implemented")
        sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False) if config.get("ddp", False) else None
        loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=config["test_batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            drop_last=False,
            sampler=sampler,
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("persistent_workers", False) if config.get("num_workers", 0) > 0 else False,
        )
        return loader

    test_data_loader_dict = {}
    for test_dataset_name in config["test_dataset"]:
        test_data_loader_dict[test_dataset_name] = get_test_data_loader(test_dataset_name)
    return test_data_loader_dict
