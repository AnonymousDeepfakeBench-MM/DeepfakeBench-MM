import argparse
import datetime
import json
import os
import sys
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from datasets.builder import prepare_training_data, prepare_validation_data
from detectors import DETECTOR
from optimizers.builder import choose_optimizer
from schedulers.builder import choose_scheduler
from trainers.base_trainer import BaseTrainer
from tools.init_seed import set_seed
from tools.logger import create_logger, write_log


def arg_parse():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training Command Line Settings")
    parser.add_argument("--detector_path", required=True, type=str, help="Path to model config file")
    parser.add_argument("--ddp", action="store_true", help="Use DDP training")
    parser.add_argument("--local_rank", default=0, type=int, help="local_rank for distributed training")
    parser.add_argument("--train_datasets", nargs="+", default=None, type=str,
                        help="Training datasets (to override settings in YAML config file")
    parser.add_argument("--val_datasets", nargs="+", default=None, type=str,
                        help="Validation datasets (to override settings in YAML config file")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (to override settings in YAML config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (to override settings in YAML config file")
    parser.add_argument("--save_ckpt", action="store_true",
                        help="Enable saving checkpoints after each epoch (to override settings in YAML config file")

    return parser.parse_args()


def main():
    args = arg_parse()

    # load base path config and detector config
    with open(os.path.join(os.path.dirname(__file__), 'configs/path.yaml'), "r") as f:
        config = yaml.safe_load(f)
    with open(args.detector_path, "r") as f:
        config.update(yaml.safe_load(f))

    # override only if CLI args provided (avoid overriding with defaults)
    if args.train_datasets is not None:
        config["train_dataset"] = args.train_datasets
    if args.val_datasets is not None:
        config["val_dataset"] = args.val_datasets
    if args.save_ckpt:
        config["save_ckpt"] = True
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.seed is not None:
        config["seed"] = args.seed

    # torchrun sets these env vars. Use them if present for robustness.
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
    if args.ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
    config["ddp"] = args.ddp
    config["local_rank"] = local_rank
    config["rank"] = rank
    config["world_size"] = world_size

    # set random seed
    assert config.get('seed', None) is not None, "[Abort] A random seed must be provided in YAML config file."
    set_seed(config['seed'] + config['rank'], deterministic=config.get('deterministic', True))

    # create logger (only process with rank=0 can write to file and console)
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(config["log_dir"], "train", f"{config['model_name']}_{time_now}")
    logger = create_logger(os.path.join(log_dir, "training.log"))
    write_log(f"Save log to {log_dir}", logger, config["rank"])

    # write config into log file
    write_log("Config:\n{}".format(json.dumps(config, indent=4)), logger, config["rank"])

    # prepare model
    model = DETECTOR[config["model_name"]](config)
    if config.get("init_weight", None) is not None:
        ckpt = torch.load(config["init_weight"], map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        write_log(f"Missing keys (parameters that model needs but does not exist in checkpoint): {missing_keys}",
                  logger, config["rank"])
        write_log(f"Unexpected keys (parameters that model does not need but exist in checkpoint): {unexpected_keys}",
                    logger, config["rank"])

    # prepare data, optimizer, and scheduler
    train_data_loader = prepare_training_data(config)
    val_data_loader_dict = prepare_validation_data(config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(optimizer, config)

    # build up trainer and train
    trainer = BaseTrainer(config, train_data_loader, val_data_loader_dict, model, optimizer, scheduler, logger, log_dir)
    try:
        for epoch in range(config["num_epochs"]):
            if config["ddp"]:
                train_data_loader.sampler.set_epoch(epoch)

            trainer.train_epoch(epoch=epoch)

    except Exception as e:
        logger.warning(str(e))

if __name__ == "__main__":
    main()
