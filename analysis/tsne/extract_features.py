import argparse
import datetime
import json
import os
import sys
import warnings
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

warnings.filterwarnings("ignore")
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, proj_root)

from datasets.builder import prepare_testing_data
from detectors import DETECTOR
from utils.init_seed import set_seed
from utils.logger import create_logger, write_log


def arg_parse():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training Command Line Settings")
    parser.add_argument("--detector_path", required=True, type=str, help="Path to model config file")
    parser.add_argument("--weight_path", required=True, type=str, help="Path to model weights file")
    parser.add_argument("--ddp", action="store_true", help="Use DDP training")
    parser.add_argument("--local_rank", default=0, type=int, help="local_rank for distributed training")
    parser.add_argument("--test_dataset", nargs="+", default=None, type=str,
                        help="Test datasets (to override settings in YAML config file")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (to override settings in YAML config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (to override settings in YAML config file")
    parser.add_argument("--target_layer", type=str, required=True,
                        help="The model layer that your need to save output feature")

    return parser.parse_args()


def main():
    args = arg_parse()

    # load base path config and detector config
    with open(os.path.join(proj_root, 'configs/path.yaml'), "r") as f:
        config = yaml.safe_load(f)
    with open(os.path.join(proj_root, args.detector_path), "r") as f:
        config.update(yaml.safe_load(f))

    # override only if CLI args provided (avoid overriding with defaults)
    # if args.save_feat:
    #     config["save_feat"] = True
    if args.detector_path is not None:
        config["detector_path"] = args.detector_path
    if args.weight_path is not None:
        config["weight_path"] = args.weight_path
    if args.test_dataset is not None:
        config["test_dataset"] = args.test_dataset
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.target_layer is not None:
        config["target_layer"] = args.target_layer

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
    log_dir = os.path.join(config["log_dir"], "predictions", f"{config['model_name']}_{time_now}")
    logger = create_logger(os.path.join(log_dir, "extract_features.log"))
    write_log(f"Save log to {log_dir}", logger, config["rank"])

    # write config into log file
    write_log("Config:\n{}".format(json.dumps(config, indent=4)), logger, config["rank"])

    # prepare model
    model = DETECTOR[config["model_name"]](config)
    device = torch.device(f"cuda:{config['local_rank']}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(config["weight_path"], map_location="cpu")
    try:
        model.load_state_dict(ckpt, strict=True)
        write_log("Successfully loaded checkpoint from {}".format(config["weight_path"]), logger, config["rank"])
    except RuntimeError as e:
        write_log("Failed to load checkpoint: {}".format(e), logger, config["rank"])

    if torch.cuda.is_available():
        lr = int(config["local_rank"])
        torch.cuda.set_device(lr)
        model = model.to(lr)
    else:
        model = model.to(device)

    if config["ddp"]:
        local_rank = int(config["local_rank"])
        # find_unused_parameters True may be slower; set to False if possible
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                        find_unused_parameters=True)
    model.eval()

    # register forward hook
    current_features = None
    def hook_fn(module, input, output):
        nonlocal current_features
        current_features = output.detach().cpu().numpy()
    target_layer = None
    for name, module in model.named_modules():
        if name == args.target_layer:
            target_layer = module
            break
    if target_layer is not None:
        raise RuntimeError(f"You have named a non-existing layer, available layers: {[name for name, _ in model.named_modules()]}")
    hook = target_layer.register_forward_hook(hook_fn)

    # go through datasets
    test_data_loader_dict = prepare_testing_data(config)
    with torch.no_grad():
        for dataset_name in config["test_dataset"]:
            test_dataloader = test_data_loader_dict[dataset_name]
            if config["rank"] == 0:
                pbar = tqdm(total=len(test_dataloader), desc=f"[Saving Predictions of layer {args.target_layer} on {dataset_name}]", dynamic_ncols=True)
            for iteration, data_dict in enumerate(test_dataloader):
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(device)

                _ = model(data_dict, inference=True)
                print(current_features.shape)

                # record predicted probability to be fake
                batch_size = data_dict.get("label", torch.tensor([0])).shape[0]
                for idx in range(batch_size):
                    save_path = data_dict["path"][idx]
                    os.makedirs(os.path.join(log_dir , save_path), exist_ok=True)
                    np.save(os.path.join(log_dir, save_path, "feature.npy"),current_features[idx])

                # update progress bar
                if config["rank"] == 0:
                    pbar.update(1)

            # close progress bar
            if config["rank"] == 0:
                pbar.close()


    hook.remove()

if __name__ == "__main__":
    main()