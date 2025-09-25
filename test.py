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
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from datasets.builder import prepare_testing_data
from detectors import DETECTOR
from metrics.builder import MetricBuilder
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
    parser.add_argument("--test_metric", nargs="+", required=True, type=str, help="Test metrics")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (to override settings in YAML config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (to override settings in YAML config file")
    parser.add_argument("--save_feat", action="store_true",
                        help="Enable saving features for each test data (to override settings in YAML config file")
    parser.add_argument("--mask_modality", default=None, choices=["video", "audio"], type=str,
                        help="Mask video modality or audio modality or none of them")
    parser.add_argument("--delay", default=0, type=float, help="Set modality delay in seconds")

    return parser.parse_args()


def main():
    args = arg_parse()

    # load base path config and detector config
    with open(os.path.join(os.path.dirname(__file__), 'configs/path.yaml'), "r") as f:
        config = yaml.safe_load(f)
    with open(args.detector_path, "r") as f:
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
    if args.test_metric is not None:
        config["test_metric"] = args.test_metric
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.mask_modality is not None:
        config["mask_modality"] = args.mask_modality
    if args.delay != 0:
        config["delay"] = args.delay


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
    log_dir = os.path.join(config["log_dir"], "test", f"{config['model_name']}_{time_now}")
    logger = create_logger(os.path.join(log_dir, "testing.log"))
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

    # prepare data and test
    test_data_loader_dict = prepare_testing_data(config)
    metric_factory_dict = {metric_name: MetricBuilder(metric_name, config["test_dataset"])
                           for metric_name in config["test_metric"]}
    with torch.no_grad():
        for dataset_name in config["test_dataset"]:
            records_on_current_gpu = []
            test_dataloader = test_data_loader_dict[dataset_name]
            if config["rank"] == 0:
                pbar = tqdm(total=len(test_dataloader), desc=f"[Testing {dataset_name}]", dynamic_ncols=True)
            for iteration, data_dict in enumerate(test_dataloader):
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(device)

                predictions = model(data_dict, inference=True)

                # record predicted probability to be fake
                batch_size = data_dict.get("label", torch.tensor([0])).shape[0]
                for idx in range(batch_size):
                    single_data = {"path": data_dict["path"][idx],
                                   "label": data_dict["label"][idx].detach().cpu().item(),
                                   "video_label": data_dict["video_label"][idx].detach().cpu().item(),
                                   "audio_label": data_dict["audio_label"][idx].detach().cpu().item(),
                                   "prob": predictions["prob"][idx].detach().cpu().item(),
                                   }
                    records_on_current_gpu.append(single_data)

                # update progress bar
                if config["rank"] == 0:
                    pbar.update(1)

            # close progress bar
            if config["rank"] == 0:
                pbar.close()

            # gather prediction results
            if config["ddp"]:
                dist.barrier()
                write_log("Synchronizing prediction records...", logger, config["rank"])
                gathered_records = [None] * config["world_size"]
                dist.all_gather_object(gathered_records, records_on_current_gpu)
                all_records = []
                for i, record in enumerate(gathered_records):
                    if record is not None:
                        all_records.extend(record)

                dist.barrier()
                write_log("Synchronizing prediction records [Done]", logger, config["rank"])
            else:
                all_records = records_on_current_gpu

            # save prediction records
            if config["rank"] == 0:
                os.makedirs(os.path.join(log_dir, "predictions"), exist_ok=True)
                with open(os.path.join(log_dir, "predictions", f"{dataset_name}.json"), "w") as f:
                    json.dump(all_records, f, indent=4)

            # compute metric
            for metric_name in config["test_metric"]:
                metric_factory_dict[metric_name].update(all_records, dataset_name)

    # show metric
    for metric_name in config["test_metric"]:
        metric_factory_dict[metric_name].update_best()
        write_log(f"Test Metric: {metric_name}\n"
                  f"{metric_factory_dict[metric_name].parse_metrics()}", logger, config["rank"])

if __name__ == "__main__":
    main()
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
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from datasets.builder import prepare_testing_data
from detectors import DETECTOR
from metrics.builder import MetricBuilder
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
    parser.add_argument("--test_metric", nargs="+", required=True, type=str, help="Test metrics")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (to override settings in YAML config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (to override settings in YAML config file")
    parser.add_argument("--save_feat", action="store_true",
                        help="Enable saving features for each test data (to override settings in YAML config file")
    parser.add_argument("--mask_modality", default=None, choices=["video", "audio"], type=str,
                        help="Mask video modality or audio modality or none of them")
    parser.add_argument("--delay", default=0, type=float, help="Set modality delay in seconds")

    return parser.parse_args()


def main():
    args = arg_parse()

    # load base path config and detector config
    with open(os.path.join(os.path.dirname(__file__), 'configs/path.yaml'), "r") as f:
        config = yaml.safe_load(f)
    with open(args.detector_path, "r") as f:
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
    if args.test_metric is not None:
        config["test_metric"] = args.test_metric
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.mask_modality is not None:
        config["mask_modality"] = args.mask_modality
    if args.delay != 0:
        config["delay"] = args.delay


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
    log_dir = os.path.join(config["log_dir"], "test", f"{config['model_name']}_{time_now}")
    logger = create_logger(os.path.join(log_dir, "testing.log"))
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

    # prepare data and test
    test_data_loader_dict = prepare_testing_data(config)
    metric_factory_dict = {metric_name: MetricBuilder(metric_name, config["test_dataset"])
                           for metric_name in config["test_metric"]}
    with torch.no_grad():
        for dataset_name in config["test_dataset"]:
            records_on_current_gpu = []
            test_dataloader = test_data_loader_dict[dataset_name]
            if config["rank"] == 0:
                pbar = tqdm(total=len(test_dataloader), desc=f"[Testing {dataset_name}]", dynamic_ncols=True)
            for iteration, data_dict in enumerate(test_dataloader):
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(device)

                predictions = model(data_dict, inference=True)

                # record predicted probability to be fake
                batch_size = data_dict.get("label", torch.tensor([0])).shape[0]
                for idx in range(batch_size):
                    single_data = {"path": data_dict["path"][idx],
                                   "label": data_dict["label"][idx].detach().cpu().item(),
                                   "video_label": data_dict["video_label"][idx].detach().cpu().item(),
                                   "audio_label": data_dict["audio_label"][idx].detach().cpu().item(),
                                   "prob": predictions["prob"][idx].detach().cpu().item(),
                                   }
                    records_on_current_gpu.append(single_data)

                # update progress bar
                if config["rank"] == 0:
                    pbar.update(1)

            # close progress bar
            if config["rank"] == 0:
                pbar.close()

            # gather prediction results
            if config["ddp"]:
                dist.barrier()
                write_log("Synchronizing prediction records...", logger, config["rank"])
                gathered_records = [None] * config["world_size"]
                dist.all_gather_object(gathered_records, records_on_current_gpu)
                all_records = []
                for i, record in enumerate(gathered_records):
                    if record is not None:
                        all_records.extend(record)

                dist.barrier()
                write_log("Synchronizing prediction records [Done]", logger, config["rank"])
            else:
                all_records = records_on_current_gpu

            # save prediction records
            if config["rank"] == 0:
                os.makedirs(os.path.join(log_dir, "predictions"), exist_ok=True)
                with open(os.path.join(log_dir, "predictions", f"{dataset_name}.json"), "w") as f:
                    json.dump(all_records, f, indent=4)

            # compute metric
            for metric_name in config["test_metric"]:
                metric_factory_dict[metric_name].update(all_records, dataset_name)

    # show metric
    for metric_name in config["test_metric"]:
        metric_factory_dict[metric_name].update_best()
        write_log(f"Test Metric: {metric_name}\n"
                  f"{metric_factory_dict[metric_name].parse_metrics()}", logger, config["rank"])

if __name__ == "__main__":
    main()
