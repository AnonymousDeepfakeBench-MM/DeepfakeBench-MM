import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from metrics.parser import parse_metric
from utils.logger import write_log

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from metrics.builder import MetricBuilder
from utils.recorder import Recorder


class BaseTrainer(object):
    def __init__(self, config, train_data_loader, val_data_loader_dict, model, optimizer, scheduler, logger, log_dir):
        self.config = config
        self.train_data_loader = train_data_loader
        self.val_data_loader_dict = val_data_loader_dict
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.log_dir = log_dir
        self.metric_factory = MetricBuilder(config["metric_scoring"], config['val_dataset'])
        self.best_average_score = None
        self.device = torch.device(f"cuda:{config['local_rank']}" if torch.cuda.is_available() else "cpu")

        # Move model to current CUDA device if available
        if torch.cuda.is_available():
            lr = int(self.config["local_rank"])
            torch.cuda.set_device(lr)
            self.model = self.model.to(lr)
        else:
            self.model = self.model.to(self.device)

        if self.config["ddp"]:
            local_rank = int(self.config["local_rank"])
            # find_unused_parameters True may be slower; set to False if possible
            self.model = DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank,
                                                 find_unused_parameters=True)

    def _save_ckpt(self, save_dir, ckpt_name):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{ckpt_name}.pth")
        if self.config["ddp"] and self.config["rank"] == 0:
            torch.save(self.model.module.state_dict(), save_path)
        elif not self.config["ddp"]:
            torch.save(self.model.state_dict(), save_path)

        write_log(f"A checkpoint has successfully been saved  at {save_path}", self.logger, self.config["rank"])

    def _train_step(self, data_dict):
        predictions = self.model(data_dict)
        if self.config["ddp"]:
            losses = self.model.module.get_losses(data_dict, predictions)
        else:
            losses = self.model.get_losses(data_dict, predictions)

        if "overall" not in losses.keys():
            raise KeyError("[Abort] get_losses() must return a dict with key 'overall' as the optimization loss.")

        self.optimizer.zero_grad()
        losses["overall"].backward()
        self.optimizer.step()

        return losses, predictions

    @torch.no_grad()
    def _val_step(self, data_dict):
        return self.model(data_dict, inference=True)

    def train_epoch(self, epoch):
        # find out steps to start validation
        val_times_per_epoch = self.config.get("val_frequency", 1)
        val_steps = np.linspace(0, len(self.train_data_loader), val_times_per_epoch + 1).astype(np.int32).tolist()[1:]

        if self.config["rank"] == 0:
            pbar = tqdm(total=len(self.train_data_loader), desc=f"[Epoch {epoch}] Training",
                        postfix={"Total Loss": "N/A"}, dynamic_ncols=True)
        self.model.train()
        train_recorder_loss = defaultdict(Recorder)
        for iteration, data_dict in enumerate(self.train_data_loader):
            # move tensors to CUDA if available
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            losses, predictions = self._train_step(data_dict)

            # record losses
            batch_size = data_dict.get("label", torch.tensor([0])).shape[0]
            for name, value in losses.items():
                v = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
                train_recorder_loss[name].update(v, num=batch_size)

            # update tqdm on main
            if self.config["rank"] == 0:
                pbar.set_postfix({"Total Loss": f'{train_recorder_loss["overall"].average():.6g}'})
                pbar.update(1)

            # validation
            if (iteration + 1) in val_steps:
                # write training loss to log
                write_log(f"[Epoch {epoch} Iter {iteration + 1}/{len(self.train_data_loader)}] Training Losses:",
                          self.logger, self.config["rank"])
                for k, v in train_recorder_loss.items():
                    write_log(f"{k} = {v.average() if v.average() is not None else 'N/A'}",
                              self.logger, self.config["rank"])

                # ensure each process finishes its work before validation starts
                if self.config["ddp"]:
                    dist.barrier()

                # start validation
                self._val_epoch(epoch, iteration)

        # clear loss recorders:
        for name, recorder in train_recorder_loss.items():
            recorder.clear()

        # close progress bar
        if self.config["rank"] == 0:
            pbar.close()

        # save checkpoint after each epoch
        if self.config.get("save_ckpt", False):
            self._save_ckpt(os.path.join(self.log_dir, "checkpoints"), f"epoch{epoch + 1}")

    def _val_one_dataset(self, dataset_name, val_dataloader):
        if self.config["rank"] == 0:
            pbar = tqdm(total=len(val_dataloader), desc=f"[Validating {dataset_name}]", dynamic_ncols=True)
        val_recorder_loss = defaultdict(Recorder)
        records_on_current_gpu = []
        self.model.eval()
        with torch.no_grad():
            for iteration, data_dict in enumerate(val_dataloader):
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(self.device)

                predictions = self._val_step(data_dict)

                # record loss or predicted probability to be fake
                batch_size = data_dict.get("label", torch.tensor([0])).shape[0]
                if self.config["metric_scoring"] == "loss":
                    if self.config["ddp"]:
                        losses = self.model.module.get_losses(data_dict, predictions)
                    else:
                        losses = self.model.get_losses(data_dict, predictions)

                    for name, value in losses.items():
                        v = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
                        val_recorder_loss[name].update(v, num=batch_size)
                else:
                    for idx in range(batch_size):
                        single_data = {"path": data_dict["path"][idx],
                                       "label": data_dict["label"][idx].detach().cpu().item(),
                                       "video_label": data_dict["video_label"][idx].detach().cpu().item(),
                                       "audio_label": data_dict["audio_label"][idx].detach().cpu().item(),
                                       "prob": predictions["prob"][idx].detach().cpu().item(),
                                       # "logit": predictions["cls"][idx].detach().cpu().tolist()
                                       }
                        records_on_current_gpu.append(single_data)

                # update progress bar
                if self.config["rank"] == 0:
                    pbar.update(1)

        # close progress bar
        if self.config["rank"] == 0:
            pbar.close()

        # gather loss or prediction results and calculate metric score
        if self.config["ddp"]:
            dist.barrier()

        print(self.config["ddp"], self.config["metric_scoring"])
        if self.config["metric_scoring"] == "loss":
            # Gather losses if ddp
            if self.config["ddp"]:
                write_log("Synchronizing loss recorder...", self.logger, self.config["rank"])
                for name, value in val_recorder_loss.items():
                    val_recorder_loss[name].sync(self.device)
                    dist.barrier()
                write_log("Synchronizing loss recorder [Done]", self.logger, self.config["rank"])
            # Update loss record
            # if self.config["rank"] == 0:
            self.metric_factory.update(val_recorder_loss["overall"], dataset_name)
        else:
            # Gather prediction records if ddp
            if self.config["ddp"]:
                write_log("Synchronizing prediction records...", self.logger, self.config["rank"])
                # if self.config["rank"] == 0:
                #     gathered_records = [None] * self.config["world_size"]
                #     dist.gather_object(records_on_current_gpu, gathered_records, dst=0)
                #
                #     all_records = []
                #     for i, record in enumerate(gathered_records):
                #         if record is not None:
                #             all_records.extend(record)
                # else:
                #     dist.gather_object(records_on_current_gpu, None, dst=0)

                gathered_records = [None] * self.config["world_size"]
                dist.all_gather_object(gathered_records, records_on_current_gpu)
                all_records = []
                for i, record in enumerate(gathered_records):
                    if record is not None:
                        all_records.extend(record)

                dist.barrier()
                write_log("Synchronizing prediction records [Done]", self.logger, self.config["rank"])
            else:
                all_records = records_on_current_gpu

            print("update metric factory")
            # Update loss record
            # if self.config["rank"] == 0:
            self.metric_factory.update(all_records, dataset_name)

        # clear loss recorders:
        for name, recorder in val_recorder_loss.items():
            recorder.clear()
        print(f"finish val {dataset_name}")

    def _val_epoch(self, epoch, iteration):
        self.model.eval()
        score_dict = {}
        for key, value in self.val_data_loader_dict.items():
            self._val_one_dataset(key, value)
            if self.config["ddp"]:
                dist.barrier()

        # only main proces has metric records and needs to save checkpoint
        # if self.config["rank"] == 0:
        print("update best metric")
        update_list = self.metric_factory.update_best()
        print("save checkpoint")
        for update_item in update_list:
            if self.config["rank"] == 0:
                self._save_ckpt(os.path.join(self.log_dir, "checkpoints"), f"{update_item}_best")

        write_log(f"[Epoch {epoch} Iter {iteration + 1}/{len(self.train_data_loader)}] Validation Result:\n"
                         f"{self.metric_factory.parse_metrics(include_latest=True)}", self.logger, self.config["rank"])

        # write_log(f"[Epoch {epoch} Iter {iteration + 1}/{len(self.train_data_loader)}] Validation Metric:\n"
        #           f"{parse_metric(self.config['metric_scoring'], score_dict)}", self.logger, self.config["rank"])
        #
        # if self.config["rank"] == 0:
        #     if average_score != "N/A" and self.metric_factory.update_best(self.best_average_score, average_score):
        #         self.best_average_score = average_score
        #         write_log(f"[Epoch {epoch} Iter {iteration + 1}/{len(self.train_data_loader)}] Achieves Best Metric:\n"
        #                   f"{parse_metric(self.config['metric_scoring'], score_dict)}", self.logger, self.config["rank"])
        #         self._save_ckpt(os.path.join(self.log_dir, "checkpoints"), "best")

        self.model.train()
