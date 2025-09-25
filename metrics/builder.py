import numpy as np

from prettytable import PrettyTable

from metrics.accuracy import calculate_binary_classification_accuracy
from metrics.auc import calculate_binary_classification_auc
from metrics.average_precision import calculate_binary_classification_average_precision
from metrics.eer import calculate_binary_classification_eer


class MetricBuilder:
    metrics_to_minimize = ["loss", "eer"]
    metrics_to_maximize = ["auc", "ap", "acc"]

    def __init__(self, metric_name, dataset_list):
        if metric_name not in self.metrics_to_minimize and metric_name not in self.metrics_to_maximize:
            raise NotImplementedError(f"{metric_name} has not been implemented.")
        self.metric_name = metric_name
        self.dataset_list = dataset_list + ["average"]

        if self.metric_name in self.metrics_to_maximize:
            self.metric_records = {dataset_name: [] for dataset_name in self.dataset_list}
            self.best_scores = {dataset_name: -float("inf") for dataset_name in self.dataset_list}
        else:
            self.metric_records = {dataset_name: [] for dataset_name in self.dataset_list}
            self.best_scores = {dataset_name: float("inf") for dataset_name in self.dataset_list}

    def update(self, records, dataset_name):
        """
        Calculate metric scores and update to records.
        Args:
            records (list or utils.recorder.Record): List of records (when metric is not loss) or loss records (otherwise).
        """
        ground_truths = np.array([record["label"] for record in records])
        video_ground_truths = np.array([record["video_label"] for record in records])
        audio_ground_truths = np.array([record["audio_label"] for record in records])
        prob = np.array([record["prob"] for record in records])

        if self.metric_name == "loss":
            current_score = records.average()
        elif self.metric_name == "binary":
            current_score = calculate_binary_classification_eer(ground_truths, prob)
        elif self.metric_name == "auc":
            current_score = calculate_binary_classification_auc(ground_truths, prob)
        elif self.metric_name == "ap":
            current_score = calculate_binary_classification_average_precision(ground_truths, prob)
        elif self.metric_name == "acc":
            current_score = calculate_binary_classification_accuracy(ground_truths, prob)
        else:
            raise NotImplementedError(f"{self.metric_name} has not been implemented.")

        self.metric_records[dataset_name].append(current_score)

    def _update_average(self):
        # safety check
        for dataset_name in self.dataset_list:
            if dataset_name != "average":
                assert len(self.metric_records[dataset_name]) == len(self.metric_records['average']) + 1

        latest_scores = [self.metric_records[dataset_name][-1] for dataset_name in self.dataset_list
                         if dataset_name != "average"]
        if float("inf") in latest_scores:
            self.metric_records["average"].append(float("inf"))
        elif -float("inf") in latest_scores:
            self.metric_records["average"].append(-float("inf"))
        else:
            self.metric_records["average"].append(np.array(latest_scores).mean())

    def _is_best(self, best_until_now, current_score):
        """
        Decide whether best_metric should be updated or not.
        Args:
            best_until_now (float): Best metric until now
            current_score (float): Current score
        Returns:
            is_best (bool): True if current_score is best
        """
        if self.metric_name in self.metrics_to_minimize:
            return current_score < best_until_now
        else:
            return current_score > best_until_now

    def update_best(self):
        self._update_average()
        update_list = []
        for dataset_name in self.dataset_list:
            if self._is_best(self.best_scores[dataset_name], self.metric_records[dataset_name][-1]):
                update_list.append(dataset_name)
                self.best_scores[dataset_name] = self.metric_records[dataset_name][-1]

        return update_list

    def parse_metrics(self, include_latest=False):
        table = PrettyTable()
        table.field_names = ["Dataset", "Latest", "Best"] if include_latest else ["Dataset", "Best"]
        for dataset_name in self.dataset_list:
            if include_latest:
                table.add_row([dataset_name,
                               f"{self.metric_records[dataset_name][-1]:.4f}",
                               f"{self.best_scores[dataset_name]:.4f}"])
            else:
                table.add_row([dataset_name,
                               f"{self.best_scores[dataset_name]:.4f}"])

        return table
