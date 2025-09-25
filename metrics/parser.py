import torch.distributed as dist

def parse_metric(metric_name, metric_dict):
    if dist.get_rank() != 0:
        return ""

    datasets = list(metric_dict.keys())

    # Calculate column width
    dataset_col_width = max(len("Dataset"), max(len(str(ds)) for ds in datasets))
    metric_col_width = max(len(metric_name), 10)

    # table header
    header = f"{'Dataset':<{dataset_col_width}}  {metric_name:<{metric_col_width}}"
    separator = "-" * dataset_col_width + "  " + "-" * metric_col_width

    # table content
    rows = []
    for dataset in datasets:
        value = metric_dict[dataset]
        if value is None:
            cell_value = "N/A"
        elif isinstance(value, (int, float)):
            cell_value = f"{value:.4g}"
        else:
            cell_value = str(value)

        row = f"{dataset:<{dataset_col_width}}  {cell_value:<{metric_col_width}}"
        rows.append(row)

    # if multiple dataset, print average metric
    if len(datasets) > 1:
        average_score = "N/A" if None in score_dict.values() else np.mean(np.array(list(score_dict.values())))
        if isinstance(average_score, (int, float)):
            avg_display = f"{average_score:.4g}"
        else:
            avg_display = str(average_score)

        row = f"{'Average':<{dataset_col_width}}  {avg_display:<{metric_col_width}}"
        table_parts = [header, separator] + rows + [separator, row]
    else:
        table_parts = [header, separator] + rows

    return "\n".join(table_parts)


if __name__ == "__main__":
    accuracy_metrics = {
        "cifar10": 0.85234567,
        "cifar100": 0.78123456,
        "imagenet": 0.91234567,
        "test_set": None,
    }

    valid_values = [v for v in accuracy_metrics.values() if v is not None]
    average_acc = sum(valid_values) / len(valid_values) if valid_values else "N/A"

    print(parse_metric("Accuracy", accuracy_metrics, average_acc))