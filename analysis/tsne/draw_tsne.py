import argparse
import datetime
import json
import numpy as np
import os
import random
import sys
import warnings
import yaml

warnings.filterwarnings("ignore")
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, proj_root)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", required=True, type=str, help="root path of saved features")
    parser.add_argument("--dataset", required=True, type=str, help="dataset name which to be drawn TSNE with")
    return parser.parse_args()

def filter_data(data_list, feature_root):
    """
    You need to write function here to extract desired number of data and group them
    Args:
        data_list: (list) entire test data list
        feature_root: (str) root path of saved features
    Returns:
        features: (ndarray in [num_data, num_dim]) concatenated features
        class_indices: (ndarray in [num_data, ]) class index list
        class_name: (dict) class index -> class name dictionary
        color_list: (list) color list
    """
    feature_list = []
    class_list = []

    # FakeAVCeleb RARV, FARV, RAFV, FAFV
    # class_name = {0: "RARV", 1: "RAFV", 2: "FARV", 3: "FAFV"}
    # color_list = {"red", "yellow", "green", "blue"}   # customize color settings if you don't use pre-defined color map
    # rarv_list = random.sample([data for data in data_list if "RealVideo-RealAudio" in data["path"]], 97)
    # rafv_list = random.sample([data for data in data_list if "FakeVideo-RealAudio" in data["path"]], 97)
    # farv_list = random.sample([data for data in data_list if "RealVideo-FakeAudio" in data["path"]], 97)
    # fafv_list = random.sample([data for data in data_list if "FakeVideo-FakeAudio" in data["path"]], 97)
    # data_list = rarv_list + rafv_list + farv_list + fafv_list
    # class_list = [0] * 97 + [1] * 97 + [2] * 97 + [3] * 97

    class_name = {0: "Real", 1: "FS", 2: "FR", 3: "EFS+FS", 4: "EFS+FR", 5: "FE+FR", 6: "FS+FR"}
    real_pipelines = ["/pipeline1/"]
    real_list = random.sample([data for data in data_list if any(s in data["path"] for s in real_pipelines)], 3000)
    fs_pipelines = ["/pipeline2/", "/pipeline5/"]
    fs_list = random.sample([data for data in data_list if any(s in data["path"] for s in fs_pipelines)], 500)
    fs_fr_pipelines = ["/pipeline10/", "/pipeline14/", "/pipeline18/", "/pipeline22/"]
    fs_fr_list = random.sample([data for data in data_list if any(s in data["path"] for s in fs_fr_pipelines)], 500)
    efs_fs_pipelines = ["/pipeline7/", "/pipeline11/"]
    efs_fs_list = random.sample([data for data in data_list if any(s in data["path"] for s in efs_fs_pipelines)], 500)
    efs_fr_pipelines = ["/pipeline8/", "/pipeline12/", "/pipeline16/", "/pipeline20/"]
    efs_fr_list = random.sample([data for data in data_list if any(s in data["path"] for s in efs_fr_pipelines)], 500)
    fe_fr_pipelines = ["/pipeline9/", "/pipeline13/", "/pipeline17/", "/pipeline21/"]
    fe_fr_list = random.sample([data for data in data_list if any(s in data["path"] for s in fe_fr_pipelines)], 500)
    fr_pipelines = ["/pipeline3/", "pipeline6", "pipeline15", "pipeline19"]
    fr_list = random.sample([data for data in data_list if any(s in data["path"] for s in fr_pipelines)], 500)
    data_list = real_list + fs_list + fs_fr_list + efs_fs_list + efs_fr_list + fe_fr_list + fr_list
    class_list = [0] * 3000 + [1] * 500 + [2] * 500 + [3] * 500 + [4] * 500 + [5] * 500 + [6] * 500
    color_list = [
        '#4A90E2',
        '#8E44AD',
        '#27AE60',
        '#E74C3C',
        '#F1C40F',
        '#F39C12',
        'pink'
    ]

    print("Start to load features ...")
    for data in tqdm(data_list):
        feature_path = os.path.join(feature_root, data["path"], "feature.npy")
        feature_list.append(np.load(feature_path))

    return np.stack(feature_list), np.array(class_list), class_name, color_list

def draw_tsne(features, class_indices, class_name, color_list, save_path):
    """
    Draw TSNE with given features and class indices
    Args:
        features: (ndarray in [num_data, num_dim]) concatenated features
        class_indices: (ndarray in [num_data, ]) class index list
        class_name: (dict) class index -> class name dictionary
        color_list: (list) color list
        save_path: (str) TSNE figure save path
    """
    print("Start to fit TSNE ...")
    # tsne = TSNE(n_components=2, perplexity=50, learning_rate=500, n_iter=2000, init='pca', random_state=1024)
    tsne = TSNE(n_components=2, perplexity=20, random_state=1024, learning_rate=250)
    tsne_results = tsne.fit_transform(features)

    print("Start to draw TSNE ...")
    cmap = ListedColormap(color_list)
    plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=class_indices, cmap="tab10", alpha=0.7, s=10)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=class_indices, cmap=cmap, s=8, alpha=1, edgecolors='none')

    cmap = scatter.cmap
    norm = scatter.norm

    # Only include classes that appear in class_indices
    unique_classes, counts = np.unique(class_indices, return_counts=True)

    # Print each class and its count
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} samples")

    handles = [
        mpatches.Patch(color=cmap(norm(i)), label=f"{class_name[i]}")
        for i in unique_classes
    ]
    plt.legend(handles=handles)

    # Remove ticks and axis labels
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    # plt.show()

if __name__ == "__main__":
    args = arg_parse()
    with open(os.path.join(proj_root, "configs/path.yaml"), "r") as f:
        path_config = yaml.safe_load(f)

    with open(os.path.join(path_config["json_dir"], args.dataset + ".json"), "r") as f:
        data_list = json.load(f)["test"]

    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(args.feature_root, f"TSNE-{args.dataset}-{time_now}.png")

    features, class_indices, class_name, color_list = filter_data(data_list, args.feature_root)
    draw_tsne(features, class_indices, class_name, color_list, save_path)