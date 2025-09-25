import albumentations as A
import av
import cv2
import json
import io
import librosa
import lmdb
import numpy as np
import os
import pickle
import python_speech_features
import random

import scipy.signal
import torch
import torch.nn.functional as F
import warnings
import yaml

warnings.filterwarnings("ignore")

from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain
from torch.utils.data import Dataset, DataLoader


class AudioVideoDataset(Dataset):
    def __init__(self, config=None, mode="train"):
        assert mode in ["train", "val", "test"]
        self.config = config
        self.mode = mode
        self.video_aug = self.init_video_aug_method()
        self.audio_aug = self.init_audio_aug_method()
        self.data_list = []
        self._env = None  # lazy open per worker
        self._lmdb_path = None

        if mode == "train":  # Only in train mode, a list of dataset name provided in config["train_dataset"].
            dataset_list = config["train_dataset"]
            if self.config.get("lmdb", False):
                if len(dataset_list) > 1:
                    raise NotImplementedError("Dataset for more than 1 dataset has not been supported.")
                else:
                    # lmdb_path = os.path.join(self.config["lmdb_dir"], dataset_list[0])
                    # self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
                    self._lmdb_path = os.path.join(self.config["lmdb_dir"], dataset_list[0])

            for dataset_name in dataset_list:
                with open(os.path.join(config["json_dir"], f"{dataset_name}.json"), "r") as f:
                    self.data_list.extend(json.load(f)["train"])
        else:  # In val and test mode, a string of dataset name is provided in it.
            if self.config.get("lmdb", False):
                # lmdb_path = os.path.join(self.config["lmdb_dir"], config[f"{self.mode}_dataset"])
                # self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
                self._lmdb_path = os.path.join(self.config["lmdb_dir"], config[f"{self.mode}_dataset"])
            with open(os.path.join(config["json_dir"], f"{config[f'{mode}_dataset']}.json"), "r") as f:
                self.data_list = json.load(f)[mode]

    def _get_env(self):
        if self._env is None and self._lmdb_path is not None:
            self._env = lmdb.open(
                self._lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=1024,
                map_size=1 << 40
            )
        return self._env

    def init_video_aug_method(self):
        augmentation_config = self.config["augmentations"].get("video", {})
        trans = []
        if augmentation_config.get("flip", None) is not None:
            if augmentation_config["flip"].get("type", "horizontal") == "horizontal":
                trans.append(A.HorizontalFlip(p=augmentation_config["flip"].get("prob", 0.5)))
            else:
                raise NotImplementedError(f"{augmentation_config['flip']['type']} is not supported.")
        if augmentation_config.get("rotate", None) is not None:
            trans.append(A.Rotate(limit=augmentation_config["rotate"].get("rotate_limit", [-10, 10]),
                                  p=augmentation_config["rotate"].get("prob", 0.5)))
        if augmentation_config.get("gaussian_blur", None) is not None:
            trans.append(A.GaussianBlur(blur_limit=augmentation_config["gaussian_blur"].get("blur_limit", [3, 7]),
                                        p=augmentation_config["gaussian_blur"].get("prob", 0.5)))
        if augmentation_config.get("color", None) is not None:
            trans.append(A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=augmentation_config["color"].get("brightness_limit", [-0.1, 0.1]),
                    contrast_limit=augmentation_config["color"].get("contrast_limit", [-0.1, 0.1])
                ),
                A.FancyPCA(),
                A.HueSaturationValue()], p=augmentation_config["color"].get("prob", 0.5))
            )
        if augmentation_config.get("quality", None) is not None:
            trans.append(A.ImageCompression(quality_lower=augmentation_config["quality"].get("quality_lower", 40),
                                            quality_upper=augmentation_config["quality"].get("quality_upper", 100)))

        if self.config.get("with_landmarks", False):
            return A.ReplayCompose(trans, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
        else:
            return A.ReplayCompose(trans)

    def init_audio_aug_method(self):
        augmentation_config = self.config["augmentations"].get("audio", {})
        trans = []
        if augmentation_config.get("gaussian_noise", None) is not None:
            trans.append(AddGaussianNoise(min_amplitude=augmentation_config["gaussian_noise"].get("min", 0.001),
                                          max_amplitude=augmentation_config["gaussian_noise"].get("max", 0.01),
                                          p=augmentation_config["gaussian_noise"].get("prob", 0.5)))
        if augmentation_config.get("pitch_shift", None) is not None:
            trans.append(PitchShift(min_semitones=augmentation_config["pitch_shift"].get("min", -2),
                                    max_semitones=augmentation_config["pitch_shift"].get("max", 2),
                                    p=augmentation_config["pitch_shift"].get("prob", 0.5)))
        if augmentation_config.get("volume_gain", None) is not None:
            trans.append(Gain(min_gain_db=augmentation_config["volume_gain"].get("min", -6),
                              max_gain_db=augmentation_config["volume_gain"].get("max", 6),
                              p=augmentation_config["volume_gain"].get("prob", 0.5)))
        return Compose(trans)

    def crop_audio(self, audio, length, start_idx):
        """
        Crop 1s audio from long audio. A robust approach to avoiding fetch non-existing audio time stamp, e.g., -0.001s
        Args:
            audio: [np.ndarray] original long audio
            length: [int] number of elements of output ndarray, i.e., sr * duration
            start_idx: [int] starting ndarray index when cropping
        Returns:
            cropped_audio: [np.ndarray]
        """
        cropped_audio = np.zeros(length, dtype=audio.dtype)
        if start_idx >= len(audio) or start_idx + length < 0:
            return cropped_audio

        audio_start = max(start_idx, 0)
        audio_end = min(start_idx + length, len(audio))
        crop_start = max(0, -start_idx)
        crop_end = crop_start + (audio_end - audio_start)
        cropped_audio[crop_start:crop_end] = audio[audio_start:audio_end]

        return cropped_audio

    def __getitem__(self, index):
        data = self.data_list[index]

        # load data
        if self.config.get("lmdb", False):
            env = self._get_env()
            if env is None:
                raise RuntimeError("LMDB path not set but lmdb True in config.")

            with env.begin() as txn:
                raw = txn.get(data["path"].replace("\\", "/").encode())
                if raw is None:
                    raise KeyError(f"Missing key in lmdb: {data['path']}")
                raw_bytes = bytes(raw)

            lmdb_data = pickle.loads(raw_bytes)
            container = av.open(io.BytesIO(lmdb_data["video"]))
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            frames = np.stack(frames, axis=0)
            landmarks = lmdb_data["landmarks"]
            audio, sr = librosa.load(io.BytesIO(lmdb_data["audio"]), sr=None)
        else:
            data_path = os.path.join(self.config["preprocess_dir"], data["path"])
            container = av.open(os.path.join(data_path, "frames.mp4"))
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            frames = np.stack(frames, axis=0)
            landmarks = np.load(os.path.join(data_path, "landmarks.npy"))
            audio, sr = librosa.load(os.path.dirname(data_path) + ".wav", sr=None)

        # data augmentation
        frame_range = os.path.basename(data["path"]).split("_")
        if self.mode == "train":
            if self.config.get("augmentations"):
                # audio-video augmentation, in train mode, random delay audio using delay_range (in seconds)
                if self.config["augmentations"].get("audio_video"):
                    delay_range = self.config["augmentations"]["audio_video"].get("delay", [-0.1, 0.1])
                    delay = random.randint(delay_range[0] * sr / data["FPS"], delay_range[1] * sr / data["FPS"])
                else:
                    delay = 0
                audio = self.crop_audio(audio=audio, length=int(sr * data["duration"]),
                                        start_idx=int(int(frame_range[0]) * sr / data["FPS"] + delay))

                # video augmentation
                if self.config["augmentations"].get("video"):
                    for i in range(frames.shape[0]):
                        kwargs = {"image": frames[i]}
                        if self.config.get("with_landmarks", False):
                            kwargs["keypoints"] = [tuple(pt) for pt in landmarks[i]]

                        if i == 0:
                            augmented = self.video_aug(**kwargs)
                            replay_params = augmented["replay"]
                        else:
                            augmented = A.ReplayCompose.replay(replay_params, **kwargs)

                        frames[i] = augmented["image"]
                        if self.config.get("with_landmarks", False):
                            landmarks[i] = np.array(augmented["keypoints"])
                # audio augmentation
                if self.config["augmentations"].get("audio"):
                    audio = self.audio_aug(samples=audio, sample_rate=sr)
        elif self.mode == "val":        # When validation, no delay.
            audio = self.crop_audio(audio=audio, length=int(sr * data["duration"]),
                                    start_idx=int(int(frame_range[0]) * sr / data["FPS"]))
        elif self.mode == "test":       # When test, optional choice is to set delay command argument (in seconds)
            audio = self.crop_audio(audio=audio, length=int(sr * data["duration"]),
                                    start_idx=int(int(frame_range[0]) * sr / data["FPS"] + self.config.get("delay", 0) * sr))

        # when test, masked modality if option provided
        if self.mode == "test":
            if self.config.get("mask_modality", None) == "video":
                frames = np.zeros_like(frames)
            elif self.config.get("mask_modality", None) == "audio":
                audio = np.zeros_like(audio)

        # Resize to video resolution setting
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=self.config["video_resolution"], mode="bilinear", align_corners=False)
        landmarks = landmarks * self.config["video_resolution"] / 256

        # normalize
        frames = frames.permute(1, 0, 2, 3)
        frames = (frames / 255.0 - torch.tensor(self.config["mean"]).reshape(3, 1, 1, 1)) / torch.tensor(
            self.config["std"]).reshape(3, 1, 1, 1)

        # audio spectrogram conversion
        if self.config.get("audio_conversion", None) is None:
            pass
        elif self.config.get("audio_conversion") == "STFT":
            raise NotImplementedError(f"{self.config['audio_conversion_backend']} MFCC not implemented.")
            # audio = librosa.stft(y=audio, **self.config.get("audio_conversion_params", {}))
        elif self.config.get("audio_conversion") == "Mel":
            raise NotImplementedError(f"{self.config['audio_conversion_backend']} MFCC not implemented.")
            # audio = librosa.feature.melspectrogram(y=audio, sr=sr, **self.config.get("audio_conversion_params", {}))
        elif self.config.get("audio_conversion") == "Log-Mel":
            if self.config["audio_conversion_backend"] == "librosa":
                audio = librosa.feature.melspectrogram(y=audio, sr=sr, **self.config.get("audio_conversion_params", {}))
                audio = librosa.power_to_db(audio)
            elif self.config["audio_conversion_backend"] == "python_speech_features":
                audio = (audio * 32768).astype(np.int16)
                audio = python_speech_features.logfbank(audio, samplerate=sr, **self.config.get("audio_conversion_params", {}))
            else:
                raise NotImplementedError(f"{self.config['audio_conversion_backend']} MFCC not implemented.")
        elif self.config.get("audio_conversion") == "MFCC":
            if self.config["audio_conversion_backend"] == "python_speech_features":
                audio = (audio * 32768).astype(np.int16)
                audio = python_speech_features.mfcc(audio, sr, **self.config.get("audio_conversion_params", {}))
            else:
                raise NotImplementedError(f"{self.config['audio_conversion_backend']} MFCC not implemented.")
        else:
            raise NotImplementedError(f"{self.config['audio_conversion']} has not been implemented")

        if self.config.get("mask_modality", None) == "video":
            return {"video": frames,
                    "audio": torch.from_numpy(audio).float(),
                    "landmarks": torch.from_numpy(landmarks).float(),
                    "video_label": 0,
                    "audio_label": data["audio_label"],
                    "label": data["audio_label"],
                    "path": data["path"]}
        elif self.config.get("mask_modality", None) == "audio":
            return {"video": frames,
                    "audio": torch.from_numpy(audio).float(),
                    "landmarks": torch.from_numpy(landmarks).float(),
                    "video_label": data["video_label"],
                    "audio_label": 0,
                    "label": data["video_label"],
                    "path": data["path"]}
        else:
            return {"video": frames,
                    "audio": torch.from_numpy(audio).float(),
                    "landmarks": torch.from_numpy(landmarks).float(),
                    "video_label": data["video_label"],
                    "audio_label": data["audio_label"],
                    "label": data["label"],
                    "path": data["path"]}

    def __len__(self):
        return len(self.data_list)

    # @staticmethod
    # def collate_fn(batch):
    #     return data_dict["video"]


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "path.yaml"), "r") as f:
        path_config = yaml.safe_load(f)

    test_config = {
        "lmdb": True,
        # "audio_conversion": "MFCC",
        # "audio_conversion_backend": "python_speech_features",
        "video_resolution": 128,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "train_dataset": ["FakeAVCeleb"],
        "val_dataset": "FakeAVCeleb",
        "test_dataset": "FakeAVCeleb",
        "with_landmarks": True,
        "augmentations": {
            # "video": {
            #     "flip": {"type": "horizontal", "prob": 0.5},
            #     "rotate": {"rotate_limit": [-50, 50], "prob": 1},
            #     "gaussian_blur": {"blur_limit": [3, 7], "blur_prob": 0.5},
            #     "color": {"brightness_limit": [-0.1, 0.1], "contrast_limit": [-0.1, 0.1], "prob": 0.5},
            #     "quality": {"quality_lower": 40, "quality_upper": 100},
            # },
            # "audio": {
            #     "gaussian_noise": {"min": 0.001, "max": 0.01, "prob": 0.5},
            #     "pitch_shift": {"min": -2, "max": 2, "prob": 0.5},
            #     "volume_gain": {"min": -6, "max": 6, "prob": 0.5}
            # },
            # "audio_video": {"delay": [-0.1, 0.1]}
        },
        # "mask_modality": "audio",
        "delay": 0.5,
    }

    dataset = AudioVideoDataset(config={**path_config, **test_config}, mode="test")
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8)
    from tqdm import tqdm
    for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # pass
        for key, value in batch_data.items():
            if key in ["audio", "video", "landmarks"]:
                print(key, value.shape, value.dtype)
            else:
                print(key, value)

        # Write landmarks on frames of the first vide
        for i in range(batch_data["video"].shape[2]):
            img = batch_data["video"].permute(0, 2, 3, 4, 1).numpy()
            img = img * np.array(test_config["std"]).reshape(1, 1, 1, 1, 3) + \
                  np.array(test_config["mean"]).reshape(1, 1, 1, 1, 3)
            img = (img * 255).astype(np.uint8)
            img = img[0, i, :, :, ::-1].astype(np.uint8).copy()  # to BGR
            # for (x, y) in batch_data["landmarks"][0, i, :, :]:
            #     cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
            cv2.imwrite(f"temp_{i:04d}.png", img)
        import soundfile as sf
        sf.write("temp.wav", batch_data["audio"][0], 16000)
        break
