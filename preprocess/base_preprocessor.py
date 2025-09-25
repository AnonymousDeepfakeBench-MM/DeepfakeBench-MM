import copy
import cv2
import io
import json
import librosa
import lmdb
import numpy as np
import os
import pickle
import shutil
import sys

from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess.utils.tools import split_video_and_audio_multiprocess, video_and_audio_segmentation_multiprocess

class BasePreprocessor:
    def __init__(self, config, output_name):
        self.config = config
        self.output_name = output_name
        print(f'[Collecting data]')
        self.train_data_list, self.val_data_list, self.test_data_list = [], [], []
        if config['split'] in ['all', 'train']:
            self.train_data_list = self.collect_data_attributes('train')
            print(f'Totally training {len(self.train_data_list)} audio-video found.')
        if config['split'] in ['all', 'val']:
            self.val_data_list = self.collect_data_attributes('val')
            print(f'Totally validation {len(self.val_data_list)} audio-video found.')
        if config['split'] in ['all', 'test']:
            self.test_data_list = self.collect_data_attributes('test')
            print(f'Totally testing {len(self.test_data_list)} audio-video found.')
        self.data_list = self.train_data_list + self.val_data_list + self.test_data_list

    def separate_stream(self):
        print(f'[Separating video and audio stream] '
              f'Adjust video to {self.config["FPS"]} FPS and audio to {self.config["sample_rate"]}Hz sampling rate.')
        split_video_and_audio_multiprocess(self.data_list, self.config.get('num_proc', os.cpu_count()))

    def split_clip(self):
        print(f'[Preprocessing] Start to detect, align, and crop face for each frame. Segment video and audio.')
        video_and_audio_segmentation_multiprocess(self.data_list, self.config.get('num_proc', os.cpu_count()))

    def assemble(self):
        def assemble_data_split(data_split, data_list):
            for i in tqdm(range(len(data_list)), dynamic_ncols=True):
                data = data_list[i]
                save_root = data['video_path'][:-4]
                # No segment saved in preprocess()
                if not os.path.exists(save_root):
                    continue

                # rearrange json dict for every clip, record relative path instead of absolute path
                for segment_name in list(Path(save_root).glob('*')):
                    clip_json = copy.deepcopy(data)
                    clip_json['path'] = str(segment_name.relative_to(self.config['preprocess_dir']))
                    if clip_json.get('original_path', None) is not None:
                        clip_json['original_path'] = str(Path(clip_json['original_path']).relative_to(self.config['original_dir'])).replace('\\', '/')
                    elif clip_json.get('original_video_path', None) is not None and clip_json.get('original_audio_path', None) is not None:
                        clip_json['original_video_path'] = str(Path(clip_json['original_video_path']).relative_to(self.config['original_dir'])).replace('\\', '/')
                        clip_json['original_audio_path'] = str(Path(clip_json['original_audio_path']).relative_to(self.config['original_dir'])).replace('\\', '/')
                    del clip_json['video_path'], clip_json['audio_path'], clip_json['clip_amount']
                    all_data_json[data_split].append(clip_json)

        print(f'[Assembling] Assemble preprocessed data with labels and metadata, write into json')
        all_data_json = {'train': [], 'val': [], 'test': []}
        assemble_data_split('train', self.train_data_list)
        assemble_data_split('val', self.val_data_list)
        assemble_data_split('test', self.test_data_list)

        os.makedirs(os.path.join(self.config['json_dir']), exist_ok=True)
        with open(os.path.join(self.config['json_dir'], f'{self.output_name}.json'), 'w') as f:
            json.dump(all_data_json, f, indent=4)

    # def convert_lmdb(self):
    #     lmdb_path = os.path.join(self.config['lmdb_dir'], self.output_name)
    #     if os.path.exists(lmdb_path):
    #         shutil.rmtree(lmdb_path)
    #
    #     with open(os.path.join(self.config['json_dir'], f'{self.output_name}.json'), 'r') as f:
    #         all_data_json = json.load(f)
    #
    #     os.makedirs(lmdb_path, exist_ok=True)
    #     data_list = all_data_json['train'] + all_data_json['val'] + all_data_json['test']
    #     map_size = len(data_list) * 3 * 1024 ** 2  # 3MB max size for each data
    #     env = lmdb.open(lmdb_path, map_size=map_size)
    #     txn = env.begin(write=True)
    #     with tqdm(total=len(data_list), dynamic_ncols=True) as pbar:
    #         for i, data in enumerate(data_list):
    #             buffer_io = io.BytesIO()
    #             np.savez_compressed(buffer_io, **load_data(os.path.join(self.config['preprocess_dir'], data['path'])))
    #             buffer = buffer_io.getvalue()
    #             txn.put(os.path.join(data['path']).replace('\\', '/').encode(), buffer)
    #             if (i + 1) % 256 == 0:
    #                 txn.commit()
    #                 txn = env.begin(write=True)
    #             pbar.update(1)
    #
    #     txn.commit()
    #     env.sync()
    #     env.close()

    def convert_lmdb(self):
        lmdb_path = os.path.join(self.config['lmdb_dir'], self.output_name)
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)

        with open(os.path.join(self.config['json_dir'], f'{self.output_name}.json'), 'r') as f:
            all_data_json = json.load(f)
        os.makedirs(lmdb_path, exist_ok=True)
        data_list = all_data_json['train'] + all_data_json['val'] + all_data_json['test']

        map_size = len(data_list) * 1024 ** 2  # 1MB max size for each data
        env = lmdb.open(lmdb_path, map_size=map_size)
        txn = env.begin(write=True)
        with tqdm(total=len(data_list), dynamic_ncols=True) as pbar:
            for i, data in enumerate(data_list):
                data_path = os.path.join(self.config['preprocess_dir'], data['path'])
                with open(os.path.join(data_path, 'frames.mp4'), 'rb') as f:
                    video_bytes = f.read()
                with open(os.path.dirname(data_path) + '.wav', 'rb') as f:
                    audio_bytes = f.read()
                landmarks = np.load(os.path.join(data_path, 'landmarks.npy')).astype(np.float32)
                temp = {'video': video_bytes, 'audio': audio_bytes, 'landmarks': landmarks}
                txn.put(os.path.join(data['path']).replace('\\', '/').encode(), pickle.dumps(temp))
                if (i + 1) % 1024 == 0:     # Commit every 1GB (max)
                    txn.commit()
                    txn = env.begin(write=True)
                pbar.update(1)

        txn.commit()
        env.sync()
        env.close()


