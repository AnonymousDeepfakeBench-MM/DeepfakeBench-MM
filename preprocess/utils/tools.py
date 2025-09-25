import cv2
import dlib
import librosa
import multiprocessing

import math
import numpy as np
import os
import shutil
import soundfile as sf
import subprocess

from skimage import transform as tf
from tqdm import tqdm

dlib_tool_path = os.path.join(os.path.dirname(__file__))
mean_face_landmarks = np.load(os.path.join(dlib_tool_path, '20words_mean_face.npy'))
STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]
WINDOW_MARGIN = 12
FACE_SIZE = 256
MOUTH_SIZE = 96


def split_video_and_audio_multiprocess(data_list, num_proc):
    """
    Split video and audio from given list.
    Args:
        data_list: [list] each element is a dictionary containing attributes of this data.
        num_proc: [int] number of processors
    """
    if num_proc != 1:
        with multiprocessing.Pool(processes=num_proc) as pool:
            with tqdm(total=len(data_list), dynamic_ncols=True) as pbar:
                for _ in pool.imap_unordered(split_video_and_audio, data_list):
                    pbar.update(1)
    else:
        for data in tqdm(data_list, dynamic_ncols=True):
            split_video_and_audio(data)


def split_video_and_audio(data):
    """
    Split video and audio from one single *.mp4 file. Two usage cases:
        1. audio-video file input: use key 'original_path' to store audio-video file path.
        2. audio and video files: use key 'original_video_path' and 'original_audio_path' to store video and audio file
            path, respectively.
    Other Keys Used:
        1. FPS: video FPS
        2. sample_rate: audio sampling rate
        3. video_path: video stream separation and FPS adjustment output path
        4. audio_path: audio stream separation and sampling rate adjustment output path

    Args:
        data: [dict] a dictionary containing attributes of this data
    """
    # To avoid partially streamed file, we need to remove all existing files.
    if os.path.exists(data['video_path']):
        os.remove(data['video_path'])
    if os.path.exists(data['audio_path']):
        os.remove(data['audio_path'])

    os.makedirs(os.path.dirname(data['video_path']), exist_ok=True)
    # case 1: audio-video file input
    if data.get('original_path', None) is not None:
        try:
            cmd = ['ffmpeg', '-i', data['original_path'],
                   '-an', '-r', str(data['FPS']), '-c:v', 'libx264', '-crf', '23', data['video_path'],
                   '-vn', '-ar', str(data['sample_rate']), '-ac', '1', '-c:a', 'pcm_s16le', data['audio_path']]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Split audio and video failed: {data['original_path']}")
    # case 2: audio file and video file inputs
    elif data.get('original_video_path', None) is not None and data.get('original_audio_path', None) is not None:
        try:
            cmd = ['ffmpeg', '-i', data['original_video_path'], '-an', '-r', str(data['FPS']), '-c:v', 'libx264',
                   '-crf', '23', data['video_path']]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Adjusting video FPS failed: {data['original_video_path']}")

        try:
            cmd = ['ffmpeg', '-i', data['original_audio_path'], '-ar', str(data['sample_rate']), '-ac', '1',
                   '-c:a', 'pcm_s16le', data['audio_path']]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Adjusting audio sampling rate, format or channels failed: {data['original_audio_path']}")
    else:
        raise NotImplementedError('Unsupported original data format')


def video_and_audio_segmentation_multiprocess(data_list, num_proc):
    """
    Segment video and audio based on data_list.
    Args:
        data_list: [list] each element is a dictionary containing attributes of this data.
        num_proc: [int] number of processors
    """
    if num_proc != 1:
        with multiprocessing.Pool(processes=num_proc) as pool:
            with tqdm(total=len(data_list), dynamic_ncols=True) as pbar:
                for _ in pool.imap_unordered(video_and_audio_segmentation, data_list):
                    pbar.update(1)
    else:
        for data in tqdm(data_list, dynamic_ncols=True):
            video_and_audio_segmentation(data)


def video_and_audio_segmentation(data):
    """
    Segment video and corresponding audio if all consecutive frames in one required-length video segment have face detected.
    Args:
        data: [dict] data dictionary with file path stored.
    """

    # This part of codes is from https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation/align_mouth.py
    # ----- start of this part -----
    def warp_img(src, dst, img, std_size):
        tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped, tform

    def cut_patch(img, landmarks, height, width, threshold=5):
        center_x, center_y = np.mean(landmarks, axis=0)

        if center_y - height < 0:
            center_y = height
        if center_y - height < 0 - threshold:
            raise Exception('too much bias in height')
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception('too much bias in width')

        if center_y + height > img.shape[0]:
            center_y = img.shape[0] - height
        if center_y + height > img.shape[0] + threshold:
            raise Exception('too much bias in height')
        if center_x + width > img.shape[1]:
            center_x = img.shape[1] - width
        if center_x + width > img.shape[1] + threshold:
            raise Exception('too much bias in width')

        return np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                       int(round(center_x) - round(width)): int(round(center_x) + round(width))])

    # ----- end of this part -----

    def save_segment():
        """
        Save current audio-video segment to storage. Here, we use cropped faces in frame_list to construct 1s video. As
            for audio, we keep the full audio stream files.
        """
        # crop and save frames
        for i in range(len(frame_list)):
            smoothed_landmarks = np.mean(landmark_list[i:min(len(landmark_list), i + WINDOW_MARGIN)], axis=0)
            trans_frame, trans = warp_img(smoothed_landmarks[STABLE_POINTS, :],
                                          mean_face_landmarks[STABLE_POINTS, :],
                                          frame_list[i],
                                          STD_SIZE)
            landmark_list[i] = trans(landmark_list[i])
            frame_list[i] = cut_patch(trans_frame, landmark_list[i], FACE_SIZE // 2, FACE_SIZE // 2)

        # Write to disk
        os.makedirs(os.path.join(save_root, current_folder), exist_ok=True)
        height, width = frame_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(save_root, current_folder, 'frames.mp4'), fourcc, data['FPS'],
                              (width, height))
        for i in range(len(frame_list)):
            out.write(frame_list[i])
        out.release()
        np.save(os.path.join(save_root, current_folder, 'landmarks.npy'), np.stack(landmark_list))

    def empty_preprocess_cache():
        nonlocal current_folder, frame_list, landmark_list
        current_folder = f'{frame_idx + 1:04d}_{frame_idx + num_frames_per_segment + 1:04d}'
        frame_list.clear()
        landmark_list.clear()

    # ----- Start of function video_and_audio_segmentation() -----
    save_root = data['video_path'][:-4]
    if os.path.exists(save_root):   # If previous result exists, skip preprocessing of this data.
        return

    num_frames_per_segment = int(data['FPS'] * data['duration'])
    clip_cnt = 0
    current_folder = f'{0:04d}_{num_frames_per_segment:04d}'
    frame_list, landmark_list = [], []
    face_detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(os.path.join(dlib_tool_path, 'mmod_human_face_detector.dat'))
    face_predictor = dlib.shape_predictor(os.path.join(dlib_tool_path, 'shape_predictor_68_face_landmarks.dat'))

    # Open video stream file.
    cap = cv2.VideoCapture(data['video_path'])
    if not cap.isOpened():
        print(f'Failed to open {data["video_path"]}')
        return

    # If partially manipulated, only consider fake segments.
    if data.get('fake_segments', None) is None:
        target_frames = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    else:
        target_frames = []
        for start_time, end_time in data['fake_segments']:
            start_frame = math.ceil(start_time * data['FPS'])
            end_frame = math.floor(end_time * data['FPS']) + 1
            if end_frame - start_frame < num_frames_per_segment:
                start_frame = max(end_frame - num_frames_per_segment, 0)
                end_frame = min(start_frame + num_frames_per_segment, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            target_frames.extend(list(range(start_frame, end_frame)))
        target_frames = list(set(target_frames))

    # Remove first two frames to avoid leading silence.
    # target_frames = list(target_frames)
    target_frames = [x for x in target_frames if x not in range(2)]
    # for i in range(2):
        # target_frames.remove(i)

    # Start to go through video to detect face(s) every frame.
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Skip those frames that aren't in 'target_frames', e.g., first 2 frames or those frames that don't contain any
        # fake segment.
        if frame_idx not in target_frames:
            empty_preprocess_cache()
            continue

        # Failure when reading frame.
        ret, frame = cap.read()
        if not ret:
            print(f'Failed to open {frame_idx}-th frame of {data["video_path"]}')
            empty_preprocess_cache()
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces) == 0:
            faces = cnn_detector(gray)
            faces = [d.rect for d in faces]

        # If no face detected or more than 1 face detected, discard previous results.
        # Todo: Multiple face case. How to match voice and speaker? How to track face? How to decide labels for each face?
        if len(faces) != 1:
            print(f'{len(faces)} faces detected in {frame_idx}-th frame of {data["video_path"]}')
            empty_preprocess_cache()
            continue

        landmarks = face_predictor(gray, faces[0])
        landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        frame_list.append(frame)
        landmark_list.append(landmarks)
        if len(frame_list) == num_frames_per_segment:
            save_segment()
            empty_preprocess_cache()
            clip_cnt += 1
            if clip_cnt == data['clip_amount']:
                break

    cap.release()
    os.remove(data['video_path'])
    if clip_cnt == 0:
        os.remove(data['audio_path'])


def merge_audio_and_video_multiprocess(data_list, num_proc):
    """
    Merge video and audio for visualization using multi-process.
    Args:
        data_list: [list] each element is a dictionary containing attributes of this data.
        num_proc: [int] number of processors
    """
    with multiprocessing.Pool(processes=num_proc) as pool:
        with tqdm(total=len(data_list)) as pbar:
            for _ in pool.imap_unordered(merge_audio_and_video, data_list):
                pbar.update(1)


def merge_audio_and_video(data):
    """
    Merge video and audio for visualization
    Args:
        data: [dict] data dictionary with file path stored.
    """
    output_dir = os.path.dirname(data['audio_video_path'])
    temp_dir = os.path.join(output_dir, 'temp')
    if os.path.exists(data['audio_video_path']):
        return

    frames = np.load(data['video'])['frames']
    if len(frames) != int(data['FPS'] * data['duration']):
        raise AssertionError(
            f'Incorrect duration, got {len(frames)} frames, expect {int(data["FPS"] * data["duration"])} frames.')

    audio, sr = librosa.load(data['audio'], sr=None)
    if (len(audio) / sr - data['duration']) >= 0.01:
        raise AssertionError(f'Incorrect duration, got {len(audio) / sr:.2f}s, expect {data["duration"]}s')

    try:
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, data['FPS'], (frames.shape[2], frames.shape[1]))
        for frame in frames:
            out.write(frame)
        out.release()

        cmd = ['ffmpeg', '-y', '-i', temp_video_path, '-i', data['audio'], '-c:v', 'copy', '-c:a', 'aac', '-strict',
               'experimental', '-shortest', data['audio_video_path']]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    finally:
        shutil.rmtree(temp_dir)


def load_data(path):
    frame_list = []
    cap = cv2.VideoCapture(os.path.join(path, 'frames.mp4'))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)
    cap.release()
    frame_list = np.stack(frame_list, axis=0)
    landmarks = np.load(os.path.join(path, 'landmarks.npy'))
    audio, sr = librosa.load(os.path.dirname(path) + '.wav', sr=None)
    return {'frames': frame_list, 'audio': audio, 'landmarks': landmarks}


def load_data_multiprocess(path_list, num_proc):
    data_list = []
    if num_proc != 1:
        with multiprocessing.Pool(processes=num_proc) as pool:
            data_list = pool.map(load_data, path_list)
    else:
        for path in path_list:
            data_list.append(load_data(path))

    return data_list