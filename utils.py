import cv2
import numpy as np

def sample_clips(video_path, start_sec, duration_sec, fps=5):
    cap = cv2.VideoCapture(video_path)
    clips = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * video_fps)
    num_sample = int(duration_sec * fps)
    for i in range(num_sample):
        frame_id = start_frame + int(i * (video_fps / fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret: break
        clips.append(frame)
    cap.release()
    return clips