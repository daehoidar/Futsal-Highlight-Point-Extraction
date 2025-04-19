import cv2
from torchvision import transforms
from PIL import Image

def sample_clips(video_path, center_time, clip_duration=10, fps=5):

    start_time = max(0, center_time - clip_duration // 2)
    end_time = center_time + clip_duration // 2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    step = int(video_fps // fps)
    frames = []

    for sec in range(start_time, end_time):
        frame_idx = int(sec * video_fps)
        for i in range(0, int(video_fps), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + i)
            ret, frame = cap.read()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                frames.append(transform(image))
            else:
                break
    cap.release()
    return frames
