import torch
import argparse
from utils import sample_clips
from detectors.ball_detection import BallDetector
from detectors.number_extraction import NumberExtractor
from detectors.foot_keypoint_extraction import FootKeypointExtractor
from event_extraction.train import EventModel
import os


def predict(args):
    # load event model
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"[INFO] 모델 경로 {model_path}가 존재하지 않습니다. pretrained 모델을 사용합니다.")
        model_path = os.path.join('pretrained', 'event_model_soccernet.pt')

    model = EventModel(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device).eval()
    video = args.video_path

    # slide through video in 5s windows
    events = []
    for t in range(0, int(args.video_length), 5):
        clips = sample_clips(video, t, 5)
        # preprocess and predict
        # TODO: implement batch inference
        logits = model(torch.stack(clips).unsqueeze(0).to(args.device))
        cls = torch.argmax(logits, dim=1).item()
        if cls != 2:  # not 'none'
            events.append((t, cls))

    # for each event, find subject
    ball = BallDetector(args.ball_model)
    number = NumberExtractor()
    pose = FootKeypointExtractor()
    results = []
    for t, cls in events:
        frames = sample_clips(video, max(0, t-3), 3)
        bpos = [ball.detect(f) for f in frames]
        ppos = [pose.extract(f) for f in frames]
        # compute average distances
        best = (None, float('inf'))
        for person_idx in range(len(ppos)):
            d = sum(((bp[0]-fp[0])**2+(bp[1]-fp[1])**2)**0.5 for bp, fp in zip(bpos, ppos) if bp and fp)
            if d < best[1]: best = (person_idx, d)
        # crop that person's bbox and extract number
        # TODO: implement bbox tracking
        num = None
        results.append({'time': t, 'event': cls, 'player': num})
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, default='pretrained/event_model_pretrained.pt')
    parser.add_argument('--ball-model', type=str, default='yolov8_ball.pt')
    parser.add_argument('--video-length', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    predict(args)
