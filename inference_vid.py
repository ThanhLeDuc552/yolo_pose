from ultralytics import YOLO
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="Path to the model or just the name")
parser.add_argument("--video", type=str, required=True, help="Path to the video")

args = parser.parse_args()

model = YOLO(args.model)

results = model.predict(args.video, save=True)

print("Inference completed. Result saved in the same workspace")