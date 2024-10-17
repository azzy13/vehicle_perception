import json
import os

base_dir = "/isis/home/hasana3/ByteTrack/datasets/dataset/annotations"
path = os.path.join(base_dir, "train.json")

with open(path, 'r') as f:
    data = json.load(f)

for annotation in data['annotations']:
    annotation['is_crowd'] = 0
    annotation['segmentation'] = []

with open(path, 'w') as f:
    json.dump(data, f, indent=4)