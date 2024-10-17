import json
import os

base_dir = "/isis/home/hasana3/ByteTrack/datasets/dataset/annotations"
path = os.path.join(base_dir, "valid.json")

with open(path, 'r') as f:
    data = json.load(f)

valid_annotations = []
for annotation in data['annotations']:
    bbox = annotation['bbox']
    if all(x >= 0 and len(str(x)) < 5 for x in bbox):
        valid_annotations.append(annotation)

data['annotations'] = valid_annotations

with open(path, 'w') as f:
    json.dump(data, f, indent=4)