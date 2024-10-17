#script adds a field id for each object in the dataset

import json,os

base_dir = "/isis/home/hasana3/ByteTrack/datasets/dataset/annotations"
path = os.path.join(base_dir, "valid.json")

with open(path, 'r') as f:
    data = json.load(f)

id = 1
for annotation in data['annotations']:
    annotation['id'] = id
    id += 1

with open(path, 'w') as f:
    json.dump(data, f, indent=4)