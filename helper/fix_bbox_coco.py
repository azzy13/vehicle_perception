import json

def convert_bbox_to_coco_format(annotations):
    for annotation in annotations:
        # Extract the old bbox format [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = annotation["bbox"]
        
        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min
        
        # Update bbox to the new format [x_min, y_min, width, height]
        annotation["bbox"] = [x_min, y_min, width, height]
        
        # Update the area
        annotation["area"] = width * height

    return annotations

def update_annotations(json_file):
    # Load the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Convert bounding boxes and areas
    data["annotations"] = convert_bbox_to_coco_format(data["annotations"])
    
    # Save the updated content back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

# Usage
json_file = '/isis/home/hasana3/ByteTrack/datasets/dataset/annotations/valid.json'
update_annotations(json_file)