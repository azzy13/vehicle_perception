# yolox/models/color_classifier.py
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image

class ColorClassifier:
    def __init__(self, device='cuda'):
        self.device = device
        # Just example classes (ImageNet); these won't reflect car colors accurately
        self.classes = [f'class_{i}' for i in range(1000)]  

        # Load pretrained ResNet-18 from torchvision (no .pth file needed!)
        self.model = models.resnet18(pretrained=True)
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

        # Return just class index and confidence (dummy labels for test)
        return self.classes[pred_class.item()], confidence.item()
