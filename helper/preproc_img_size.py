import cv2
import numpy as np

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    
    # Return the image and the resizing ratio
    return padded_img, r

# Testing the function
img = cv2.imread('/isis/home/hasana3/ByteTrack/datasets/dataset/train/164784.png')  # Load your image
input_size = (1920, 1080)  # Example input size
mean = (0.485, 0.456, 0.406)  # Mean normalization values
std = (0.229, 0.224, 0.225)  # Std deviation normalization values

# Preprocess the image
preprocessed_img, resize_ratio = preproc(img, input_size, mean, std)

# Print the size of the preprocessed image
print(f"Preprocessed image size: {preprocessed_img.shape}")