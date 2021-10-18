import torch

CHOSEN_MODEL_FEATURES = ['0', '5', '10', '19', '28']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 128

ORIGINAL_IMAGE_NAME = "hourse_image.jpg"
STYLE_IMAGE_NAME = "vincent-van-gogh-the-starry-night.jpg"
