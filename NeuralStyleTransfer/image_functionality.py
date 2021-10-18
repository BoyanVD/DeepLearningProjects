from PIL import Image
import torchvision.transforms as transforms
import constants

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(constants.DEVICE)

loader = transforms.Compose(
    [
        transforms.Resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)
