import torch
import constants
import model
import image_functionality
import train

original_image = image_functionality.load_image(constants.ORIGINAL_IMAGE_NAME)
style_image = image_functionality.load_image(constants.STYLE_IMAGE_NAME)

model = model.VGG().to(constants.DEVICE).eval()
generated_image = original_image.clone().requires_grad_(True)
#generated_image = torch.randn(original_image.shape, device=constants.DEVICE, requires_grad=True)
generated_image = train.train(model, original_image, style_image, generated_image, "generated.png")
