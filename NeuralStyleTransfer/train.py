import torch
import torch.optim as optim
from torchvision.utils import save_image

def gram_matrix(M):
    return M.mm(M.t())

def train(model, original_image, style_image, generated_image, generated_image_file_name, steps=5000, learning_rate=0.001, alpha=1, beta=0.01):

    #Hyperparameters
    steps = 5000
    learining_rate = 0.001
    alpha = 1
    beta = 0.01
    optimizer = optim.Adam([generated_image], lr=learining_rate)

    for step in range(steps):
        generated_image_features = model(generated_image)
        original_image_features = model(original_image)
        style_image_features = model(style_image)

        style_loss = 0
        original_loss = 0

        for generated_image_feature, original_image_feature, style_image_feature in zip(generated_image_features, original_image_features, style_image_features):
             batch_size, channel, height, width = generated_image_feature.shape
             original_loss += torch.mean((generated_image_feature - original_image_feature)**2)

             #Compute gram matrix
             G = gram_matrix(generated_image_feature.view(channel, height * width))
             A = gram_matrix(style_image_feature.view(channel, height * width))

             style_loss += torch.mean((G - A)**2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(total_loss)
            save_image(generated_image, generated_image_file_name)

    return generated_image
