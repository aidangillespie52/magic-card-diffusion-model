import torch
import torch.nn
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# Residual Block Definition
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If the input and output channels differ, apply a 1x1 convolution in the skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out

# Encoder with ResNet blocks
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_size=(50, 50)):
        super(Encoder, self).__init__()
        self.initial = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(64, 128, stride=2)  # Downsample by 2
        self.res2 = ResidualBlock(128, 256, stride=2)  # Downsample by 2
        self.res3 = ResidualBlock(256, 512, stride=2)  # Downsample

        # Update adaptive pooling to a custom latent size (e.g., 8x8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(latent_size)

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.adaptive_pool(x)  # Apply the new latent size
        return x

class Decoder(nn.Module):
    def __init__(self, output_channels, latent_size=(50, 50)):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)

        # Calculate the final output size to match the input size
        self.latent_size = latent_size

    def forward(self, x, input_size):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # Use interpolation if necessary to match the input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        x = self.final(x)
        return x

# Autoencoder that combines the Encoder and Decoder
class ResNetAutoencoder(nn.Module):
    def __init__(self, in_channels=3, output_channels=3, latent_size=(50, 50)):
        super(ResNetAutoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_size)
        self.decoder = Decoder(output_channels, latent_size)

    def forward(self, x):
        input_size = x.size()[2:]
        latent = self.encoder(x)
        reconstructed = self.decoder(latent, input_size)
        return reconstructed

    def test(self, img):
        img = img.to(device)

        pred = self.forward(img.unsqueeze(0))[0].to(device)
        
        pred_img = pred.permute(1, 2, 0).detach().cpu().numpy()
        
        pred_img = np.clip(pred_img, 0, 1)

        act_img = img.permute(1, 2, 0).cpu().numpy()
        
        _, axes = plt.subplots(1, 2, figsize=(7, 7))

        axes[0].imshow(act_img)
        axes[0].set_title('Actual Image')
        axes[0].axis('off')

        axes[1].imshow(pred_img)
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')

        # Adjust layout to avoid overlap
        plt.tight_layout()

        plt.show()

        return pred_img