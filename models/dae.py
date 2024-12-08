from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (488, 680) -> (488, 680)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (488, 680) -> (244, 340)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (244, 340) -> (122, 170)
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # (122, 170) -> (244, 340)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (244, 340) -> (488, 680)
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # (488, 680) -> (488, 680)
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
        axes[1].set_title('Enchanced Image')
        axes[1].axis('off')

        # Adjust layout to avoid overlap
        plt.tight_layout()

        plt.show()

        return pred_img