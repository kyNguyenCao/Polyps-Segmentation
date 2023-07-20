import streamlit as st
import torch
import numpy as np
import cv2
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2  # np.array -> torch.Tensor

IMG_SIZE = 256
DEVICE = 'cpu'

test_transform = A.Compose([
    A.Resize(width=IMG_SIZE, height=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])


def unet_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.backbone = timm.create_model('resnet152', pretrained=True, features_only=True)
        self.block_neck = unet_block(2048, 1024)
        self.block_up1 = unet_block(1024 + 1024, 512)
        self.block_up2 = unet_block(512 + 512, 256)
        self.block_up3 = unet_block(256 + 256, 128)
        self.block_up4 = unet_block(128 + 64, 64)
        self.conv_cls = nn.Conv2d(64, self.n_classes, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.backbone(x)
        x = self.block_neck(x5)
        x = torch.cat([x4, self.upsample(x)], dim=1)
        x = self.block_up1(x)
        x = torch.cat([x3, self.upsample(x)], dim=1)
        x = self.block_up2(x)
        x = torch.cat([x2, self.upsample(x)], dim=1)
        x = self.block_up3(x)
        x = torch.cat([x1, self.upsample(x)], dim=1)
        x = self.block_up4(x)
        x = self.conv_cls(x)
        x = self.upsample(x)

        return x


def main():
    st.title('Image Segmentation App')
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        model = UNet(1).to(DEVICE)
        checkpoint = torch.load("weights/custom_unet_with_resnet152_backboned.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        file_bytes = uploaded_image.read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        image_byte = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_byte, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            trans_image = test_transform(image=image_rgb)['image']
            image_fed = trans_image.to(DEVICE).float().unsqueeze(0)
            mask = model(image_fed).squeeze().sigmoid().round().long().numpy()
            mask = mask * 255
            st.image(mask, caption='Uploaded Image', use_column_width=True)


if __name__ == '__main__':
    main()
