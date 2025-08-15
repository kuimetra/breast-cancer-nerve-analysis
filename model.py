from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    """DoubleConv block: (Conv -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock -> MaxPool"""

    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        skip_connection = x  # Save skip connections (before pooling)
        x = self.max_pool(x)
        return x, skip_connection


class DecoderBlock(nn.Module):
    """Decoder block: UpConv -> ConvBlock"""

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip_features):
        x = self.up_conv(x)
        # Concatenate upsampled features with the skip connection along the channel dimension
        x = torch.cat((x, skip_features), dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=38, out_channels=1):
        super(UNet, self).__init__()
        # Contracting path
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck layers
        self.bottleneck_conv_block = DoubleConv(512, 1024)

        # Expanding path
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        # Output layer (1x1 convolution to map each feature vector to out_channels number of classes)
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        enc1_out, skip1 = self.encoder1(x)
        enc2_out, skip2 = self.encoder2(enc1_out)
        enc3_out, skip3 = self.encoder3(enc2_out)
        enc4_out, skip4 = self.encoder4(enc3_out)

        # Bottleneck
        bottleneck_out = self.bottleneck_conv_block(enc4_out)

        # Expanding path
        dec1_out = self.decoder1(bottleneck_out, skip4)
        dec2_out = self.decoder2(dec1_out, skip3)
        dec3_out = self.decoder3(dec2_out, skip2)
        dec4_out = self.decoder4(dec3_out, skip1)

        out = self.output_conv(dec4_out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)

    summary(model, input_size=(1, 848, 848))
