import torch
from torch import nn
import torchinfo
import torchsummary

class UNet(nn.Module):
    def __init__(self, in_channels, n_features, num_classes):
        super().__init__()
        self.conv1 = self.conv_bn_relu(in_channels, n_features, 3)           # 224
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = self.conv_bn_relu(n_features, n_features * 2, 3)        # 112
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = self.conv_bn_relu(n_features * 2, n_features * 4, 3)    #  56
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = self.conv_bn_relu(n_features * 4, n_features * 8, 3)    #  28
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = self.conv_bn_relu(n_features * 8, n_features * 8, 3)    #  14
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.conv6 = self.conv_bn_relu(n_features * 8, n_features * 8, 3)    #   7
        
        self.conv06 = self.conv_bn_relu(n_features * 8, n_features * 8, 3)    #   7
        self.upsample6 = nn.Upsample(scale_factor=2)
        self.conv05 = self.conv_bn_relu(n_features * 16, n_features * 8, 3)    #  14
        self.upsample5 = nn.Upsample(scale_factor=2)
        self.conv04 = self.conv_bn_relu(n_features * 16, n_features * 8, 3)    #  28
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.conv03 = self.conv_bn_relu(n_features * 12, n_features * 2, 3)    #  56
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv02 = self.conv_bn_relu(n_features * 4, n_features * 1, 3)        # 112
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv01 = self.conv_bn_relu(n_features, n_features, 3)           # 224
        self.upsample1 = nn.Upsample(scale_factor=2)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features, kernel_size=3, padding='same'),
            nn.Conv2d(n_features, num_classes, kernel_size=1, padding='same'),
        )
        
    
    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)
        x4 = self.conv4(p3)
        p4 = self.pool4(x4)
        x5 = self.conv5(p4)
        p5 = self.pool5(x5)
        x6 = self.conv6(p5)
        
        x06 = self.conv06(x6)           # 7
        u6 = self.upsample6(x06)        # 14
        u6 = torch.concat([u6, x5], dim=1)
        
        x05 = self.conv05(u6)
        u5 = self.upsample5(x05)        # 28
        u5 = torch.concat([u5, x4], dim=1)
        
        x04 = self.conv04(u5)
        u4 = self.upsample4(x04)        # 56
        u4 = torch.concat([u4, x3], dim=1)
        
        x03 = self.conv03(u4)
        u3 = self.upsample3(x03)        # 112
        u3 = torch.concat([u3, x2], dim=1)
        
        x02 = self.conv02(u3)
        u2 = self.upsample2(x02)        # 224
        u2 = torch.concat([u2, x1], dim=1)
        
        return self.classifier(u2)
    
    def conv_bn_relu(self, in_channels, out_channels, kernel_size, ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

if __name__ == "__main__":
    model = UNet(3, 32, 2).to('cuda')
    # torchinfo.summary(model.to('cuda'), input_size=(1, 3, 224, 224))
    torchsummary.summary(model, input_size=(3, 224, 224), batch_size=32)
    
    from torchviz import make_dot

    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).type(torch.float32).to('cuda')  # Batch size of 1, input size of 10

    # Generate the graph
    output = model(dummy_input)
    dot = make_dot(output, params=dict(list(model.named_parameters())))

    # Render the graph
    dot.render("model_architecture", format="png")