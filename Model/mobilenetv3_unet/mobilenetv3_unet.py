import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.Blockconv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.Blockconv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.Blockconv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.Blockconv1(x)))
        out = self.nolinear2(self.bn2(self.Blockconv2(out)))
        out = self.bn3(self.Blockconv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


# class MobileNetV3_Large(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(MobileNetV3_Large, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.hs1 = hswish()
#
#         self.bneck = nn.Sequential(
#             Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
#             Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
#             Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
#             Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
#             Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
#             Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
#             Block(3, 40, 240, 80, hswish(), None, 2),
#             Block(3, 80, 200, 80, hswish(), None, 1),
#             Block(3, 80, 184, 80, hswish(), None, 1),
#             Block(3, 80, 184, 80, hswish(), None, 1),
#             Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
#             Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
#             Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
#             Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
#             Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
#         )
#
#
#         self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(960)
#         self.hs2 = hswish()
#         self.linear3 = nn.Linear(960, 1280)
#         self.bn3 = nn.BatchNorm1d(1280)
#         self.hs3 = hswish()
#         self.linear4 = nn.Linear(1280, num_classes)
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         out = self.hs1(x)
#         out = self.bneck(out)
#         out = self.hs2(self.bn2(self.conv2(out)))
#         out = F.avg_pool2d(out, 7)
#         out = out.view(out.size(0), -1)
#         out = self.hs3(self.bn3(self.linear3(out)))
#         out = self.linear4(out)
#         return out
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)



class MobileNetV3_UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(MobileNetV3_UNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40,  hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()
        self.center_conv = DoubleConv(576, 96)
        self.up_1 = nn.Conv2d(576, 96, 3, 1, 1)
        self.right_conv_1 = DoubleConv(192, 96)

        self.up_2 = nn.ConvTranspose2d(96, 48, 2, 2)
        self.right_conv_2 = DoubleConv(136, 48)

        self.up_3 = nn.ConvTranspose2d(48, 24, 2, 2)
        self.right_conv_3 = DoubleConv(48, 24)

        self.up_4 = nn.ConvTranspose2d(24, 16, 2, 2)
        self.right_conv_4 = DoubleConv(32, 16)

        self.up_5 = nn.ConvTranspose2d(16, 16, 2, 2)
        self.right_conv_5 = DoubleConv(32, 16)

        # output
        self.output = nn.ConvTranspose2d(16, 1, 2, 2)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = []
        x = self.hs1(self.bn1(self.conv1(x))) #16 256 256
        x0 = x
        for m in self.bneck:
            x = m(x)
            y.append(x)
        # x2 = self.bneck(x1) #96 16 16
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = y
        x12 = self.hs2(self.bn2(self.conv2(x11))) #576 16 16
        # out = F.avg_pool2d(x12, 7)
        # out = out.view(out.size(0), -1)
        # out = self.hs3(self.bn3(self.linear3(out)))
        # out = self.linear4(out)
        y1 = self.center_conv(x12)

        temp = torch.cat((y1,x11), dim=1)
        y6 = self.right_conv_1(temp)

        y5_up = self.up_2(y6)
        temp = torch.cat((y5_up, x8,x6), dim=1)
        y5 = self.right_conv_2(temp)

        y4_up = self.up_3(y5)
        temp = torch.cat((y4_up, x3), dim=1)
        y4 = self.right_conv_3(temp)

        y3_up = self.up_4(y4)
        temp = torch.cat((y3_up, x1), dim=1)
        y3 = self.right_conv_4(temp)

        y2_up = self.up_5(y3)
        temp = torch.cat((y2_up, x0), dim=1)
        y2 = self.right_conv_5(temp)
        y1 = self.output(y2)
        #out = nn.Sigmoid()(y1)
        return y1

if __name__ == "__main__":
    model = MobileNetV3_UNet(in_channels=3, num_classes=1)
    x = torch.randn(10, 3, 256, 256)
    y = model(x)
    print("Output:", y.shape)