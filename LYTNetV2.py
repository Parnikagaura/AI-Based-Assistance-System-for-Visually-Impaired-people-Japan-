import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True),
        nn.MaxPool2d(2,2),
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self, in_channel, skip_channel, out_channel):
        super(Decoder, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channel+skip_channel, out_channel, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True), 
            
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True), 
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        cat = torch.cat([x, skip], dim=1)
        out = self.sequence(cat)
        
        return out

class Head(nn.Module):
    """Some Information about Head"""
    def __init__(self, in_channel, n_class):
        super(Head, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, n_class, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, original_size):
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        x = self.sequence(x)

        return x

class LYTNetV2(nn.Module):
    def __init__(self, n_class=5, input_size=768, width_mult=1.0):
        super(LYTNetV2, self).__init__()
        input_channel = 16
        last_channel = 1280

        # refer to Table 1 in paper
        mobile_setting = [
            # k, exp, c,  se,     nl,  s,
            [3, 16,  16,  False, 'RE', 1],
            [3, 64,  24,  False, 'RE', 2],
            [3, 72,  24,  False, 'RE', 1],
            [5, 72,  40,  True,  'RE', 2],
            [5, 120, 40,  True,  'RE', 1],
            [3, 240, 80,  False, 'HS', 2],
            [3, 200, 80,  False, 'HS', 1],
            [3, 480, 112, True,  'HS', 1],
            [5, 672, 160, True,  'HS', 2],
            [5, 960, 160, True,  'HS', 1],
            [3, 960, 320, False,  'RE', 1]
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 16!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]

        # building mobile blocks
        io_list = []
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            io_list.append(output_channel)
            input_channel = output_channel

        print(f"Io list : {io_list}")

        # building last several layers
        last_conv = make_divisible(960 * width_mult)
        self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        # self.features.append(nn.AvgPool2d(12,9)) Original Hardcoded
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features.append(Hswish(inplace=True))
        #self.features.append(nn.Dropout(0.1))
        self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        self.features.append(Hswish(inplace=True))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features, nn.Dropout(0.1))
        

        self.light_classifier = nn.Sequential(
            nn.Linear(self.last_channel, 5),
            nn.Softmax(dim=1)
        )
        
        self.direction_regression = nn.Sequential(
            nn.Linear(self.last_channel, 4),
        )
        
        self.decoder1 = Decoder(in_channel=io_list[10], skip_channel=io_list[7], out_channel=io_list[7])
        self.decoder2 = Decoder(in_channel=io_list[7], skip_channel=io_list[4], out_channel=io_list[4])
        self.decoder3 = Decoder(in_channel=io_list[4], skip_channel=io_list[2], out_channel=io_list[2])
        self.decoder4 = Decoder(in_channel=io_list[2], skip_channel=io_list[0], out_channel=io_list[0])
        self.final = Head(in_channel=io_list[0], n_class=1)
        
        self._initialize_weights()

    def forward(self, img):
        layers = [img]
        for i, layer in enumerate(self.features):
            pred = layer(layers[-1])
            layers.append(pred)
        
        for i, data in enumerate(layers):
            print(f"Layer idx {i} \t : {data.shape}")
        # Index
        # Layer idx 0 	 : torch.Size([1, 3, 480, 480])
        # Layer idx 2 	 : torch.Size([1, 16, 120, 120])
        # Layer idx 4 	 : torch.Size([1, 24, 60, 60])
        # Layer idx 6 	 : torch.Size([1, 40, 30, 30])
        # Layer idx 9 	 : torch.Size([1, 112, 15, 15])
        # Layer idx 12 	 : torch.Size([1, 320, 8, 8])
        
        decoder1 = self.decoder1(layers[12], layers[9]) 
        decoder2 = self.decoder2(decoder1, layers[6])
        decoder3 = self.decoder3(decoder2, layers[4])
        decoder4 = self.decoder4(decoder3, layers[2])
        head = self.final(decoder4, img.shape[2:])
        
        print(f"Decoder1 : {decoder1.shape}")
        print(f"Decoder2 : {decoder2.shape}")
        print(f"Decoder3 : {decoder3.shape}")
        print(f"Decoder4 : {decoder4.shape}")
        print(f"Head : {head.shape}")
        
        x = layers[-1].view(layers[-1].size(0),-1)
        x1 = self.light_classifier(x)
        x2 = self.direction_regression(x)
        
        print(f"Head: {head.shape}, X1 : {x1.shape}, X2 : {x2.shape}")
        
        return head, x1, x2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

if __name__ == '__main__':
    model = LYTNetV2(n_class=5, input_size=256, width_mult=1.0)
    sample = torch.randn(1, 3, 256, 512)  # Example input tensor
    model(sample)