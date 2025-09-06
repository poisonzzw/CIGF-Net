
from Convnextv2.convnextv2 import ConvNeXtV2

import torch
import torch.nn.functional as F
import torch.nn as nn

from Convnextv2.module import FeatureRectify as FR
from Convnextv2.module import fuse6 as fus

class Convnextv2(nn.Module):
    def __init__(self, in_chans=3, num_classes=9, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs):
        super().__init__()
        self.num_stages = 4
        self.dec_outChannels=768

        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(self.dec_outChannels, num_classes, kernel_size=1))
        self.encoder_rgb = ConvNeXtV2(in_chans=in_chans,depths=depths, dims=dims, pretrained = True, **kwargs)
        self.encoder_thermal = ConvNeXtV2(in_chans=in_chans,depths=depths, dims=dims, pretrained = True, **kwargs)

        # Cross-modality Interaction Module (CIM)
        self.FR = nn.ModuleList([
            FR(dim=dims[0], reduction=1),
            FR(dim=dims[1], reduction=1),
            FR(dim=dims[2], reduction=1),
            FR(dim=dims[3], reduction=1)])

        #  Global-feature Fusion Module (GFM)
        self.fuse = nn.ModuleList([
            fus(dim=dims[0]),
            fus(dim=dims[1]),
            fus(dim=dims[2]),
            fus(dim=dims[3])
        ])

        from Convnextv2.MLPDecoder import DecoderHead
        self.decoder = DecoderHead(in_channels=[96, 192, 384, 768], num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                   embed_dim=512)

    def forward(self, rgb, thermal):
        raw_rgb = rgb

        enc_rgb = self.encoder_rgb(rgb)
        enc_thermal = self.encoder_thermal(thermal)
        enc_feats = []
        for i in range(self.num_stages):
            vi, ir = self.FR[i](enc_rgb[i], enc_thermal[i])
            x_fused = self.fuse[i](vi, ir)
            enc_feats.append(x_fused)

        dec_out = self.decoder(enc_feats)
        output = F.interpolate(dec_out, size=raw_rgb.size()[-2:], mode='bilinear',
                               align_corners=True)
        return output

if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 3, 224, 224)
    model = ConvNeXtV2()
    out = model(input, y)
    # print(model)
    print(out.size())
