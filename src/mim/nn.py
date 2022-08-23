
class SimMIM(nn.Module):
    def __init__(self, in_chans, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                # in_channels=self.encoder.num_features[-1],
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        features = self.encoder(x, mask)
        x_rec = self.decoder(features[-1])
        return x_rec


class SSL(nn.Module):
    def __init__(self, cfg, encoder_cfg, decoder_cfg, seg_cfg, stride=32):
        super().__init__()
        encoder_cfg.pop('model_name')
        # self.s = SwinTransformerForSimMIM(**encoder_cfg)
        self.s = SwinTransformerForSimMIM(img_size=192,
                                          win_size=6,
                                          embed_dim=128,
                                          depths=[2, 2, 18, 2],
                                          num_heads=[4,8,16,32],
                                          **encoder_cfg)
        # self.s = VisionTransformerForSimMIM(norm_layer=partial(nn.LayerNorm, eps=1e-6), **encoder_cfg)
        self.sm = SimMIM(3, self.s, stride)
        # self.p = torch.nn.Parameter(torch.ones(1))

    def forward(self, batch):
        x = batch['xb']
        mask = batch['mask']
        # print(mask.shape, x.shape)

        r = self.sm(x, mask) # B,C,H,W
        # r = self.p * x

        cls = torch.zeros(1).cuda()
        return dict(yb=r, cls=cls)
