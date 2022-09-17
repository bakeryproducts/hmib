import torch
from timm.models.coat import _create_coat
from timm.models.registry import register_model


class CoatFlat(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        r = self.model(x)
        r = [r[k] for k in ['x1_nocls', 'x2_nocls', 'x3_nocls', 'x4_nocls']]
        return r


@register_model
def coat_lite_medium(pretrained=False, **kwargs):
    model_cfg = dict(
        patch_size=4, embed_dims=[128, 256, 320, 512], serial_depths=[3, 6, 10, 8], parallel_depth=0,
        num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model = _create_coat('coat_lite_medium', pretrained=pretrained, **model_cfg)
    return model
