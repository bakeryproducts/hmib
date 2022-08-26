import numpy as np
from rasterio.features import rasterize
from shapely.geometry import Polygon


def rle_decode(rle_str, mask_shape, mask_dtype=np.uint8):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


def convex2mask(convex_str, mask_shape, mask_dtype=np.uint8):
    convex_floats = list(map(float, convex_str.split()))

    points = []
    for xi in range(0, len(convex_floats), 2):
        points.append(convex_floats[xi: xi + 2])

    polygon = Polygon(points)
    mask = rasterize([polygon], out_shape=mask_shape)

    return mask


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """ TBD

    Args:
        img (np.array):
            - 1 indicating mask
            - 0 indicating background

    Returns:
        run length as string formated
    """

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
