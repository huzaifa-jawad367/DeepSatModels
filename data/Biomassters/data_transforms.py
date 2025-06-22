# data_transforms.py

import random
import cv2
import numpy as np
import xarray as xr
from xrspatial import multispectral

def calculate_veg_indices_uint8(img_s2):
    """
    Compute NDVI, EVI, MSAVI, NDMI on the 15-band S2 array,
    normalize each to [0,1], and return as dict of H×W arrays.
    """
    img = xr.DataArray(img_s2.astype("float32"))

    ndvi = np.array(multispectral.ndvi(img[..., 6], img[..., 2]))
    evi  = np.array(multispectral.evi (img[..., 6], img[..., 2], img[..., 0]))
    # MSAVI
    b6 = img_s2[..., 6]; b2 = img_s2[..., 2]
    msavi = 0.5 * (2*b6 + 1 - np.sqrt((2*b6 + 1)**2 - 8*(b6 - b2) + epsilon))
    ndmi = np.array(multispectral.ndmi(img[..., 6], img[..., 7]))

    def norm(a):
        return ((a + 1) / 2).clip(0,1)

    return {
        "ndvi": norm(ndvi),
        "evi" : norm(evi),
        "msavi": norm(msavi),
        "ndmi": norm(ndmi),
    }


def rotate_image(image, angle, rot_pnt, scale=1):
    """
    Rotate H×W×C or H×W array around rot_pnt by angle (deg),
    reflecting at borders.
    """
    M = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    return cv2.warpAffine(
        image, M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


def train_aug(imgs, mask, target):
    """
    imgs: [T, C, H, W], mask: [T], target: [H, W]
    does random flips, 90-rotations, small rotate+scale, and “word” dropout.
    """
    # horizontal flip
    if random.random() > 0.5:
        imgs   = imgs[..., ::-1]
        target = target[..., ::-1]

    # 90° multiples
    k = random.randrange(4)
    if k:
        imgs   = np.rot90(imgs, k, axes=(-2, -1))
        target = np.rot90(target, k, axes=(-2, -1))

    # small rotate + shift
    if random.random() > 0.3:
        d = int(imgs.shape[2] * 0.1)
        rot_pnt = (
            imgs.shape[2]//2 + random.randint(-d, d),
            imgs.shape[3]//2 + random.randint(-d, d)
        )
        angle = random.randint(-45, 45)
        if angle != 0:
            T, C, H, W = imgs.shape
            flat = imgs.transpose(1,2,3,0).reshape(H, W, T*C)
            flat = rotate_image(flat, angle, rot_pnt)
            imgs = flat.reshape(H, W, T, C).transpose(2,3,0,1)
            target = rotate_image(target, angle, rot_pnt)

    # “word” dropout on entire timesteps
    if random.random() > 0.5:
        T = len(imgs)
        while True:
            drop = np.random.rand(T) < 0.3
            if not np.all(mask | drop):
                break
        mask[drop] = True
        imgs[drop] = 0

    return imgs.copy(), mask.copy(), target.copy()
