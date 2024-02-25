import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Union, Dict
import importlib

import cv2
import numpy as np
import ultralytics.cfg
from ultralytics.cfg import cfg2dict, check_dict_alignment, CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS, \
    CFG_BOOL_KEYS
from ultralytics.utils import LOGGER, IterableSimpleNamespace, DEFAULT_CFG_DICT


CFG_ASYMMETRICAL_FLOAT_KEYS = ['degrees', 'shear', 'translate', 'scale', 'perspective']


# Override the type checks
def get_cfg_modified(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data.
        overrides (str | Dict | optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """

    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        if 'save_dir' not in cfg:
            overrides.pop('save_dir', None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':  # assign model to 'name' arg
        cfg['name'] = cfg.get('model', '').split('.')[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_ASYMMETRICAL_FLOAT_KEYS:
                if not isinstance(v, (int, float, tuple)):
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                    f"Valid '{k}' types are int (i.e. '{k}=0'), float (i.e. '{k}=0.5'), or tuple (i.e. '{k}=(-0.3, 0.5)')")
                if isinstance(v, tuple) and (len(v) != 2 or not isinstance(v[0], (int, float)) or not isinstance(v[1], (int, float))):
                    raise ValueError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                     f"Valid '{k}' values of type tuple have to be of tuple[int | float, int | float]")
            elif k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. "
                                     f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be an int (i.e. '{k}=8')")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")
    # Return instance
    return IterableSimpleNamespace(**cfg)


ultralytics.cfg.get_cfg = get_cfg_modified


import ultralytics.data.augment


# Override the transform to allow for asymmetrical transforming
# e.g. allows to set rotation angle range to (d1, d2) instead of only (-d1, d1)
class RandomPerspectiveAsymmetrical(ultralytics.data.augment.RandomPerspective):

    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 border=(0, 0),
                 pre_transform=None):

        self.degrees = degrees
        if isinstance(self.degrees, (int, float)):
            self.degrees = (-self.degrees, self.degrees)

        self.translate = translate
        if isinstance(self.translate, (int, float)):
            self.translate = (-self.translate, self.translate)

        self.scale = scale
        if isinstance(self.scale, (int, float)):
            self.scale = (-self.scale, self.scale)

        self.shear = shear
        if isinstance(self.shear, (int, float)):
            self.shear = (-self.shear, self.shear)

        self.perspective = perspective
        if isinstance(self.perspective, (int, float)):
            self.perspective = (-self.perspective, self.perspective)

        # Mosaic border
        self.border = border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """Center."""
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(self.perspective[0], self.perspective[1])  # x perspective (about y)
        P[2, 1] = random.uniform(self.perspective[0], self.perspective[1])  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(self.degrees[0], self.degrees[1])
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 + self.scale[0], 1 + self.scale[1])
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(self.shear[0], self.shear[1]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(self.shear[0], self.shear[1]) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 + self.translate[0], 0.5 + self.translate[1]) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 + self.translate[0], 0.5 + self.translate[1]) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s


ultralytics.data.augment.RandomPerspective = RandomPerspectiveAsymmetrical

# so that they use the updated version of get_cfg
importlib.reload(ultralytics.engine.trainer)
importlib.reload(ultralytics.engine.validator)

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    # train the model
    print("Training on target images...")
    results = model.train(data="./datasets/detector/data.yaml", project="./", name="detector_models", exist_ok=True, epochs=50, imgsz=640, hsv_h=0.04, hsv_s=0.7, hsv_v=0.4, scale=0.5, translate=0.1, degrees=15, perspective=0.001, mosaic=1, flipud=0, fliplr=0)
    print("Adding cut out and dummy images...")
    results = model.train(data="./datasets/detector/data_with_additions.yaml", project="./", name="detector_models", exist_ok=True, epochs=25, imgsz=640, hsv_h=0.04, hsv_s=0.7, hsv_v=0.4, scale=(-0.5, 0.05), translate=0.1, degrees=15, perspective=0.001, mosaic=0, flipud=0, fliplr=0)
