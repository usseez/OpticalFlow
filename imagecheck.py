from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import cv2

p = Path("../Dataset/kitti_flow2015/training_original/flow_occ/000017_10.png")

arr_io = imageio.imread(p)
print("imageio:", arr_io.shape, arr_io.dtype, arr_io.min(), arr_io.max())

arr_cv = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # BGR, keeps depth
print("opencv :", None if arr_cv is None else (arr_cv.shape, arr_cv.dtype, arr_cv.min(), arr_cv.max()))
