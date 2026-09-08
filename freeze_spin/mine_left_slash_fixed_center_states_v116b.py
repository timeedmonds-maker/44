from __future__ import annotations

"""v116b: mechanical fix for v116 homography-diversity projection.

All acquisition, matching and gates are inherited unchanged from v116. The only
change is to project the four image corners and image centre as separate dense
OpenCV arrays instead of one invalid ragged NumPy array.
"""

import math
import cv2
import numpy as np

from freeze_spin import mine_left_slash_fixed_center_states_v116 as base


def homography_diversity_fixed(Hm):
    corners = np.float32([[0, 0], [base.W - 1, 0], [base.W - 1, base.H - 1], [0, base.H - 1]]).reshape(-1, 1, 2)
    centre = np.float32([[base.W / 2, base.H / 2]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, Hm).reshape(-1, 2)
    centre_q = cv2.perspectiveTransform(centre, Hm).reshape(-1, 2)[0]
    area = abs(cv2.contourArea(quad.astype(np.float32))) / float((base.W - 1) * (base.H - 1))
    shift = float(np.linalg.norm(centre_q - np.array([base.W / 2, base.H / 2], np.float32)))
    zoom = math.sqrt(max(area, 1e-9))
    diversity = abs(math.log(max(zoom, 1e-9))) + shift / 300.0
    return {
        'projected_frame_area_ratio': float(area),
        'approx_linear_zoom_ratio': float(zoom),
        'projected_center_shift_px': shift,
        'diversity_index': float(diversity),
        'projected_corners': quad.tolist(),
    }


base.homography_diversity = homography_diversity_fixed

if __name__ == '__main__':
    base.main()
