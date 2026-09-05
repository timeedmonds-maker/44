from __future__ import annotations

"""Authoritative NBA court / basket geometry used by freeze-spin calibration.

Coordinate convention (centimetres):
  +X : from the near baseline toward the far baseline
  +Y : from the left sideline toward the right sideline
  +Z : upward from the playing floor

The constants below are derived from NBA Rule No. 1.  The important change
from the original prototype is that this is NBA geometry (94 ft x 50 ft), not
FIBA geometry (28 m x 15 m).
"""

import math
from dataclasses import dataclass

import numpy as np

INCH_CM = 2.54
FOOT_CM = 12.0 * INCH_CM

COURT_LENGTH_CM = 94.0 * FOOT_CM
COURT_WIDTH_CM = 50.0 * FOOT_CM
COURT_CENTER_Y_CM = COURT_WIDTH_CM / 2.0

# NBA Rule No. 1 / official court diagram.
BACKBOARD_FACE_X_NEAR_CM = 4.0 * FOOT_CM
BACKBOARD_FACE_X_FAR_CM = COURT_LENGTH_CM - BACKBOARD_FACE_X_NEAR_CM
BACKBOARD_WIDTH_CM = 6.0 * FOOT_CM
BACKBOARD_HEIGHT_CM = 3.5 * FOOT_CM
BACKBOARD_INNER_RECT_WIDTH_CM = 24.0 * INCH_CM
BACKBOARD_INNER_RECT_HEIGHT_CM = 18.0 * INCH_CM

# NBA Rule No. 1: inside diameter 18 in; nearest point of the inside edge is
# 6 in from the plane of the backboard.  Therefore ring centre is 6 + 9 =
# 15 in from the board face, i.e. 5 ft 3 in from the baseline.
RIM_INSIDE_RADIUS_CM = 9.0 * INCH_CM
RIM_NEAREST_INSIDE_EDGE_FROM_BOARD_CM = 6.0 * INCH_CM
RIM_CENTER_FROM_BOARD_CM = RIM_NEAREST_INSIDE_EDGE_FROM_BOARD_CM + RIM_INSIDE_RADIUS_CM
RIM_CENTER_X_NEAR_CM = BACKBOARD_FACE_X_NEAR_CM + RIM_CENTER_FROM_BOARD_CM
RIM_CENTER_X_FAR_CM = COURT_LENGTH_CM - RIM_CENTER_X_NEAR_CM
RIM_TOP_HEIGHT_CM = 10.0 * FOOT_CM

# The 24 x 18 in target rectangle is centered horizontally behind the ring.
# Its lower edge is level with the top of the ring; therefore it spans from
# 10 ft to 11 ft 6 in above the playing floor. This relationship is also a
# strong visual anchor in the NBA source frames.
INNER_RECT_BOTTOM_Z_CM = RIM_TOP_HEIGHT_CM
INNER_RECT_TOP_Z_CM = RIM_TOP_HEIGHT_CM + BACKBOARD_INNER_RECT_HEIGHT_CM

FREE_THROW_LINE_BOARD_DISTANCE_CM = 15.0 * FOOT_CM
PAINT_WIDTH_CM = 16.0 * FOOT_CM
RESTRICTED_ARC_RADIUS_CM = 4.0 * FOOT_CM
THREE_POINT_ARC_RADIUS_CM = (23.0 * 12.0 + 9.0) * INCH_CM
CORNER_THREE_SIDELINE_OFFSET_CM = 3.0 * FOOT_CM
LINE_WIDTH_CM = 2.0 * INCH_CM


@dataclass(frozen=True)
class BasketGeometry:
    board_x_cm: float
    rim_center_x_cm: float
    y_cm: float
    z_cm: float
    attacking_direction: int


def basket(side: str = "near") -> BasketGeometry:
    side = side.lower()
    if side == "near":
        return BasketGeometry(
            board_x_cm=BACKBOARD_FACE_X_NEAR_CM,
            rim_center_x_cm=RIM_CENTER_X_NEAR_CM,
            y_cm=COURT_CENTER_Y_CM,
            z_cm=RIM_TOP_HEIGHT_CM,
            attacking_direction=+1,
        )
    if side == "far":
        return BasketGeometry(
            board_x_cm=BACKBOARD_FACE_X_FAR_CM,
            rim_center_x_cm=RIM_CENTER_X_FAR_CM,
            y_cm=COURT_CENTER_Y_CM,
            z_cm=RIM_TOP_HEIGHT_CM,
            attacking_direction=-1,
        )
    raise ValueError("side must be 'near' or 'far'")


def inner_rectangle_corners(side: str = "near") -> dict[str, np.ndarray]:
    """Corners of the regulation 24 x 18 in white target rectangle."""
    b = basket(side)
    hw = BACKBOARD_INNER_RECT_WIDTH_CM / 2.0
    return {
        "inner_rect_top_left": np.array([b.board_x_cm, b.y_cm - hw, INNER_RECT_TOP_Z_CM], dtype=np.float64),
        "inner_rect_top_right": np.array([b.board_x_cm, b.y_cm + hw, INNER_RECT_TOP_Z_CM], dtype=np.float64),
        "inner_rect_bottom_right": np.array([b.board_x_cm, b.y_cm + hw, INNER_RECT_BOTTOM_Z_CM], dtype=np.float64),
        "inner_rect_bottom_left": np.array([b.board_x_cm, b.y_cm - hw, INNER_RECT_BOTTOM_Z_CM], dtype=np.float64),
    }


def rim_cardinal_points(side: str = "near") -> dict[str, np.ndarray]:
    """Four points on the inside edge of the regulation ring plus its centre."""
    b = basket(side)
    r = RIM_INSIDE_RADIUS_CM
    d = float(b.attacking_direction)
    return {
        "rim_center": np.array([b.rim_center_x_cm, b.y_cm, b.z_cm], dtype=np.float64),
        "rim_board_side": np.array([b.rim_center_x_cm - d * r, b.y_cm, b.z_cm], dtype=np.float64),
        "rim_court_side": np.array([b.rim_center_x_cm + d * r, b.y_cm, b.z_cm], dtype=np.float64),
        "rim_left": np.array([b.rim_center_x_cm, b.y_cm - r, b.z_cm], dtype=np.float64),
        "rim_right": np.array([b.rim_center_x_cm, b.y_cm + r, b.z_cm], dtype=np.float64),
    }


def rim_circle(side: str = "near", samples: int = 96) -> np.ndarray:
    b = basket(side)
    d = float(b.attacking_direction)
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    x = b.rim_center_x_cm + d * RIM_INSIDE_RADIUS_CM * np.cos(theta)
    y = b.y_cm + RIM_INSIDE_RADIUS_CM * np.sin(theta)
    z = np.full_like(theta, b.z_cm)
    return np.column_stack([x, y, z]).astype(np.float64)


def key_floor_landmarks(side: str = "near") -> dict[str, np.ndarray]:
    """High-value floor anchors around one basket."""
    b = basket(side)
    d = float(b.attacking_direction)
    baseline_x = 0.0 if side.lower() == "near" else COURT_LENGTH_CM
    ft_x = b.board_x_cm + d * FREE_THROW_LINE_BOARD_DISTANCE_CM
    lane_half = PAINT_WIDTH_CM / 2.0
    return {
        "baseline_left_lane": np.array([baseline_x, b.y_cm - lane_half, 0.0], dtype=np.float64),
        "baseline_right_lane": np.array([baseline_x, b.y_cm + lane_half, 0.0], dtype=np.float64),
        "ft_left_lane": np.array([ft_x, b.y_cm - lane_half, 0.0], dtype=np.float64),
        "ft_right_lane": np.array([ft_x, b.y_cm + lane_half, 0.0], dtype=np.float64),
        "rim_floor_center": np.array([b.rim_center_x_cm, b.y_cm, 0.0], dtype=np.float64),
    }


def calibration_landmarks(side: str = "near") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out.update(inner_rectangle_corners(side))
    out.update(rim_cardinal_points(side))
    out.update(key_floor_landmarks(side))
    return out


def as_kalicalib_coordinates(points: np.ndarray) -> np.ndarray:
    """Convert our +Z-up world convention to KaliCalib's historical -Z-up convention."""
    out = np.asarray(points, dtype=np.float64).copy()
    out[..., 2] *= -1.0
    return out
