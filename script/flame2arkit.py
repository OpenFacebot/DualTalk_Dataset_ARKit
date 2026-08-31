#!/usr/bin/env python3
"""FLAME -> ARKit pure math conversion (no file IO in this module).

Rewrite of the base version `convert_dualtalk_flame_to_arkit_bvls_csv61_1.py`:
the same bounded, regularized inverse solve of the 51x103 ARKit-to-FLAME
matrix, reorganized as a reusable `Flame2ARKit_Linear` converter class.

This module is frame-independent: it performs per-frame math only and never
touches temporal information.  Sequence-level operations such as smoothing are
the caller's responsibility (see convert_flame2arkit.py, --smooth flag).

FLAME106 layout consumed by `convert()` (one row per frame):
    [  0: 50] expression, observed dims 0-49
    [ 50:100] expression, unobserved dims (padding, ignored by the solver)
    [100:103] neck rotation vector (radians)
    [103:106] jaw rotation vector (radians)

Output of `convert()` is 61D motion per frame, ordered as MOTION61_ORDER:
    [ 0:51] 51 ARKit blendshape weights (REFERENCE_CSV_ARKIT51_ORDER)
    [51]    TongueOut, fixed 0
    [52:55] HeadYaw / HeadPitch / HeadRoll, YXZ Euler radians
    [55:61] left/right eye rotations, fixed 0 (DualTalk has no eye pose)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Channel ordering and solver constants (identical to the base script).
# ---------------------------------------------------------------------------

MATRIX_ARKIT51_ORDER = (
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft",
    "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "EyeBlinkLeft", "EyeBlinkRight", "EyeLookDownLeft", "EyeLookDownRight",
    "EyeLookInLeft", "EyeLookInRight", "EyeLookOutLeft", "EyeLookOutRight",
    "EyeLookUpLeft", "EyeLookUpRight", "EyeSquintLeft", "EyeSquintRight",
    "EyeWideLeft", "EyeWideRight", "JawForward", "JawLeft", "JawOpen",
    "JawRight", "MouthClose", "MouthDimpleLeft", "MouthDimpleRight",
    "MouthFrownLeft", "MouthFrownRight", "MouthFunnel", "MouthLeft",
    "MouthLowerDownLeft", "MouthLowerDownRight", "MouthPressLeft",
    "MouthPressRight", "MouthPucker", "MouthRight", "MouthRollLower",
    "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthSmileLeft",
    "MouthSmileRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthUpperUpLeft", "MouthUpperUpRight", "NoseSneerLeft", "NoseSneerRight",
)

# Output order of the 51 blendshape channels in the 61D motion vector.
REFERENCE_CSV_ARKIT51_ORDER = (
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight",
    "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight",
    "EyeSquintRight", "EyeWideRight", "JawForward", "JawLeft", "JawRight",
    "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthLeft",
    "MouthRight", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft",
    "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
    "MouthStretchRight", "MouthRollLower", "MouthRollUpper", "MouthShrugLower",
    "MouthShrugUpper", "MouthPressLeft", "MouthPressRight",
    "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft",
    "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft",
    "CheekSquintRight", "NoseSneerLeft", "NoseSneerRight",
)

ROTATION_AND_EXTRA10 = (
    "TongueOut", "HeadYaw", "HeadPitch", "HeadRoll", "LeftEyeYaw",
    "LeftEyePitch", "LeftEyeRoll", "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
)
MOTION61_ORDER = REFERENCE_CSV_ARKIT51_ORDER + ROTATION_AND_EXTRA10

FLAME_EXPR_DIM = 100
FLAME_EXPR_OBSERVED = 50
FLAME_POSE_DIM = 6
FLAME106_DIM = FLAME_EXPR_DIM + FLAME_POSE_DIM  # 100 expr + 3 neck + 3 jaw
ARKIT51_DIM = len(MATRIX_ARKIT51_ORDER)         # 51
ARKIT52_DIM = ARKIT51_DIM + 1                   # + TongueOut
MOTION61_DIM = len(MOTION61_ORDER)              # 61

EYELOOK_NAMES = frozenset(name for name in MATRIX_ARKIT51_ORDER if name.startswith("EyeLook"))
JAW_NAMES = frozenset({"JawOpen", "JawLeft", "JawRight", "JawForward", "MouthClose"})
SYMMETRY_PAIRS = (
    ("MouthSmileLeft", "MouthSmileRight"),
    ("MouthFrownLeft", "MouthFrownRight"),
    ("MouthDimpleLeft", "MouthDimpleRight"),
    ("MouthStretchLeft", "MouthStretchRight"),
    ("MouthPressLeft", "MouthPressRight"),
    ("MouthLowerDownLeft", "MouthLowerDownRight"),
    ("MouthUpperUpLeft", "MouthUpperUpRight"),
)

JAW_WEIGHT = 50.0
L2_WEIGHT = 3.0
SYMMETRY_WEIGHT = 30.0

_MATRIX_ARKIT51_INDEX = {name: i for i, name in enumerate(MATRIX_ARKIT51_ORDER)}

HEAD_AXIS_NAMES = ("YAW", "PITCH", "ROLL")


def validate_head_calibration(
    signs: object, gains: object, offsets: object, limits: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate per-axis head calibration scalars, return them as float64."""
    sign_values = np.asarray(signs, dtype=np.float64)
    gain_values = np.asarray(gains, dtype=np.float64)
    offset_values = np.asarray(offsets, dtype=np.float64)
    limit_values = np.asarray(limits, dtype=np.float64)
    for name, values in (
        ("signs", sign_values), ("gains", gain_values),
        ("offsets", offset_values), ("limits", limit_values),
    ):
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"head {name} must be 3 finite values, got {values!r}")
    if not np.all(np.isin(sign_values, (-1.0, 1.0))):
        raise ValueError("head signs must each be -1 or 1")
    if np.any(gain_values < 0.0):
        raise ValueError("head gains must be non-negative")
    if np.any(limit_values <= 0.0):
        raise ValueError("head limits must be positive")
    return sign_values, gain_values, offset_values, limit_values


class Flame2ARKit_Linear:
    """Bounded regularized inverse of the ARKit51-to-FLAME103 linear matrix.

    Only math here: call `set_matrix` with the [51, 103] forward matrix,
    then `convert` FLAME106 arrays into 61D motion arrays.
    """

    def __init__(self, flame_dim: int = FLAME106_DIM, arkit_dim: int = ARKIT52_DIM):
        if flame_dim != FLAME106_DIM:
            raise ValueError(f"Only flame_dim={FLAME106_DIM} is supported, got {flame_dim}")
        if arkit_dim not in (ARKIT51_DIM, ARKIT52_DIM, MOTION61_DIM):
            raise ValueError(f"Unsupported arkit_dim: {arkit_dim}")
        self.flame_dim = flame_dim
        self.arkit_dim = arkit_dim
        self.matrix: np.ndarray | None = None
        self.design: np.ndarray | None = None
        self.active: np.ndarray | None = None
        self.jaw_global: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_matrix(self, matrix: np.ndarray) -> None:
        """Store the forward matrix A [51, 103] and build the solver."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (ARKIT51_DIM, 103) or not np.isfinite(matrix).all():
            raise ValueError(f"Expected finite matrix [51,103], got {matrix.shape}")
        self.matrix = matrix
        self._build_solver(matrix)

    def _build_solver(self, matrix: np.ndarray) -> None:
        """Assemble the stacked design matrix of the bounded least squares.

        Rows: 50 observed expression + 3 weighted jaw observations,
              7 symmetry penalties, 43 L2 shrinkage rows.
        Columns: the 43 active ARKit channels (8 EyeLook channels excluded).
        """
        active = np.asarray(
            [i for i, name in enumerate(MATRIX_ARKIT51_ORDER) if name not in EYELOOK_NAMES],
            dtype=np.int64,
        )
        active_position = {MATRIX_ARKIT51_ORDER[index]: pos for pos, index in enumerate(active)}
        jaw_local = np.asarray([active_position[name] for name in sorted(JAW_NAMES)], dtype=np.int64)
        jaw_global = active[jaw_local]

        observed = np.zeros((53, len(active)), dtype=np.float64)
        observed[:50] = matrix[active, :50].T
        observed[50:53, jaw_local] = matrix[jaw_global, 100:103].T * JAW_WEIGHT

        symmetry = np.zeros((len(SYMMETRY_PAIRS), len(active)), dtype=np.float64)
        for row, (left, right) in enumerate(SYMMETRY_PAIRS):
            symmetry[row, active_position[left]] = 1.0
            symmetry[row, active_position[right]] = -1.0

        self.design = np.vstack(
            (
                observed,
                np.sqrt(SYMMETRY_WEIGHT) * symmetry,
                np.sqrt(L2_WEIGHT) * np.eye(len(active), dtype=np.float64),
            )
        )
        self.active = active
        self.jaw_global = jaw_global

    # ------------------------------------------------------------------
    # Conversion entry
    # ------------------------------------------------------------------

    def convert(
        self,
        flame: np.ndarray,
        *,
        head_signs: object = (1.0, 1.0, 1.0),
        head_gains: object = (1.0, 1.0, 1.0),
        head_offsets: object = (0.0, 0.0, 0.0),
        head_limits: object = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Convert FLAME106 to 61D motion, frame by frame (no temporal ops).

        flame: [T, 106] or [106]; returns [T, 61] or [61].  Frames are
        independent; apply any sequence smoothing before calling this.

        Head calibration (per frame, applied to HeadYaw/Pitch/Roll):
            head = clip(ypr * signs * gains + offsets, -limits, +limits)
        Defaults reproduce the base script (plain YXZ Euler clipped to +-1).
        Sequence-level neutral-pose subtraction must be folded into
        head_offsets by the caller: offsets_eff = offsets - neutral*signs*gains.
        """
        flame = np.asarray(flame, dtype=np.float64)
        single = flame.ndim == 1
        if single:
            flame = flame[None, :]
        if flame.ndim != 2 or flame.shape[1] != self.flame_dim:
            raise ValueError(f"Expected flame [T,{self.flame_dim}], got {flame.shape}")
        if not np.isfinite(flame).all():
            raise ValueError("flame contains NaN or Inf")
        motion = self.convert_106(
            flame,
            head_signs=head_signs,
            head_gains=head_gains,
            head_offsets=head_offsets,
            head_limits=head_limits,
        )
        return motion[0] if single else motion

    def convert_106(
        self,
        flame: np.ndarray,
        *,
        head_signs: object = (1.0, 1.0, 1.0),
        head_gains: object = (1.0, 1.0, 1.0),
        head_offsets: object = (0.0, 0.0, 0.0),
        head_limits: object = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """FLAME106 [T,106] -> motion61 [T,61]."""
        expression = flame[:, :FLAME_EXPR_OBSERVED]
        neck_rotvec = flame[:, FLAME_EXPR_DIM:FLAME_EXPR_DIM + 3]
        jaw_rotvec = flame[:, FLAME_EXPR_DIM + 3:FLAME_EXPR_DIM + 6]

        arkit51 = self.convert_106_face(expression, jaw_rotvec)
        head = self.convert_106_headpose(
            neck_rotvec,
            signs=head_signs,
            gains=head_gains,
            offsets=head_offsets,
            limits=head_limits,
        )
        eyeball = self.convert_106_eyeball(len(flame))

        motion = np.zeros((len(flame), MOTION61_DIM), dtype=np.float64)
        for output_index, name in enumerate(REFERENCE_CSV_ARKIT51_ORDER):
            motion[:, output_index] = arkit51[:, _MATRIX_ARKIT51_INDEX[name]]
        # 51: TongueOut stays zero; 52:55 head euler; 55:61 eyeball stays zero.
        motion[:, 52:55] = head
        if not np.isfinite(motion).all():
            raise ValueError("Assembled motion contains NaN or Inf")
        return motion

    def convert_106_face(self, expression50: np.ndarray, jaw_rotvec: np.ndarray) -> np.ndarray:
        """Solve the 51 ARKit blendshape weights [T,51] in [0,1], per frame.

        EyeLook channels stay exactly zero.  Any temporal smoothing of the
        inputs must be done by the caller before this method.
        """
        if self.design is None:
            raise RuntimeError("Call set_matrix before convert")
        expression50 = np.asarray(expression50, dtype=np.float64)
        jaw_rotvec = np.asarray(jaw_rotvec, dtype=np.float64)
        if expression50.ndim != 2 or expression50.shape[1] != FLAME_EXPR_OBSERVED:
            raise ValueError(f"Expected expression [T,{FLAME_EXPR_OBSERVED}], got {expression50.shape}")
        if jaw_rotvec.shape != (len(expression50), 3):
            raise ValueError(f"Expected jaw rotvec [T,3], got {jaw_rotvec.shape}")

        result = np.zeros((len(expression50), ARKIT51_DIM), dtype=np.float64)
        regularization_zeros = np.zeros(len(SYMMETRY_PAIRS) + len(self.active), dtype=np.float64)
        failures = 0
        for frame, (expression, jaw) in enumerate(zip(expression50, jaw_rotvec)):
            target = np.concatenate((expression, jaw * JAW_WEIGHT, regularization_zeros))
            solved = lsq_linear(
                self.design, target, bounds=(0.0, 1.0), method="bvls", tol=1e-7, max_iter=200
            )
            valid = solved.success and np.isfinite(solved.x).all()
            if valid:
                result[frame, self.active] = np.clip(solved.x, 0.0, 1.0)
            else:
                failures += 1
        if failures:
            raise ValueError(f"BVLS solve failed on {failures} frame(s)")
        return result

    @staticmethod
    def rotvec_to_head_euler(neck_rotvec: np.ndarray) -> np.ndarray:
        """Neck rotvec [T,3] (or [3]) -> YXZ Euler radians [T,3] (or [3])."""
        neck_rotvec = np.asarray(neck_rotvec, dtype=np.float64)
        single = neck_rotvec.ndim == 1
        if single:
            neck_rotvec = neck_rotvec[None, :]
        if neck_rotvec.ndim != 2 or neck_rotvec.shape[1] != 3:
            raise ValueError(f"Expected neck rotvec [T,3], got {neck_rotvec.shape}")
        if not np.isfinite(neck_rotvec).all():
            raise ValueError("neck rotvec contains NaN or Inf")
        euler = Rotation.from_rotvec(neck_rotvec).as_euler("YXZ", degrees=False)
        return euler[0] if single else euler

    def convert_106_headpose(
        self,
        neck_rotvec: np.ndarray,
        *,
        signs: object = (1.0, 1.0, 1.0),
        gains: object = (1.0, 1.0, 1.0),
        offsets: object = (0.0, 0.0, 0.0),
        limits: object = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Neck rotvec [T,3] -> calibrated HeadYaw/Pitch/Roll [T,3] radians.

        head = clip(ypr * signs * gains + offsets, -limits, +limits)
        Defaults give the base behavior: plain YXZ Euler clipped to +-1 rad.
        """
        sign_values, gain_values, offset_values, limit_values = validate_head_calibration(
            signs, gains, offsets, limits
        )
        raw_head = self.rotvec_to_head_euler(neck_rotvec)
        corrected = raw_head * sign_values * gain_values + offset_values
        corrected = np.clip(corrected, -limit_values, limit_values)
        if not np.isfinite(corrected).all():
            raise ValueError("Corrected head rotation contains NaN or Inf")
        return corrected

    def convert_106_eyeball(self, frames: int) -> np.ndarray:
        """DualTalk has no eye pose: 6 zero channels per frame."""
        return np.zeros((frames, 6), dtype=np.float64)

    # ------------------------------------------------------------------
    # Quality metrics (same definitions as the base script).
    # ------------------------------------------------------------------

    def metrics(
        self,
        arkit51: np.ndarray,
        expression50: np.ndarray,
        jaw_rotvec: np.ndarray,
    ) -> dict[str, float]:
        """Reconstruction quality of a solved [T,51] ARKit weight sequence."""
        if self.matrix is None or self.jaw_global is None:
            raise RuntimeError("Call set_matrix before metrics")
        reconstructed = arkit51 @ self.matrix
        jaw_only = arkit51[:, self.jaw_global] @ self.matrix[self.jaw_global, 100:103]
        pair_differences = [
            np.mean(
                np.abs(
                    arkit51[:, _MATRIX_ARKIT51_INDEX[left]]
                    - arkit51[:, _MATRIX_ARKIT51_INDEX[right]]
                )
            )
            for left, right in SYMMETRY_PAIRS
        ]
        jaw_open = arkit51[:, _MATRIX_ARKIT51_INDEX["JawOpen"]]
        return {
            "expression50_rmse": float(np.sqrt(np.mean((reconstructed[:, :50] - expression50) ** 2))),
            "jaw3_rmse": float(np.sqrt(np.mean((jaw_only - jaw_rotvec) ** 2))),
            "jawopen_source_jawx_correlation": _correlation(jaw_open, jaw_rotvec[:, 0]),
            "upper_saturation_fraction": float(np.mean(arkit51 >= 1.0 - 1e-6)),
            "mean_left_right_difference": float(np.mean(pair_differences)),
        }


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    import math

    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None
