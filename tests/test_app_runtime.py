from collections import deque

from cutoutcam.app_runtime import merge_face_with_pose, smooth_face_state
from cutoutcam.state import EState, FaceState, FState


def test_merge_face_with_pose_preserves_pose_fields() -> None:
    face_state = FaceState(turn=FState.LEFT, left_eye=EState.CLOSED)
    pose_state = FaceState(
        left_arm_visible=True,
        right_arm_visible=True,
        left_shoulder_x=0.11,
        left_shoulder_y=0.22,
        right_shoulder_x=0.88,
        right_shoulder_y=0.33,
        left_elbow_x=0.15,
        left_elbow_y=0.44,
        right_elbow_x=0.79,
        right_elbow_y=0.55,
        left_hand_x=0.19,
        left_hand_y=0.66,
        right_hand_x=0.73,
        right_hand_y=0.77,
    )

    merged = merge_face_with_pose(face_state, pose_state)

    assert merged.turn == FState.LEFT
    assert merged.left_eye == EState.CLOSED
    assert merged.left_arm_visible is True
    assert merged.right_arm_visible is True
    assert merged.left_hand_x == 0.19
    assert merged.right_hand_y == 0.77


def test_smoothing_then_merge_keeps_latest_pose_state() -> None:
    history = deque(
        [
            FaceState(turn=FState.CENTER),
            FaceState(turn=FState.RIGHT),
        ],
        maxlen=2,
    )
    latest_pose = FaceState(left_arm_visible=True, left_hand_x=0.25, left_hand_y=0.75)

    merged = merge_face_with_pose(smooth_face_state(history), latest_pose)

    assert merged.turn == FState.RIGHT
    assert merged.left_arm_visible is True
    assert merged.left_hand_x == 0.25
    assert merged.left_hand_y == 0.75
