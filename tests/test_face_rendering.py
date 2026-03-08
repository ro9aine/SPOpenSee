import numpy as np

from cutoutcam.generators.character_generator import CharacterGenerator
from cutoutcam.state import BState, EState, FaceState


def test_generate_with_mask_keeps_open_eye_visible_during_wink() -> None:
    generator = CharacterGenerator("default", width=320, height=240)

    both_open, _ = generator.generate_with_mask(FaceState(left_eye=EState.OPEN, right_eye=EState.OPEN))
    wink, _ = generator.generate_with_mask(FaceState(left_eye=EState.CLOSED, right_eye=EState.OPEN))

    diff = np.abs(wink.astype(np.int16) - both_open.astype(np.int16))
    changed_columns = np.where(diff.sum(axis=(0, 2)) > 0)[0]

    assert changed_columns.size > 0
    midpoint = wink.shape[1] // 2
    assert changed_columns.min() < midpoint
    assert changed_columns.max() < midpoint


def test_generate_with_mask_applies_closed_eye_horizontal_offset() -> None:
    generator = CharacterGenerator("default", width=320, height=240)
    state = FaceState(left_eye=EState.CLOSED, right_eye=EState.OPEN)

    baseline, _ = generator.generate_with_mask(state)
    generator.set_layout({"closed_eyes_x": 18})
    shifted, _ = generator.generate_with_mask(state)

    diff = np.abs(shifted.astype(np.int16) - baseline.astype(np.int16))
    changed_columns = np.where(diff.sum(axis=(0, 2)) > 0)[0]

    assert changed_columns.size > 0
    assert changed_columns.max() < shifted.shape[1] // 2


def test_generate_with_mask_supports_independent_brow_offsets() -> None:
    generator = CharacterGenerator("default", width=320, height=240)
    state = FaceState(left_brow=BState.UP, right_brow=BState.MIDDLE)

    baseline, _ = generator.generate_with_mask(state)
    generator.set_layout({"brow_left_y": 20})
    shifted, _ = generator.generate_with_mask(state)

    diff = np.abs(shifted.astype(np.int16) - baseline.astype(np.int16))
    changed_columns = np.where(diff.sum(axis=(0, 2)) > 0)[0]

    assert changed_columns.size > 0
    assert changed_columns.max() < shifted.shape[1] // 2


def test_frontal_face_ignores_side_eye_offsets() -> None:
    generator = CharacterGenerator("default", width=320, height=240)
    state = FaceState()

    baseline, _ = generator.generate_with_mask(state)
    generator.set_layout({"eyes_left_x": -76, "eyes_right_x": 62, "eyes_left_y": 2, "eyes_right_y": 11})
    shifted, _ = generator.generate_with_mask(state)

    assert np.array_equal(baseline, shifted)
