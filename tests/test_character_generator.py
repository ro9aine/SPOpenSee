from cutoutcam.generators.character_generator import CharacterGenerator
from cutoutcam.state import FaceState


def test_generate_with_mask_returns_expected_shapes() -> None:
    generator = CharacterGenerator("default", width=320, height=240)
    frame, mask = generator.generate_with_mask(FaceState())

    assert frame.shape == (240, 320, 3)
    assert mask.shape == (240, 320)
    assert frame.dtype.name == "uint8"
    assert mask.dtype.name == "uint8"


def test_generate_with_mask_supports_visible_arms() -> None:
    generator = CharacterGenerator("default", width=320, height=240)
    state = FaceState(left_arm_visible=True, right_arm_visible=True)

    frame, mask = generator.generate_with_mask(state)

    assert frame.any()
    assert mask.any()
