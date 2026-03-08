from pathlib import Path

from cutoutcam.app_controls import LayoutControls, TRACKBARS


def test_load_settings_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    settings = LayoutControls.load_settings(tmp_path / "missing.json")

    assert settings["body_scale"] == TRACKBARS["body_scale"][0]
    assert "bg_image_path" not in settings


def test_save_and_load_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    values = {key: default for key, (default, _low, _high) in TRACKBARS.items()}
    values["body_x"] = 42
    values["mic_enabled"] = 0
    values["mouth_side_left_x"] = -31
    values["bg_image_path"] = "assets/bg.png"

    LayoutControls.save_settings(path, values)
    loaded = LayoutControls.load_settings(path)

    assert loaded["body_x"] == 42
    assert loaded["mic_enabled"] == 0
    assert loaded["mouth_side_left_x"] == -31
    assert loaded["bg_image_path"] == "assets/bg.png"


def test_load_settings_migrates_legacy_side_mouth_keys(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    path.write_text(
        '{"mouth_left_x": -31, "mouth_left_y": 7, "mouth_right_x": 35, "mouth_right_y": -4}',
        encoding="utf-8",
    )

    loaded = LayoutControls.load_settings(path)

    assert loaded["mouth_side_left_x"] == -31
    assert loaded["mouth_side_left_y"] == 7
    assert loaded["mouth_side_right_x"] == 35
    assert loaded["mouth_side_right_y"] == -4
