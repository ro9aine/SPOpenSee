import main


def test_parse_args_defaults() -> None:
    args = main.parse_args([])

    assert args.disable_hands is False
    assert args.disable_mic is False
    assert args.ui == "qt"


def test_parse_args_supports_disable_flags() -> None:
    args = main.parse_args(["--disable-hands", "--disable-mic"])

    assert args.disable_hands is True
    assert args.disable_mic is True


def test_parse_args_supports_qt_ui() -> None:
    args = main.parse_args(["--ui", "qt"])

    assert args.ui == "qt"
