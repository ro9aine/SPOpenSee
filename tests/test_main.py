import main


def test_parse_args_defaults() -> None:
    args = main.parse_args([])

    assert args.disable_hands is False
    assert args.disable_mic is False


def test_parse_args_supports_disable_flags() -> None:
    args = main.parse_args(["--disable-hands", "--disable-mic"])

    assert args.disable_hands is True
    assert args.disable_mic is True
