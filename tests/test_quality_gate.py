import json

from pet_breed_mlops.quality_gate import main, passes


def test_equal_and_within_tolerance_pass() -> None:
    assert passes(0.93, 0.93)
    assert passes(0.93, 0.925)


def test_drop_beyond_tolerance_fails() -> None:
    assert not passes(0.93, 0.90)


def test_cli_exit_codes(tmp_path) -> None:
    base, good, bad = (tmp_path / n for n in ("b.json", "g.json", "x.json"))
    base.write_text(json.dumps({"top1": 0.93}))
    good.write_text(json.dumps({"top1": 0.94}))
    bad.write_text(json.dumps({"top1": 0.50}))
    assert main(["--baseline", str(base), "--candidate", str(good)]) == 0
    assert main(["--baseline", str(base), "--candidate", str(bad)]) == 1