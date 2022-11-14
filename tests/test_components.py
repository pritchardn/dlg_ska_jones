import pytest

from dlg_ska_jones import AA05CaliTests

given = pytest.mark.parametrize


def test_AA05CaliTests():
    drop = AA05CaliTests("a", "a")
    drop.sky_model = 1
    assert drop.run() == 0
