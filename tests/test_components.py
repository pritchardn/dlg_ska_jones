import pytest

from dlg_ska_jones import AA05CaliTests

given = pytest.mark.parametrize


def test_AA05CaliTests():
    assert AA05CaliTests("a", "a").run() == 0
