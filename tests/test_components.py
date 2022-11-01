import pytest

from dlg_ska_jones import AA05CaliTests

given = pytest.mark.parametrize


def test_myApp_class():
    assert AA05CaliTests("a", "a").run() == 0
