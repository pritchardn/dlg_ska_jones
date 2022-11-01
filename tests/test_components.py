import pytest

from dlg_ska_jones import MyAppDROP

given = pytest.mark.parametrize


def test_myApp_class():
    assert MyAppDROP("a", "a").run() == "Hello from MyAppDROP"
