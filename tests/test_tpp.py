"""Test the TomlParmParse utility."""

import pytest
from cmlm.utils import TomlParmParse

# Dummy toml data with values of various types
answer1 = "apples"
answer2 = [0, 1]
answer3 = 0.02
answer4 = 4
test_docstring = "my test docstring"
tpp_string = f"""
a = '{answer1}'

[b]
i = 1
ii = 2
iii = 3

[b.states]
alaska = false
arizona = {answer2}

[c]
washington = 0.01
adams = {answer3}
jefferson = 0.03
"""


def test_get_no_default():
    """Look up required values from table."""
    tpp = TomlParmParse.parse_file(additional_args=tpp_string, output_type="doc")
    assert tpp["a"] == answer1
    assert tpp.get("a") == answer1
    assert tpp["c"].get("adams") == answer3
    # All these different formats should be equivalent
    assert tpp.get("b.states.arizona") == answer2
    assert tpp["b"].get("states.arizona") == answer2
    assert tpp["b.states"].get("arizona") == answer2
    assert tpp["b"]["states"].get("arizona") == answer2
    assert tpp["b.states.arizona"] == answer2

    # Not in table, shouldn't be found
    with pytest.raises(RuntimeError):
        tpp["c"].get("monroe")


def test_get_with_default():
    """Look up optional values from table."""
    tpp = TomlParmParse.parse_file(additional_args=tpp_string, output_type="doc")

    # Default has no effect if in table
    assert tpp["c"].get("adams", 2) == answer3

    # Default gets used if not in table
    assert tpp["c"].get("madison", answer4) == answer4

    # Default gets re-used after being added
    assert tpp["c"].get("madison") == answer4


def test_doc():
    """Document while looking up."""
    tpp = TomlParmParse.parse_file(additional_args=tpp_string, output_type="doc")
    tpp["c"].get("jefferson", doc=test_docstring)
    assert tpp["c"].get("jefferson").trivia.comment == "# " + test_docstring


def test_save_load_roundtrip():
    """Write to file, read from file, esnure results are the same."""
    # Look up some data then save the doc output
    tpp = TomlParmParse.parse_file(additional_args=tpp_string, output_type="doc")
    tpp.get("c.adams")
    tpp.get("b.states.alaska", doc=test_docstring)
    tpp.dump("test.toml")
    npp = TomlParmParse.parse_file(file_name="test.toml", output_type="original")
    assert npp.get("c.adams") == answer3

    # unused inputs should not have been saved with 'doc' output_type
    with pytest.raises(RuntimeError):
        npp["c"].get("washington")

    # Did we correctly save and read the docstring
    item = npp.get("b.states.alaska")
    assert not item
    assert item.trivia.comment == "# " + test_docstring
