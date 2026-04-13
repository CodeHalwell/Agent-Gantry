
from agent_gantry.core.router import _get_token_pattern


def test_get_token_pattern_exact_match():
    """Test that the pattern matches exact words as standalone tokens."""
    pattern = _get_token_pattern("test")
    assert pattern.search("this is a test")
    assert pattern.search("test is at the start")
    assert pattern.search("the word is test")
    assert pattern.search("test")


def test_get_token_pattern_partial_match():
    """Test that the pattern does not match partial words."""
    pattern = _get_token_pattern("test")
    assert not pattern.search("testing is fun")
    assert not pattern.search("pretest phase")
    assert not pattern.search("the tests are passing")


def test_get_token_pattern_special_chars():
    """Test that the pattern correctly escapes and matches special characters."""
    # Test with characters that have meaning in regex
    pattern_plus = _get_token_pattern("c++")
    assert pattern_plus.search("i program in c++")
    assert pattern_plus.search("c++ is hard")
    assert not pattern_plus.search("i program in c")

    # Test with dots
    pattern_dot = _get_token_pattern("node.js")
    assert pattern_dot.search("using node.js backend")
    assert not pattern_dot.search("using node_js")


def test_get_token_pattern_multi_word():
    """Test that the pattern matches phrases with spaces."""
    pattern = _get_token_pattern("open file")
    assert pattern.search("can you open file please")
    assert pattern.search("open file now")
    assert not pattern.search("can you open files please")
    assert not pattern.search("open  file")  # extra space


def test_get_token_pattern_case_sensitivity():
    """
    Test case sensitivity.
    Note: The pattern itself does not ignore case natively;
    _contains_token passes lowered strings.
    """
    pattern = _get_token_pattern("test")
    assert pattern.search("test")
    # By default, re.escape("test") makes a case-sensitive regex
    assert not pattern.search("Test")
    assert not pattern.search("TEST")


def test_get_token_pattern_punctuation():
    """Test that token matching works next to punctuation."""
    pattern = _get_token_pattern("test")
    assert pattern.search("this is a test, and it works")
    assert pattern.search("is this a test?")
    assert pattern.search("test! it worked.")
    assert pattern.search("(test)")


def test_get_token_pattern_empty_string():
    """Test behavior with an empty string."""
    pattern = _get_token_pattern("")
    # Pattern for empty string `(?<!\w)(?!\w)` matches between non-word boundaries
    # So it should find a match if there are spaces or punctuation
    assert pattern.search(" ") is not None
    # But it won't match inside a sequence of word characters
    assert pattern.search("anything") is None
