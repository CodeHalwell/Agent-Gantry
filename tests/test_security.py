from agent_gantry.core.security import validate_description


def test_validate_description_multiline_bypass():
    payload = "test {{\n payload \n}}"
    is_valid, msg = validate_description(payload)
    assert is_valid is False
    assert msg == "Description contains suspicious pattern"


def test_validate_description_valid():
    payload = "This is a normal description"
    is_valid, msg = validate_description(payload)
    assert is_valid is True
    assert msg is None
