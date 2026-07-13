import pytest
from pydantic import ValidationError

from agent_gantry.schema.a2a import AgentCard, AgentSkill


def test_a2a_newline_injection():
    with pytest.raises(ValidationError, match="Value cannot contain newline characters"):
        AgentSkill(id="test\n", name="name", description="desc")

    with pytest.raises(ValidationError, match="Value cannot contain newline characters"):
        AgentCard(name="name\n", description="desc", url="url")
