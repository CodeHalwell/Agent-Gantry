import pytest
from pydantic import ValidationError

from agent_gantry.schema.a2a import AgentCard, AgentSkill
from agent_gantry.schema.config import A2AAgentConfig


def test_a2a_newline_injection():
    with pytest.raises(ValidationError, match="Value cannot contain newline characters"):
        AgentSkill(id="test\n", name="name", description="desc")

    with pytest.raises(ValidationError, match="Value cannot contain newline characters"):
        AgentCard(name="name\n", description="desc", url="url")


def test_a2a_agent_url_must_be_http():
    with pytest.raises(ValidationError, match="http"):
        A2AAgentConfig(name="agent", url="ftp://agent.example.com")
    assert A2AAgentConfig(name="agent", url="https://agent.example.com").url.startswith("https")
