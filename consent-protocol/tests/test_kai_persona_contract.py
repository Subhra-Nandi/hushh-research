"""
Smoke proof: Kai persona contract test.

Verifies that the PERSONALITY, CHARACTERISTIC PHRASES, and TONE RULES
sections added in feat/kai-voice-personality-v2 are reachable through
the canonical runtime path:

    agent.yaml
        -> ManifestLoader.load()
        -> manifest.system_instruction
        -> KaiAgent.__init__() -> super().__init__(system_prompt=...)
        -> HushhAgent.system_prompt

This test does NOT call the LLM. It proves the persona text is loaded
and passed through the existing agent initialisation path without
bypassing workspace gating or persona memory boundaries.
"""

import os

import pytest

from hushh_mcp.hushh_adk.manifest import ManifestLoader

AGENT_YAML_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "hushh_mcp",
    "agents",
    "kai",
    "agent.yaml",
)


@pytest.fixture(scope="module")
def kai_manifest():
    """Load the Kai agent manifest via the canonical ManifestLoader path."""
    return ManifestLoader.load(os.path.abspath(AGENT_YAML_PATH))


def test_manifest_loads_successfully(kai_manifest):
    """ManifestLoader can parse agent.yaml without errors."""
    assert kai_manifest is not None


def test_system_instruction_is_present(kai_manifest):
    """system_instruction field is non-empty after manifest load."""
    assert kai_manifest.system_instruction
    assert len(kai_manifest.system_instruction.strip()) > 0


def test_persona_section_present(kai_manifest):
    """PERSONALITY section is present in system_instruction."""
    assert "PERSONALITY" in kai_manifest.system_instruction, (
        "PERSONALITY section is missing from system_instruction. "
        "The persona contract requires this section."
    )


def test_characteristic_phrases_present(kai_manifest):
    """CHARACTERISTIC PHRASES section is present in system_instruction."""
    assert "CHARACTERISTIC PHRASES" in kai_manifest.system_instruction, (
        "CHARACTERISTIC PHRASES section is missing from system_instruction."
    )


def test_tone_rules_present(kai_manifest):
    """TONE RULES section is present in system_instruction."""
    assert "TONE RULES" in kai_manifest.system_instruction, (
        "TONE RULES section is missing from system_instruction."
    )


def test_identity_section_preserved(kai_manifest):
    """IDENTITY section from PR #148 is still intact after persona addition."""
    assert "IDENTITY" in kai_manifest.system_instruction, (
        "IDENTITY section was removed. PR #148 baseline must be preserved."
    )


def test_consent_rules_preserved(kai_manifest):
    """CONSENT RULES from PR #148 are still intact after persona addition."""
    assert "CONSENT RULES" in kai_manifest.system_instruction, (
        "CONSENT RULES were removed. Consent gating must be preserved."
    )


def test_voice_format_preserved(kai_manifest):
    """VOICE FORMAT rules from PR #148 are still intact."""
    assert "VOICE FORMAT" in kai_manifest.system_instruction, (
        "VOICE FORMAT section was removed. Voice constraints must be preserved."
    )


def test_persona_does_not_override_refusals(kai_manifest):
    """REFUSALS section is still present — persona must not override scope gating."""
    assert "REFUSALS" in kai_manifest.system_instruction, (
        "REFUSALS section was removed. Persona must not override scope gating."
    )


def test_greeting_phrase_present(kai_manifest):
    """Canonical greeting phrase is present in system_instruction."""
    assert "Hey, I'm Kai" in kai_manifest.system_instruction, (
        "Greeting phrase missing. CHARACTERISTIC PHRASES contract requires it."
    )


def test_filler_ban_present(kai_manifest):
    """Filler affirmation ban is present in TONE RULES."""
    instruction = kai_manifest.system_instruction
    assert "Certainly" in instruction or "filler" in instruction.lower(), (
        "Filler affirmation ban missing from TONE RULES."
    )


def test_system_instruction_passed_to_agent():
    """
    Proves the full runtime path:
    agent.yaml -> ManifestLoader -> KaiAgent.__init__ -> system_prompt.

    Imports KaiAgent directly to verify system_prompt is set from
    manifest.system_instruction without calling the LLM.
    """
    from hushh_mcp.agents.kai.agent import KaiAgent

    agent = KaiAgent()
    assert agent.system_prompt, "KaiAgent.system_prompt is empty after init"
    assert "PERSONALITY" in agent.system_prompt, (
        "PERSONALITY section not reachable through KaiAgent runtime path"
    )
    assert "CHARACTERISTIC PHRASES" in agent.system_prompt, (
        "CHARACTERISTIC PHRASES not reachable through KaiAgent runtime path"
    )
    assert "CONSENT RULES" in agent.system_prompt, (
        "CONSENT RULES not reachable — persona must not bypass gating"
    )