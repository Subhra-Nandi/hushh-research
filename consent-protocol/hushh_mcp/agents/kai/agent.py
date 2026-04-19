"""
Hushh Kai Financial Agent (ADK Port)

Advanced Financial Analyst Coordinator.
MIGRATED TO ADK (v2.0.0)
"""

import logging
import os
import re
from typing import Any, Dict

from hushh_mcp.hushh_adk.core import HushhAgent
from hushh_mcp.hushh_adk.manifest import ManifestLoader
from hushh_mcp.types import UserID

# Import tools
from .tools import (
    perform_fundamental_analysis,
    perform_sentiment_analysis,
    perform_valuation_analysis,
)

logger = logging.getLogger(__name__)


class KaiAgent(HushhAgent):
    """
    Agentic Kai Financial Coordinator.
    """

    def __init__(self):
        manifest_path = os.path.join(os.path.dirname(__file__), "agent.yaml")
        self.manifest = ManifestLoader.load(manifest_path)

        super().__init__(
            name=self.manifest.name,
            model=self.manifest.model,
            system_prompt=self.manifest.system_instruction,
            tools=[
                perform_fundamental_analysis,
                perform_sentiment_analysis,
                perform_valuation_analysis,
            ],
            required_scopes=self.manifest.required_scopes,
        )

    def handle_message(
        self, message: str, user_id: UserID, consent_token: str = ""
    ) -> Dict[str, Any]:
        """
        Agentic Entry Point.
        Detects the language of the incoming message and passes it
        to the tools so responses are returned in the user's language.
        """
        try:
            # Detect language from the incoming message
            language = self._detect_language(message)
            logger.info(f"KaiAgent detected language: {language}")

            # Inject language into the message context for the LLM
            language_instruction = (
                f"\n\n[SYSTEM: The user is communicating in language code '{language}'. "
                f"Call all tools with language='{language}' and respond to the user "
                f"in the same language as their message.]"
            )
            message_with_language = message + language_instruction

            # Execute ADK run
            response = self.run(
                message_with_language,
                user_id=user_id,
                consent_token=consent_token,
            )

            return {
                "response": response.text if hasattr(response, "text") else str(response),
                "is_complete": True,
                "language": language,
            }

        except Exception as e:
            logger.error(f"KaiAgent error: {e}")
            return {
                "response": "I encountered an error analyzing the market data.",
                "error": str(e),
            }

    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text using character analysis.
        Returns a BCP-47 language code (e.g., 'en', 'hi', 'fr', 'ja').
        Falls back to 'en' if detection is uncertain.
        """
        if not text or len(text.strip()) < 3:
            return "en"

        # Detect script-based languages using unicode ranges
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        chinese_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u30FF]', text))
        korean_chars = len(re.findall(r'[\uAC00-\uD7AF]', text))
        cyrillic_chars = len(re.findall(r'[\u0400-\u04FF]', text))
        greek_chars = len(re.findall(r'[\u0370-\u03FF]', text))

        total = len(text)

        if hindi_chars / total > 0.2:
            return "hi"
        if arabic_chars / total > 0.2:
            return "ar"
        if japanese_chars / total > 0.2:
            return "ja"
        if chinese_chars / total > 0.2:
            return "zh"
        if korean_chars / total > 0.2:
            return "ko"
        if cyrillic_chars / total > 0.2:
            return "ru"
        if greek_chars / total > 0.2:
            return "el"

        # Default to English for Latin-script languages
        return "en"


# Singleton
_kai_agent = None


def get_kai_agent():
    global _kai_agent
    if not _kai_agent:
        _kai_agent = KaiAgent()
    return _kai_agent