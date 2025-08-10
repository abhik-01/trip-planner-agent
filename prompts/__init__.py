"""
Prompts module for Trip Planner AI
Centralized prompt management with LangChain PromptTemplate
"""

from .prompt import (
    PromptType,
    PromptRegistry,
    get_prompt,
    format_prompt
)

__all__ = [
    'PromptType',
    'PromptRegistry', 
    'get_prompt',
    'format_prompt'
]
