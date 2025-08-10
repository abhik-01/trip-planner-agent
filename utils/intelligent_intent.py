"""
Enhanced Intent Classification System
Uses LLM as primary method with semantic understanding
"""

from typing import Dict, Any, List
from utils.set_llm import get_llm
from prompts import format_prompt, PromptType
from json import loads
import re


class IntelligentIntentClassifier:
    """
    LLM-powered intent classification with semantic understanding
    No hardcoded keywords - fully adaptive to natural language
    """
    
    @staticmethod
    def classify_intent(user_input: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Primary intent classification using LLM with rich context understanding
        """
        llm = get_llm(temperature=0.2)  # Low temp for consistent classification
        
        # Build conversation context
        recent_context = ""
        for msg in chat_history[-4:]:  # More context for better understanding
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                recent_context += f"{role}: {content[:150]}\n"
        
        prompt = format_prompt(
            PromptType.INTENT_CLASSIFICATION,
            recent_context=recent_context,
            user_input=user_input
        )
        
        try:
            response = llm.invoke(prompt).content.strip()
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                result = loads(json_match.group())
                
                # Enhanced validation and confidence scoring
                if IntelligentIntentClassifier._validate_classification(result, user_input):
                    return result
                    
        except Exception:
            # Intelligent fallback - still LLM based but with simpler prompt
            return IntelligentIntentClassifier._intelligent_fallback(user_input, chat_history)

    @staticmethod
    def _validate_classification(result: Dict[str, Any], user_input: str) -> bool:
        """Validate LLM classification results for consistency"""
        required_fields = ['intent', 'ready_to_plan']
        return all(field in result for field in required_fields)
    
    @staticmethod
    def _intelligent_fallback(user_input: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Intelligent fallback using simpler LLM classification"""
        llm = get_llm(temperature=0.3)
        
        # Use centralized semantic intent fallback prompt
        prompt = format_prompt(
            PromptType.SEMANTIC_INTENT_FALLBACK,
            user_input=user_input
        )
        
        try:
            response = llm.invoke(prompt).content.strip()
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                result = loads(json_match.group())
                intent = result.get('intent', 'chat')
                confidence = result.get('confidence', 0.5)
                
                return {
                    'intent': intent,
                    'exploring': 'destinations' if intent == 'explore' else None,
                    'planning_destination': None,
                    'ready_to_plan': intent == 'plan',
                    'confidence': confidence
                }

        except Exception:
            # Ultimate fallback - assume chat
            return {
                'intent': 'chat',
                'exploring': None,
                'planning_destination': None,
                'ready_to_plan': False,
                'confidence': 0.1
            }


class SemanticQueryClassifier:
    """
    Classify specific types of travel queries without hardcoded keywords
    """
    
    @staticmethod
    def classify_travel_query(user_input: str, context: Dict[str, Any]) -> str:
        """
        Classify travel-related queries semantically
        Returns: weather|activities|nearby|budget|flights|accommodation|food|general
        """
        llm = get_llm(temperature=0.2)
        
        # Use centralized semantic query classification prompt
        prompt = format_prompt(
            PromptType.SEMANTIC_QUERY_CLASSIFICATION,
            user_input=user_input,
            destination=context.get('destination', 'Unknown destination')
        )
        
        try:
            response = llm.invoke(prompt).content.strip().lower()
            
            # Validate response is one of expected categories
            valid_categories = ['weather', 'activities', 'nearby', 'budget', 'flights', 'accommodation', 'food', 'general']
            
            for category in valid_categories:
                if category in response:
                    return category

        except Exception as e:
            return 'general'  # Safe fallback
