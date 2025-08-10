from json import loads
from re import search, DOTALL
from typing import Dict, Any

from utils.set_llm import get_llm
from prompts import format_prompt, PromptType


def screen_user_input_safety(user_input: str) -> Dict[str, Any]:
    """Use LLM to intelligently assess safety of user input"""
    
    if not user_input or not user_input.strip():
        return {"is_safe": True, "concern_type": "safe"}
    
    llm = get_llm(temperature=0.1)  # Low temp for consistent safety decisions
    
    prompt = format_prompt(
        PromptType.SAFETY_INPUT_SCREENING,
        user_input=user_input
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Extract JSON from response
        json_match = search(r'\{.*\}', content, DOTALL)

        if json_match:
            safety_result = loads(json_match.group())

            # Validate required fields
            if 'is_safe' not in safety_result:
                safety_result['is_safe'] = True
            if 'concern_type' not in safety_result:
                safety_result['concern_type'] = 'safe'
                
            return safety_result

    except Exception:
        # Fail-safe: if safety check fails, allow but log
        return {
            "is_safe": True, 
            "concern_type": "unknown", 
            "explanation": "Safety check unavailable, proceeding with caution"
        }


def validate_response_safety(response: str, user_context: str = "") -> Dict[str, Any]:
    """Ensure agent responses are safe and responsible"""
    
    if not response or not response.strip():
        return {"is_safe": True, "issues": []}
    
    llm = get_llm(temperature=0.1)
    
    prompt = format_prompt(
        PromptType.SAFETY_RESPONSE_VALIDATION,
        user_context=user_context,
        response=response
    )

    try:
        response_check = llm.invoke(prompt)
        content = response_check.content.strip()
        
        # Extract JSON from response
        json_match = search(r'\{.*\}', content, DOTALL)
        if json_match:
            safety_result = loads(json_match.group())
            
            # Validate required fields
            if 'is_safe' not in safety_result:
                safety_result['is_safe'] = True
            if 'issues' not in safety_result:
                safety_result['issues'] = []
                
            return safety_result

    except Exception:
        # Fail-safe: if safety check fails, allow original response
        return {"is_safe": True, "issues": [], "improved_response": ""}


def get_safety_refusal_response(concern_type: str, suggested_response: str = "") -> str:
    """Generate appropriate refusal responses for different concern types"""
    
    if suggested_response and suggested_response.strip():
        return suggested_response
    
    # Default responses based on concern type
    responses = {
        "illegal": "I can't provide assistance with illegal activities. However, I'd be happy to help you plan legal and exciting travel experiences! What destinations are you interested in?",
        
        "dangerous": "I want to keep you safe, so I can't recommend potentially dangerous activities. Let me help you discover amazing and safe travel adventures instead! Where would you like to explore?",
        
        "harmful": "I'm designed to provide helpful and positive travel assistance. Let's focus on planning an incredible trip - what kind of destinations or experiences interest you?",
        
        "off_topic": "I specialize in travel planning and would love to help you discover amazing destinations! What kind of travel experience are you looking for?",
        
        "exploitation": "I promote responsible and ethical tourism that respects local communities and environments. Let me help you plan a meaningful and sustainable travel experience! What destinations appeal to you?",
        
        "inappropriate": "I'm here to help with travel planning in a professional manner. What destinations or travel experiences would you like to explore?"
    }
    
    return responses.get(concern_type, 
        "I'm here to help you plan amazing and responsible travel experiences! What destinations interest you?")


def is_sensitive_destination(destination: str) -> bool:
    """Check if destination requires special safety considerations"""
    
    if not destination:
        return False
    
    llm = get_llm(temperature=0.1)
    
    prompt = format_prompt(
        PromptType.SAFETY_DESTINATION_ASSESSMENT,
        destination=destination
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        json_match = search(r'\{.*\}', content, DOTALL)
        if json_match:
            result = loads(json_match.group())
            return result.get('is_sensitive', False)
            
    except Exception:    
        return False  # Default to not sensitive if check fails
