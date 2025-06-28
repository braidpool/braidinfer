"""
Utility functions for handling model responses.
"""

def extract_response_after_thinking(response: str) -> str:
    """
    Extract the actual response after thinking tags if present.
    
    Args:
        response: Raw model output that may contain <think>...</think> tags
        
    Returns:
        The response content after thinking tags, or original if no tags found
    """
    if "<think>" in response and "</think>" in response:
        # Find the closing tag
        think_end = response.find("</think>")
        if think_end != -1:
            # Extract everything after the closing tag
            actual_response = response[think_end + 8:].strip()
            return actual_response
    
    # If no thinking tags, return original
    return response.strip()


def clean_response(response: str, max_length: int = None) -> str:
    """
    Clean and optionally truncate a response.
    
    Args:
        response: Raw response text
        max_length: Optional maximum length for truncation
        
    Returns:
        Cleaned response
    """
    # Remove thinking tags
    cleaned = extract_response_after_thinking(response)
    
    # Truncate if requested
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    
    return cleaned