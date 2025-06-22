"""
Utility functions for Context Manager
"""

def resolve_chunk_hash(chunk_hash: str, chunks: dict) -> str:
    """
    Resolve a potentially partial chunk hash to a full hash.
    
    Args:
        chunk_hash: Full or partial hash
        chunks: Dictionary of chunk hashes
        
    Returns:
        Full chunk hash
        
    Raises:
        ValueError: If hash is ambiguous or not found
    """
    # If it looks like a full hash, return it
    if len(chunk_hash) >= 64:
        return chunk_hash
        
    # Try to find by prefix
    matches = [h for h in chunks.keys() if h.startswith(chunk_hash)]
    
    if len(matches) == 0:
        raise ValueError(f"No chunk found matching: {chunk_hash}")
    elif len(matches) > 1:
        raise ValueError(f"Ambiguous chunk hash prefix: {chunk_hash}. Matches: {', '.join(m[:16] + '...' for m in matches)}")
    
    return matches[0]