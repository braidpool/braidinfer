"""
Token Manager for manual special token handling in context management mode.

This module provides direct control over special tokens without using chat templates.
"""

from typing import List, Dict, Optional
from transformers import AutoTokenizer


class TokenManager:
    """Manages special tokens for manual encoding in context manager mode."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Extract special token IDs by analyzing chat template output
        self._extract_special_tokens()
        
    def _extract_special_tokens(self):
        """Extract special token IDs from the tokenizer's chat template."""
        # Encode test messages to identify token structure
        system_msg = [{"role": "system", "content": "test"}]
        user_msg = [{"role": "user", "content": "test"}]
        assistant_msg = [{"role": "assistant", "content": "test"}]
        
        system_tokens = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(system_msg, tokenize=False, add_generation_prompt=False)
        )
        user_tokens = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=False)
        )
        assistant_tokens = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(assistant_msg, tokenize=False, add_generation_prompt=False)
        )
        
        # Extract the special tokens
        self.im_start_id = system_tokens[0]  # <|im_start|>
        self.im_end_id = system_tokens[-2]   # <|im_end|>
        self.system_id = system_tokens[1]    # system
        self.user_id = user_tokens[1]        # user
        self.assistant_id = assistant_tokens[1]  # assistant
        self.newline_id = system_tokens[2]   # \n
        
        print(f"[TokenManager] Extracted special tokens:")
        print(f"  im_start: {self.im_start_id}")
        print(f"  im_end: {self.im_end_id}")
        print(f"  system: {self.system_id}")
        print(f"  user: {self.user_id}")
        print(f"  assistant: {self.assistant_id}")
        print(f"  newline: {self.newline_id}")
    
    def encode_system_chunk(self, content: str, include_user_start: bool = True) -> List[int]:
        """
        Encode a system chunk with proper special tokens.
        
        Structure:
        <|im_start|>system
        {content}
        <|im_end|>
        <|im_start|>user    (if include_user_start is True)
        
        Args:
            content: The system message content
            include_user_start: Whether to include the user role start tokens
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        # System message start
        tokens.extend([self.im_start_id, self.system_id, self.newline_id])
        
        # Content tokens
        content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
        tokens.extend(content_tokens)
        
        # System message end
        if not tokens[-1] == self.newline_id:
            tokens.append(self.newline_id)
        tokens.extend([self.im_end_id, self.newline_id])
        
        # Start of user section if requested
        if include_user_start:
            tokens.extend([self.im_start_id, self.user_id, self.newline_id])
        
        return tokens
    
    def encode_user_chunk(self, content: str, is_first: bool = False, is_last: bool = False) -> List[int]:
        """
        Encode a user chunk for the context.
        
        For first chunk: includes user role start
        For middle chunks: just content
        For last chunk: includes user end and assistant start
        
        Args:
            content: The user content
            is_first: Whether this is the first user chunk (after system)
            is_last: Whether this is the last user chunk (before inference)
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        # If not first chunk and not continuing from system, we're in the middle
        # so just add content
        if not is_first and not is_last:
            return self.tokenizer.encode(content, add_special_tokens=False)
        
        # For first chunk (but this shouldn't happen if system chunk includes user start)
        if is_first:
            tokens.extend([self.im_start_id, self.user_id, self.newline_id])
        
        # Content tokens
        content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
        tokens.extend(content_tokens)
        
        # For last chunk, close user and start assistant
        if is_last:
            if not tokens[-1] == self.newline_id:
                tokens.append(self.newline_id)
            tokens.extend([self.im_end_id, self.newline_id])
            tokens.extend([self.im_start_id, self.assistant_id, self.newline_id])
        
        return tokens
    
    def encode_inference_prompt(self) -> List[int]:
        """
        Encode just the tokens needed to trigger assistant generation.
        
        Structure:
        <|im_end|>
        <|im_start|>assistant
        
        Returns:
            List of token IDs
        """
        return [
            self.im_end_id,
            self.newline_id,
            self.im_start_id,
            self.assistant_id,
            self.newline_id
        ]
    
    def encode_with_role(self, content: str, role: str) -> List[int]:
        """
        Encode content with specified role markers.
        
        Args:
            content: The content to encode
            role: One of 'system', 'user', or 'assistant'
            
        Returns:
            List of token IDs
        """
        role_map = {
            'system': self.system_id,
            'user': self.user_id,
            'assistant': self.assistant_id
        }
        
        if role not in role_map:
            raise ValueError(f"Unknown role: {role}")
        
        tokens = []
        tokens.extend([self.im_start_id, role_map[role], self.newline_id])
        tokens.extend(self.tokenizer.encode(content, add_special_tokens=False))
        
        if tokens[-1] != self.newline_id:
            tokens.append(self.newline_id)
        
        tokens.extend([self.im_end_id, self.newline_id])
        
        return tokens