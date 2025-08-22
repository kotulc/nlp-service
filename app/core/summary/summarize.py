import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional


# Model options:
# - "facebook/bart-large-cnn" (good general-purpose)
# - "google/flan-t5-large" (versatile, instruction-tuned)
# - "google/pegasus-xsum" (excellent for extreme summarization)
# - "allenai/led-base-16384" (for very long documents)
# - "distilbart-cnn-12-6" (smaller, faster, less accurate)

# Initialize the model and tokenizer
model_name = "google/flan-t5-large" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create the summarization pipeline
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Tone-specific prefixes to guide the model
TONE_PREFIXES = {
    'neutral': '',
    'formal': 'In formal language, ',
    'casual': 'In simple terms, ',
    'technical': 'From a technical perspective, ',
    'friendly': 'In a friendly manner, ',
    'academic': 'From an academic standpoint, ',
    'business': 'For business purposes, ',
    'professional': 'In a professional tone, '
}

# Buffer tokens to account for special tokens and prevent truncation
BUFFER_TOKENS = 10


def summarize_text(
    content: str,
    prompt: Optional[str] = "Summarize the following text",
    tone: str = "neutral",
    temperature: float = 0.5,
    max_length: int = 240,
    min_length: int = 16
    ) -> str:
    """
    Summarize text content with optional prompt guidance, tone control, and temperature.
    
    Args:
        content (str): The text content to summarize
        prompt (str, optional): Additional context or instruction for summarization
        tone (str): Desired tone ('neutral', 'formal', 'casual', 'technical', 'friendly')
        temperature (float): Controls randomness (0.0-1.0, higher = more creative)
        max_length (int): Maximum token length of summary
        min_length (int): Minimum token length of summary
        
    Returns:
        str: The generated summary text
    """
    
    # Validate inputs
    if not content.strip():
        raise ValueError("Content cannot be empty")
    
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("Temperature must be between 0.0 and 1.0")
    
    try:
        # Build the full input text with tone and prompt
        tone_prefix = TONE_PREFIXES.get(tone.lower(), '')
        full_prompt = f"{tone_prefix}{prompt}\n\nText to summarize:\n{content}"
        
        # Calculate maximum safe input tokens (different models use different config names)
        if hasattr(model.config, 'max_position_embeddings'):
            # BART, GPT-2, etc.
            max_model_tokens = model.config.max_position_embeddings
        elif hasattr(model.config, 'n_positions'):
            # T5, FLAN-T5, etc.
            max_model_tokens = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            # Some other models
            max_model_tokens = model.config.max_sequence_length
        else:
            # Fallback - conservative estimate
            max_model_tokens = 512
            print(f"Warning: Could not determine max tokens for {model.config.__class__.__name__}, using {max_model_tokens}")
        
        max_input_tokens = max_model_tokens - max_length - BUFFER_TOKENS
        
        # Tokenize and check if truncation is needed
        input_tokens = tokenizer.encode(full_prompt, truncation=False)
        
        if len(input_tokens) > max_input_tokens:
            print(f"Input too long ({len(input_tokens)} tokens), truncating to {max_input_tokens}")
            
            # Create the prompt header (everything before the content)
            prompt_header = f"{tone_prefix}{prompt}\n\nText to summarize:\n"
            prompt_header_tokens = tokenizer.encode(prompt_header, truncation=False)
            
            # Check if prompt header itself is too long
            if len(prompt_header_tokens) >= max_input_tokens:
                raise ValueError(f"Prompt header too long ({len(prompt_header_tokens)} tokens). Maximum: {max_input_tokens}")
            
            # Calculate available space for content
            available_content_tokens = max_input_tokens - len(prompt_header_tokens)
            
            # Tokenize just the content and truncate it
            content_tokens = tokenizer.encode(content, truncation=False)
            truncated_content_tokens = content_tokens[:available_content_tokens]
            
            # Combine prompt header + truncated content
            final_tokens = prompt_header_tokens + truncated_content_tokens
            final_input = tokenizer.decode(final_tokens, skip_special_tokens=True)
            
            print(f"Using {len(prompt_header_tokens)} prompt + {len(truncated_content_tokens)} content = {len(final_tokens)} total tokens")
        else:
            # No truncation needed
            final_input = full_prompt
        
        # Generate summary with temperature control
        summary = summarizer(
            final_input,
            max_new_tokens=max_length,
            min_length=min_length,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            num_beams=4 if temperature == 0 else 2,
            length_penalty=1.0,
            early_stopping=True
        )
        
        # Extract and return the summary
        return summary[0]['summary_text'].strip()
        
    except Exception as e:
        return f"Error during summarization: {str(e)}"


# Example usage and testing function
def test_summarizer():
    """Test the summarization function with different parameters."""

    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to 
    natural intelligence displayed by animals including humans. Leading AI textbooks define 
    the field as the study of "intelligent agents": any system that perceives its environment 
    and takes actions that maximize its chance of achieving its goals. Some popular accounts 
    use the term "artificial intelligence" to describe machines that mimic "cognitive" 
    functions that humans associate with the human mind, such as "learning" and "problem solving".
    As machines become increasingly capable, tasks considered to require "intelligence" are 
    often removed from the definition of AI, a phenomenon known as the AI effect. For instance, 
    optical character recognition is frequently excluded from things considered to be AI, 
    having become a routine technology.
    """
    
    print("=== Basic Summary ===")
    result = summarize_text(sample_text)
    print(result)
    
    print("\n=== With Custom Prompt ===")
    result = summarize_text(
        sample_text, prompt="Summarize this for a high school student"
    )
    print(result)
    
    print("\n=== Formal Tone ===")
    result = summarize_text(sample_text, tone="formal")
    print(result)
    
    print("\n=== Casual Tone with High Temperature ===")
    result = summarize_text(sample_text, tone="casual", temperature=0.9)
    print(result)

if __name__ == "__main__":
    test_summarizer()