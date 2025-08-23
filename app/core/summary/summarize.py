import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional


# Model options:
# - "facebook/bart-large-cnn" (good general-purpose)
# - "google/flan-t5-large" (versatile, instruction-tuned)
# - "google/pegasus-xsum" (excellent for extreme summarization)
# - "allenai/led-base-16384" (for very long documents)
# - "distilbart-cnn-12-6" (smaller, faster, less accurate)

# # Initialize the model and tokenizer
# model_name = "google/pegasus-xsum"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Create the summarization pipeline
# summarizer = pipeline(
#     "summarization",
#     model=model,
#     tokenizer=tokenizer,
#     device=0 if torch.cuda.is_available() else -1
# )

# Tone-specific instructions for T5/FLAN-T5 models - more direct for summarization
TONE_INSTRUCTIONS = {
    'neutral': 'Summarize:',
    'formal': 'Provide a formal summary:',
    'casual': 'Explain this simply:',
    'technical': 'Provide a technical overview:',
    'friendly': 'Summarize this in a friendly way:',
    'academic': 'Provide an academic summary:',
    'business': 'Provide a business summary:',
    'professional': 'Provide a professional summary:'
}

# Buffer tokens to account for special tokens and prevent truncation
BUFFER_TOKENS = 10


def summarize_text(
    content: str,
    prompt: Optional[str] = "Summarize the following text",
    tone: str = "neutral",
    temperature: float = 0.5,
    max_length: int = 150,
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
        # For T5, use simple and direct formatting
        if tone.lower() in TONE_INSTRUCTIONS:
            instruction = TONE_INSTRUCTIONS[tone.lower()]
        else:
            instruction = "Summarize:"
            
        # T5 format: "instruction content" (space, not newline)
        full_prompt = f"{instruction} {content.strip()}"
        
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
            
            # For truncation, separate instruction from content
            instruction_tokens = tokenizer.encode(f"{instruction} ", truncation=False)
            
            # Check if instruction itself is too long
            if len(instruction_tokens) >= max_input_tokens:
                raise ValueError(f"Instruction too long ({len(instruction_tokens)} tokens). Maximum: {max_input_tokens}")
            
            # Calculate available space for content
            available_content_tokens = max_input_tokens - len(instruction_tokens)
            
            # Tokenize and truncate content
            content_tokens = tokenizer.encode(content.strip(), truncation=False)
            truncated_content_tokens = content_tokens[:available_content_tokens]
            
            # Combine instruction + truncated content  
            final_tokens = instruction_tokens + truncated_content_tokens
            final_input = tokenizer.decode(final_tokens, skip_special_tokens=True)
            
            print(f"Using {len(instruction_tokens)} instruction + {len(truncated_content_tokens)} content = {len(final_tokens)} total tokens")
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


from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM

import torch
print(torch.cuda.is_available())

model_id = "google/gemma-3-1b-it"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Gemma3ForCausalLM.from_pretrained(model_id, quantization_config=quantization_config).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)


def summarize_gemma(
        content: str, 
        prompt: str="Summarize the following:", 
        tone: str="friendly", 
        temperature: float = 0.7
    ) -> str:
    """Generate a summary using the Gemma model with specified tone and temperature."""

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are a helpful assistant, reply with a {tone} tone."},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{prompt} {content}"}]
            },
        ],
    ]

    print(messages)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device).to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=32)

    outputs = tokenizer.batch_decode(outputs)
    return outputs


# Example usage and testing function
def demo_summarizer():
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
    result = summarize_gemma(sample_text)
    print(result)
    
    print("\n=== With Custom Prompt ===")
    result = summarize_gemma(sample_text, prompt="Summarize this for a high school student")
    print(result)
    
    print("\n=== Formal Tone ===")
    result = summarize_gemma(sample_text, tone="formal")
    print(result)
    
    print("\n=== Casual Tone with High Temperature ===")
    result = summarize_gemma(sample_text, tone="casual", temperature=0.9)
    print(result)
    
    print("\n=== Technical Tone ===")
    result = summarize_gemma(sample_text, tone="technical", temperature=0.7)
    print(result)


if __name__ == "__main__":
    demo_summarizer()
