import json
import pathlib
import re
import torch
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import get_settings


# Extract constants from settings
settings = get_settings()
DEFAULT_MODEL = settings.defaults.models.default
DEFAULT_TEMPLATE = settings.defaults.template
DEFAULT_KWARGS = settings.defaults.transformers.model_dump()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
generator = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_response(content: str, prompt: str, delimiter: str="Output:", **kwargs) -> list[str]:
    """Generate a content summary string using a specified model and prompt"""
    # Overload default arguments with user supplied arguments
    user_config = DEFAULT_KWARGS.copy()
    user_config.update(**kwargs)

    # Apply the prompt template and generate the summary
    text_prompt = DEFAULT_TEMPLATE.format(prompt=prompt, content=content, delimiter=delimiter)
    sequences = generator(text_prompt, do_sample=True, return_full_text=False, **user_config)

    # For each returned text sequence extract the generated content
    text_sequences = [sequence["generated_text"] for sequence in sequences]

    # Split the result to extract the generated summary after the prompt delimiter
    return text_sequences


def generate_summary(content: str, prompt: str, format: str=None, tone: str=None, **kwargs) -> list[str]:
    """Generate a summary with the provided prompt and parse the model output accordingly"""
    # Add a conversational tone to the supplied prompt if requested
    if tone: prompt += f" in a {tone} tone"

    # Apply the prompt template and generate the summary text
    if format:
        # Use the supplied summary type as the generated text output delimeter
        response = generate_response(content, prompt, delimiter=f"<|{format}|>:", **kwargs)
    else:
        # Use the default delimiter
        response = generate_response(content, prompt, **kwargs)
    
    # Parse and format the generated text based on known special characters:
    if format and format.lower() in ("outline", "list", "points"):
        # Split results on common list delimiters
        regex_pattern = r"[.,:;<>\[\]`|\n*-]|\*\*|--|---"
    else:
        # Simply attempt to split results into sentences or phrases
        regex_pattern = r"[.,:;<>\[\]`|\n]"

    # Return a list of extracted summary items
    parsed_list = []
    for response_string in response:
        if len(response_string) >= 2:
            substrings = re.split(regex_pattern, response_string)
            parsed_list.extend([s.strip() for s in substrings if s and re.search(r'[a-zA-Z]', s)])

    return parsed_list 


# Example usage and testing function
def demo_generator():
    """Test the summarization function with different parameters"""

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
    
    # Define generation parameters, prompt, and related arguments
    generate_kwargs = dict(max_new_tokens=32, temperature=0.7)
    sample_kwargs = [
        dict(content=sample_text, prompt="In 5 words or less, generate several concise and engaging titles", format="titles"),
        dict(content=sample_text, prompt="In as few words as possible, list several tangentially related concepts", format="list"),
        dict(content=sample_text, prompt="In as few words as possible, outline the following text", format="outline"),
        dict(content=sample_text, prompt="Provide a list of 2-5 word summaries of the following text", format="list"),
        dict(content=sample_text, prompt="Briefly summarize the following text", format="summary", tone='academic')
    ]

    print("\n== Basic Summary ===")
    print(f"Model: {model_id}")
    result = generate_summary(sample_text, "Provide a short summary of the following text", **generate_kwargs)
    print(result)

    print("\n=== With Custom Prompt ===")
    for kwargs in sample_kwargs:
        result = generate_summary(**kwargs, **generate_kwargs)
        print(kwargs.get("prompt"), result)


if __name__ == "__main__":
    demo_generator()
