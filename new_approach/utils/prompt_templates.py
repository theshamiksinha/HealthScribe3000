# utils/prompt_templates.py

def create_extraction_prompt(question, answers, perspectives):
    """
    Create a prompt for the LLM to extract and summarize perspective spans
    
    Args:
        question: The question text
        answers: List of answer texts
        perspectives: Dictionary of perspective definitions
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Question: {question}\n\n"
    
    # Add perspective definitions
    prompt += "TASK: For each of the following perspectives, identify relevant spans in the provided answers and summarize them:\n\n"
    
    for perspective, definition in perspectives.items():
        prompt += f"- {perspective}: {definition['definition']}\n"
        prompt += f"  Tone: {definition['tone']}\n"
    
    prompt += "\nAnswers:\n"
    
    # Add all answers
    for i, answer in enumerate(answers):
        prompt += f"[{i+1}] {answer}\n\n"
    
    # Add output instructions
    prompt += "\nFor each perspective, provide a concise summary of the relevant information from the answers.\n"
    prompt += "Use the following format for each perspective:\n"
    
    for perspective, definition in perspectives.items():
        prompt += f"{perspective} SUMMARY: {definition.get('start_phrase', '')}\n"
    
    return prompt

def create_classification_prompt(question, answer):
    """
    Create a prompt for perspective classification
    
    Args:
        question: The question text
        answer: The answer text
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Question: {question}\n\n"
    prompt += f"Answer: {answer}\n\n"
    prompt += "Task: Identify which of the following perspectives are present in this answer:\n"
    prompt += "- INFORMATION: Knowledge about diseases, disorders, and health-related facts\n"
    prompt += "- CAUSE: Reasons responsible for medical conditions or symptoms\n"
    prompt += "- SUGGESTION: Advice or recommendations for medical decisions or health issues\n"
    prompt += "- QUESTION: Inquiries made for deeper understanding\n"
    prompt += "- EXPERIENCE: Individual experiences, anecdotes, or firsthand insights\n\n"
    prompt += "List all perspectives that apply, one per line."
    
    return prompt