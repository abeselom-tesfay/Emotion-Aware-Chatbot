def build_prompt(user_input: str, detected_emotion: str = None) -> str:
    """
    Build a prompt for the chatbot that includes detected emotion context.
    Args:
        user_input (str): The message from the user
        detected_emotion (str): The emotion detected from the user's face

    Returns:
        (str) prompt for the language model
    """
    base_prompt = "You are a helpful, empathetic assistant."

    if detected_emotion:
        emotion_context = f"The user currently seems to be feeling {detected_emotion.lower()}."
    else:
        emotion_context = "The user's current emotion is unknown."

    full_prompt = f"{base_prompt}\n{emotion_context}\nRespond appropriately to the user input below:\nUser: {user_input}\nAssistant:"
    return full_prompt
