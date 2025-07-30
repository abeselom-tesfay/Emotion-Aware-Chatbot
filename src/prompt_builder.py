def build_prompt(user_input: str, detected_emotion: str = None, followup: str = None) -> str:
    """
    Create an emotionally-aware prompt for the chatbot.
    """
    base_prompt = "You are a compassionate, helpful AI assistant designed to respond empathetically."

    if detected_emotion:
        emotion_context = f"The user appears to be feeling {detected_emotion.lower()}."
        if detected_emotion.lower() == "sad":
            emotion_context += " Respond gently and ask if they'd like to talk about it."
        elif detected_emotion.lower() == "happy":
            emotion_context += " Celebrate the moment with them!"
        elif detected_emotion.lower() == "angry":
            emotion_context += " Help them calm down and offer solutions or understanding."
    else:
        emotion_context = "The user's emotional state is unknown."

    user_message = f"User: {user_input}"
    if followup:
        user_message += f"\nFollow-up Info: {followup}"

    return f"{base_prompt}\n{emotion_context}\n{user_message}\nAssistant:"
