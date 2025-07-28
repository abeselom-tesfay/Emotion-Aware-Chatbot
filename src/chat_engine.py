from transformers import pipeline

class ChatEngine:
    def __init__(self):
        self.generator = pipeline('text-generation', model='gpt2')

    def get_response(self, prompt: str, max_length=100) -> str:
        """
        Generate a chatbot response from the LLM based on the prompt.
        Args:
            prompt (str): The prompt text including context and user input
            max_length (int): Maximum number of tokens in the output

        Returns:
            (str) generated text response
        """
        responses = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        text = responses[0]['generated_text']

        reply = text[len(prompt):].strip()
        return reply
