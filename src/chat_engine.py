from transformers import pipeline

class ChatEngine:
    def __init__(self):
        # Use PyTorch backend explicitly to avoid Keras/TensorFlow issues
        self.generator = pipeline('text-generation', model='gpt2', framework='pt')

    def get_response(self, prompt: str, max_length=100) -> str:
        responses = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        reply = responses[0]['generated_text'][len(prompt):].strip()
        return reply
