import gradio as gr
import requests

def chat_with_ollama(message, history):
    # Format the chat history
    messages = []
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add the new message
    messages.append({"role": "user", "content": message})
    
    # Make direct API call to Ollama
    response = requests.post('http://localhost:11434/api/chat', 
        json={
            "model": "llama3.2:3b",
            "messages": messages,
            "stream": False
        }
    )
    
    # Print response for debugging
    response_json = response.json()
    print("Ollama Response:", response_json)
    
    # Return the content from the message
    return response_json['message']['content']

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_ollama,
    title="Chat with Ollama",
    description="Chat with a locally hosted Ollama model. Type your message below.",
    examples=[
        "What is artificial intelligence?",
        "Write a short poem about technology",
        "Explain quantum computing in simple terms"
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()