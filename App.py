import gradio as gr

def greet(name):
    if not name:
        name = "World"
    return f"Hello, {name}!"

# Create the Gradio interface
demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(placeholder="Enter your name"),
    outputs="text",
    title="Greeting App",
    description="Enter your name and get a personalized greeting!"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()