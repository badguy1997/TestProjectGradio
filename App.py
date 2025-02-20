import gradio as gr
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGChat:
    def __init__(self):
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def process_pdf(self, pdf_file):
        # Load PDF
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        return "PDF processed successfully! You can now ask questions about it."
    
    def get_relevant_context(self, query, k=3):
        if not self.vectorstore:
            return ""
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in docs)

    def chat(self, message, history):
        messages = []
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # Get relevant context from PDF
        context = self.get_relevant_context(message)
        
        # Create prompt with context
        prompt = f"""Context from PDF: {context}\n\nUser question: {message}\n
        Based on the context provided, please answer the question. If the context doesn't contain relevant information, 
        use your general knowledge but mention that it's not from the PDF."""
        
        messages.append({"role": "user", "content": prompt})
        
        # Make API call to Ollama
        response = requests.post('http://localhost:11434/api/chat', 
            json={
                "model": "llama3.2:3b",
                "messages": messages,
                "stream": False
            }
        )
        
        return response.json()['message']['content']

rag_chat = RAGChat()

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# PDF-based Chat with RAG")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        process_button = gr.Button("Process PDF")
    
    status_text = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.ChatInterface(
        fn=rag_chat.chat,
        title="",
        description="Ask questions about the uploaded PDF document.",
        examples=[
            "What are the main points in the document?",
            "Can you summarize the first section?",
            "What are the key findings?"
        ],
        theme=gr.themes.Soft()
    )
    
    process_button.click(
        fn=rag_chat.process_pdf,
        inputs=[pdf_input],
        outputs=[status_text]
    )

if __name__ == "__main__":
    demo.launch()