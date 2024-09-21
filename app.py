# Import necessary libraries

import os
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
import gradio as gr
from langchain_mistralai import ChatMistralAI

# Fetch the API key from environment variables
mistral_api_key = os.getenv('MISTRALAI_API_KEY') 

# Ensure API key is available
if not mistral_api_key:
    raise ValueError("MISTRALAI_API_KEY not found in environment variables.")

def summarize_pdf(pdf_file, custom_prompt=""):
    try:
        # Load the PDF directly from the file object
        loader = PyPDFLoader(pdf_file.name)
        docs = loader.load()  # Load the document
        llm = ChatMistralAI(api_key=mistral_api_key, temperature=0, model="mistral-large-latest")
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        return summary
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    input_pdf = gr.File(label="Upload your PDF file")
    output_summary = gr.Textbox(label="Summary")
    
    interface = gr.Interface(
        fn=summarize_pdf,
        inputs=input_pdf,
        outputs=output_summary,
        title="Your PDF Summarizer",
        description="This app allows you to summarize your PDF files.",
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()