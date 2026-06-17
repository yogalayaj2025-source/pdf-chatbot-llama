from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import pdfplumber
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import io
import os

load_dotenv()

# UI
st.title("PDF Chatbot using LLaMA 3 (RAG System)")

with st.sidebar:
    st.title("Document Upload")
    file = st.file_uploader("Upload a PDF file to begin", type="pdf")

# PDF Processing
if file is not None:
    st.success("Document uploaded successfully")

    @st.cache_data
    def process_pdf(file_bytes):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_text(text)

    chunks = process_pdf(file.read())
    st.info("Document processed and ready for questions")

    # Retrieval
    def pure_python_retriever(question):
        question_words = set(question.lower().split())
        scored_chunks = []

        for c in chunks:
            score = sum(1 for word in question_words if word in c.lower())
            scored_chunks.append((score, c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return "\n\n".join(c for _, c in scored_chunks[:3])

    # LLM Call
    def openrouter_llm(prompt_value):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        messages = [
            {
                "role": "system" if msg.type == "system" else "user",
                "content": msg.content
            }
            for msg in prompt_value.messages
        ]

        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=messages
        )

        return response.choices[0].message.content

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions based on a PDF document.\n\n"
         "Rules:\n"
         "- Use only the provided context\n"
         "- Be clear and structured\n"
         "- Include key details when relevant\n"
         "- If answer is not in context, say you cannot find it\n\n"
         "Context:\n{context}"
         ),
        ("human", "{question}")
    ])

    # RAG Chain
    chain = (
        {
            "context": RunnableLambda(pure_python_retriever),
            "question": RunnablePassthrough()
        }
        | prompt
        | RunnableLambda(openrouter_llm)
        | StrOutputParser()
    )

    # UI Input
    user_question = st.text_input("Ask a question from the document")

    if user_question:
        with st.spinner("Generating answer..."):
            try:
                response = chain.invoke(user_question)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")