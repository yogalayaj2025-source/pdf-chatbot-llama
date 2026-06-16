from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import pdfplumber
import streamlit as st
from openai import OpenAI

st.title("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

if file is not None:

    @st.cache_data
    def process_pdf(file_bytes):
        import io
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)

    chunks = process_pdf(file.read())

    def pure_python_retriever(question):
        question_words = set(question.lower().split())
        scored_chunks = []
        for c in chunks:
            score = sum(1 for word in question_words if word in c.lower())
            scored_chunks.append((score, c))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)  # ✅ fixed sort
        return "\n\n".join(c for _, c in scored_chunks[:3])

    def openrouter_llm(prompt_value):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",  # ✅ fixed URL
            api_key="sk-or-v1- ENTER YOUR API KEY HERE"
        )
        # ✅ Properly convert LangChain messages to OpenAI format
        messages = [
            {"role": "system" if msg.type == "system" else "user", "content": msg.content}
            for msg in prompt_value.messages
        ]
        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=messages
        )
        return response.choices[0].message.content

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant answering questions about a PDF document.\n"
            "Guidelines:\n"
            "1. Provide complete, well-explained answers using the context below.\n"
            "2. Include relevant details, numbers, and explanations.\n"
            "3. Only use information from the provided context.\n"
            "4. Summarize long information in bullets where needed.\n"
            "5. If the information is not in the context, say so politely.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{question}")
    ])

    chain = (
        {
            "context": RunnableLambda(pure_python_retriever),
            "question": RunnablePassthrough()
        }
        | prompt
        | RunnableLambda(openrouter_llm)
        | StrOutputParser()
    )

    user_question = st.text_input("Type your question here")

    if user_question:
        with st.spinner("LLaMA is reading your chunks and answering..."):
            try:
                response = chain.invoke(user_question)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")