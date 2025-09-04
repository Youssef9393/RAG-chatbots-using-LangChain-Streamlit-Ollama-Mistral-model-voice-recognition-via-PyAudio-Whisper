# using streamlit to create web application. 
import streamlit as st

# using PyPDF2 to extract text in pdf (extract pdf content)
from PyPDF2 import PdfReader

# using this module from langchain to transform long text to chunks
from langchain.text_splitter import CharacterTextSplitter

# using to extract embeddings of chunks get from pdf
from langchain_community.embeddings import FastEmbedEmbeddings

# my database to store vector embeddings.
from langchain.vectorstores import FAISS 

# using this model to call llms Ollama 
from langchain_community.llms import Ollama

# using this module to create prompt to chat with context and question.
from langchain_core.prompts import ChatPromptTemplate

# prepare the retrieved documents for the LLM
from langchain.chains.combine_documents import create_stuff_documents_chain

# integrates document preparation with a retrieval mechanism
from langchain.chains import create_retrieval_chain

# for speech recognition
import speech_recognition as sr

# ---- Helper functions ----
def clean_text(text):
    if not text:
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8")

def recognize_speech():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Parlez maintenant...")
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        text = r.recognize_google(audio, language="fr-FR")
        return text
    except sr.WaitTimeoutError:
        st.warning("⏱️ Aucun son détecté. Réessayez.")
    except sr.UnknownValueError:
        st.warning("❌ Je n'ai pas compris la voix.")
    except Exception as e:
        st.error(f"❌ Erreur reconnaissance vocale : {e}")
    return ""

def main():
    st.set_page_config(layout="wide")
    st.title("🤖 RAG Chatbot 100% Local avec reconnaissance vocale")

    # ---- Sidebar pour uploader PDF ----
    with st.sidebar:
        st.header("📂 Data Loader")
        st.image("rag.jpeg", width=300)
        pdf_docs = st.file_uploader(
            "Upload Your PDFs", accept_multiple_files=True, type=["pdf"]
        )

        if st.button("Submit"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF.")
                return

            with st.spinner("🔄 Processing PDFs..."):
                pdf_content = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        pdf_content += clean_text(page.extract_text() or "")

                if not pdf_content.strip():
                    st.warning("⚠️ No text found in uploaded PDFs.")
                    return

                # Découpage en chunks
                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(pdf_content)
                st.success(f"PDF loaded and split into {len(chunks)} chunks")

                # Embeddings locaux
                embeddings = FastEmbedEmbeddings()
                vector_store = FAISS.from_texts(chunks, embeddings)

                # LLM Ollama
                llm = Ollama(model="mistral", temperature=0)

                # Prompt template
                prompt = ChatPromptTemplate.from_template(
                    """
                    Answer the following question based only on the provided context.
                    Be concise and clear.

                    <context>
                      {context}
                    </context>
                    Question: {input}
                    """
                )

                # RAG chain
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                st.session_state.retrieve_chain = retrieval_chain
                st.success("✅ Local RAG Chain is ready!")

    # ---- Chatbot zone ----
    st.header("💬 Chatbot zone")

    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    # Colonnes pour input texte et bouton micro
    col_text, col_button = st.columns([8,1])

    with col_text:
        user_question = st.text_input(
            "Posez votre question:",
            value=st.session_state.user_question,
            key="input_text"
        )
        st.session_state.user_question = user_question  # synchroniser

    with col_button:
        if st.button("🎤 Parler"):
            texte_detecte = recognize_speech()
            if texte_detecte:
                st.session_state.user_question = texte_detecte
                user_question = texte_detecte

    # Envoi de la question au chatbot
    if user_question:
        if "retrieve_chain" not in st.session_state:
            st.warning("⚠️ Please upload PDFs and click Submit before asking questions.")
        else:
            try:
                response = st.session_state.retrieve_chain.invoke({"input": user_question})
                st.markdown(response["answer"], unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error while generating response: {e}")

if __name__ == "__main__":
    main()