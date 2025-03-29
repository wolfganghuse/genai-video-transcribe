import streamlit as st
from openai import OpenAI
import requests

from sqlalchemy import create_engine, Column, String, Float, select, Integer
from pgvector.sqlalchemy import Vector
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
from streamlit.logger import get_logger
from config import *

logger = get_logger(__name__)

client = OpenAI(api_key=api_key, base_url=base_url) #add base_url by time

# Database Setup
engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class VideoEmbedding(Base):
    __tablename__ = "video_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-increment ID
    embedding = Column(Vector(2048))  # Indexed for faster search
    initial_time = Column(Float)
    title = Column(String)
    thumbnail = Column(String)
    video_url = Column(String)
    text = Column(String)

Base.metadata.create_all(bind=engine)
    
def query_pgvector(input_text):
    session = SessionLocal()
    try:
        # Generate embedding for the input text
        embedding_response = client.embeddings.create(
            input=[input_text], model="nv-embed-1b-v2"
        )
        input_embedding = embedding_response.data[0].embedding
        
        # Query the database for the most similar embeddings
        #id, title, text, video_url
        query = (
            select(VideoEmbedding.id, VideoEmbedding.title, VideoEmbedding.text, VideoEmbedding.video_url, VideoEmbedding.embedding.cosine_distance(input_embedding).label('distance'))
            .order_by('distance')
            .limit(5)
        )
        results = session.execute(query).fetchall()
        return results
    except Exception as e:
        logger.error(f"Error querying pgvector: {e}")
        return []
    finally:
        session.close()

def rerank_results(query, results):
    rerank_api_url = base_url + "/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    documents = [{"text": result.text} for result in results]
    payload = {
        "model": "nv-rerank-1b-v2",
        "query": query,
        "documents": documents,
        "truncate": False
    }
    
    response = requests.post(rerank_api_url, headers=headers, json=payload, verify=False)
    if response.status_code == 200:
        rerank_data = response.json()["results"]
        
        # Log relevance scores and attach them in a new structure
        ranked_results = []
        for doc in rerank_data:
            result = results[doc["index"]]
            ranked_results.append({
                "result": result,  # Original result
                "relevance_score": doc["relevance_score"]  # Relevance score
            })
            print(f"Index: {doc['index']}, Relevance Score: {doc['relevance_score']}")
        return ranked_results
    else:
        print("Re-ranker API error:", response.text)
        return results  # Return original results as fallback

    
def generate_response(input_text):
    results = query_pgvector(input_text)
    results = rerank_results(input_text, results)

    context = "The following are the top video transcriptions that match your query: \n"
    references = ""
    
    for res in results:
        result = res["result"]  # Access the original result
        relevance_score = res["relevance_score"]
        context += f"Title: {result.title}\n"
        context += f"Relevance Score: {relevance_score}\n"
        context += f"Transcription: {result.text}\n"
        references += f"\n - {result.video_url}\n"


    primer = """You are Q&A bot. A highly intelligent system that answers 
    user questions based on the information provided by video transcriptions. You can use your inner knowledge,
    but consider more with emphasis the information provided. Put emphasis on the transcriptions provided. If you see titles repeated, you can assume it is the same video.
    Provide samples of the transcriptions that are important to your query.
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": context},
            {"role": "user", "content": input_text},
        ],
        model="llama-3-1-8b-inst",
    )
    response = chat_completion.choices[0].message.content
    
    response += "\n Click on the following for more information: " + references
    
    return response

logger = get_logger(__name__)

styl = """
<style>
    .element-container:has([aria-label="Select RAG mode"]) {{
      position: fixed;
      bottom: 33px;
      background: white;
      z-index: 101;
    }}
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}
    textarea[aria-label="Description"] {{
        height: 200px;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

def chat_input():
    user_input = st.chat_input("What do you want to know about your videos?")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.caption("Nutanix Video Chatbot")
            result = generate_response(user_input)
            st.session_state["user_input"].append(user_input)
            st.session_state["generated"].append(result)
            st.rerun()

def display_chat():
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = []
    
    if st.session_state["generated"]:
        size = len(st.session_state["generated"])
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state["user_input"][i])
            with st.chat_message("assistant"):
                st.caption("Nutanix Video Chatbot")
                st.write(st.session_state["generated"][i])
        with st.container():
            st.write("&nbsp;")

display_chat()
chat_input()
