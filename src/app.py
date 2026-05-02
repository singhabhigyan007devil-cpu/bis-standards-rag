import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from rag_pipeline import load_index, get_recommendations
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="BIS Standards Finder", page_icon="🏗️")
st.title("🏗️ BIS Standards Recommendation Engine")
st.caption("AI-powered BIS standard discovery for Indian MSEs")

@st.cache_resource

def load_everything():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, metadata = load_index("data/")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return model, index, metadata, client

model, index, metadata, client = load_everything()

query = st.text_input("Describe your product:", 
                       placeholder="e.g. high strength cement for construction")

if st.button("Find Standards") and query:
    if len(query.strip()) < 5:
        st.warning("Please enter a more descriptive product query.")
    else:
        with st.spinner("Searching BIS standards..."):
            results = get_recommendations(query, model, index, metadata, client)
    
            st.success(f"Found {len(results)} relevant standards!")
            for i, r in enumerate(results, 1):
                with st.expander(f"#{i} — {r['standard']}"):
                    st.write(r['reason'])