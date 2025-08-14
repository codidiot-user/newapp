import streamlit as st
import google.generativeai as genai
import requests
import json
from google.api_core import exceptions

# --- Page & CSS Configuration ---
st.set_page_config(
    page_title="Gamkers AI",
    page_icon="🤖",
    layout="centered"
)

# Function to inject custom CSS
def load_css():
    st.markdown("""
    <style>
        /* General styling */
        .st-emotion-cache-1c7y2kd {
            flex-direction: row;
            align-items: center;
        }
        /* Style for user chat message */
        [data-testid="stChatMessage"]:has(div[data-testid="stTextAreaIcon-user"]) {
            background-color: #2b313e; /* A darker background for user messages */
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
        }
        /* Style for model chat message */
        [data-testid="stChatMessage"]:has(div[data-testid="stTextAreaIcon-bot"]) {
            background-color: #404656; /* A slightly lighter background for bot messages */
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- API Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("ERROR: API keys not found. Please add GOOGLE_API_KEY and SERPER_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()

# --- System Prompt & Helper Functions (mostly unchanged) ---
SYSTEM_PROMPT = """
You are "Gamkers," a professional AI assistant created by Akash M. Your persona is that of an expert ethical hacker, cloud data engineer, and an experienced Python programmer. When using search results, synthesize the information into a comprehensive answer and start by saying "Searching the web, I found that...". For all other queries, respond directly.
"""

def is_search_query(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    search_keywords = ["latest news", "current price", "who is", "what is the status of", "recent events", "today's weather", "what happened in"]
    if prompt_lower.startswith(("what is", "what are", "who is")):
        if "your name" not in prompt_lower and "your purpose" not in prompt_lower:
            return True
    for keyword in search_keywords:
        if keyword in prompt_lower:
            return True
    return False

def perform_web_search(query: str):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"An error occurred during web search: {e}"

# --- UPDATED AI function to support streaming ---
def get_ai_response_stream(history, user_prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    final_prompt_for_history = user_prompt
    searched_web = False

    if is_search_query(user_prompt):
        searched_web = True
        st.write("Performing a real-time web search...")
        search_results = perform_web_search(user_prompt)
        final_prompt_for_history = f"Based on these web search results: {json.dumps(search_results)}, provide a comprehensive answer to the user's original query: {user_prompt}"

    full_history = history + [{"role": "user", "parts": [final_prompt_for_history]}]

    generation_config = {"temperature": 0.7, "max_output_tokens": 2048}
    
    # The new part: set stream=True
    response_stream = model.generate_content(full_history, generation_config=generation_config, stream=True)
    
    # We yield the stream and whether a search was performed
    yield searched_web
    yield response_stream

# --- Streamlit App UI ---
load_css() # Apply our custom styles

st.title("Gamkers AI Assistant 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["Understood. I am Gamkers, your expert AI assistant with live web access. How can I help you today?"]}
    ]

# Display past messages, now with avatars
for message in st.session_state.messages[2:]:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["parts"][0])

user_prompt = st.chat_input("Ask about coding, recent events, or anything else...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "parts": [user_prompt]})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_prompt)

    with st.chat_message("model", avatar="🤖"):
        try:
            response_generator = get_ai_response_stream(st.session_state.messages, user_prompt)
            
            # First, get the boolean indicating if a search was performed
            searched_web = next(response_generator)
            if searched_web:
                st.info("I've used real-time web search to answer your question.", icon="💡")

            # Now, get the stream and display it
            stream = next(response_generator)
            # Use st.write_stream to display the "typing" effect
            full_response = st.write_stream(stream)
            
            # Add the complete response to history once it's finished streaming
            st.session_state.messages.append({"role": "model", "parts": [full_response]})
        
        except exceptions.ResourceExhausted as e:
            st.error("I'm receiving too many requests right now. Please wait a moment and try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred. Please try again. Details: {e}")
