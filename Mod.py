import json
import numpy as np
import torch
import os
import sys
import webbrowser
import pyttsx3
import threading
import speech_recognition as sr  # ✅ For Voice Input
import customtkinter as ctk
import re


# Sentence Transformers
from sentence_transformers import SentenceTransformer

# Transformers for Flan-T5 Large
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

###############################################################################
# USER CONFIGURATION
###############################################################################
MODEL_PATH = r"models\flan-t5-large"   # ✅ FLAN-T5 Large Model
KB_FILE = "my_data.jsonl"  # Your JSONL knowledge base
TTS_ENABLED = True  # ✅ Set to False to disable TTS if issues occur

# ✅ Global mute toggle
mute_enabled = [False]

# ✅ Predefined simple responses
SIMPLE_RESPONSES = {
    "hello": "Hello! Welcome to MIC College of Technology's Chat assistant.",
    "hi": "Hi there! How can I assist you today?",
    "good morning": "Good morning! How can I help?",
    "good night": "Good night! See you later!",
    "good evening": "Good evening! What can I do for you?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "bye": "Goodbye! Have a great day!",
    "see you later": "See you later!",
    "exit": "You may exit this chat window now.",
}

###############################################################################
# 1. LOAD THE MODEL & KNOWLEDGE BASE **ONCE**
###############################################################################
print("Loading tokenizer from local path...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

print("Loading Flan-T5 Large model from local path...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Create a pipeline for seq2seq generation
t5_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

print("Loading embeddings model (SentenceTransformer)...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

knowledge_base = []

print(f"Reading knowledge base from {KB_FILE} ...")
with open(KB_FILE, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        question = record["instruction"]
        answer = record["output"]
        q_emb = embedding_model.encode(question, convert_to_numpy=True)
        knowledge_base.append((question, answer, q_emb))

print(f"Loaded {len(knowledge_base)} knowledge entries.")

###############################################################################
# 2. RETRIEVAL FUNCTION
###############################################################################
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_docs(query, top_k=3):
    """Retrieves the best matching answer from the knowledge base."""
    for (q_text, a_text, q_emb) in knowledge_base:
        if q_text.lower().strip() == query.lower().strip():
            return [(1.0, q_text, a_text)]  # Return exact match immediately

    query_emb = embedding_model.encode(query, convert_to_numpy=True)
    scored = []
    for (q_text, a_text, q_emb) in knowledge_base:
        sim = cosine_similarity(query_emb, q_emb)
        scored.append((sim, q_text, a_text))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored[0][0] < 0.6:
        return []

    return scored[:top_k]

###############################################################################
# 3. FLAN-T5 ANSWER GENERATION
###############################################################################
def build_prompt(user_query, top_docs):
    """Build a prompt for Flan-T5 to generate a response in its own words."""
    context_str = ""
    for (_, _, a_text) in top_docs:
        context_str += f"{a_text}\n"

    prompt = (
        "You are an AI assistant that provides informative responses about MIC College. "
        "Rephrase the given knowledge in a conversational and natural way without copying it exactly.\n\n"
        f"Knowledge:\n{context_str}\n"
        f"User Question: {user_query}\n"
        "Provide a well-structured answer in your own words:"
    )
    return prompt


def answer_question(user_query, top_k=3):
    """Retrieves relevant knowledge and forces Flan-T5 to rephrase the response."""
    user_query_lower = user_query.lower().strip()

    # ✅ Extract website request (Using regex to catch different user inputs)
    website_match = re.search(r"(open|go to|launch) (.*)", user_query_lower)
    
    if website_match:
        site_name = website_match.group(2).strip()

        # ✅ Predefined websites mapping
        website_mapping = {
            "official website": "https://www.mictech.edu.in/",
            "official site": "https://www.mictech.edu.in/",
            "fee payment": "https://feepay.mictech.ac.in/",
            "student login": "http://exams.mictech.ac.in/Login.aspx",
            "feedback": "https://feedback.mictech.edu.in/studentdepartment.php",
            "contact page": "https://www.mictech.edu.in/contact"
        }

        # ✅ Check if the request matches a known site
        for key in website_mapping:
            if key in site_name:
                webbrowser.open(website_mapping[key])
                return f"Opening {key}..."

        # ✅ If no predefined match, check if the user provided a valid URL
        if "." in site_name:
            url = "https://" + site_name if not site_name.startswith("http") else site_name
            webbrowser.open(url)
            return f"Opening {site_name}..."

    # ✅ Return simple response if found (for greetings, thanks, etc.)
    if user_query_lower in SIMPLE_RESPONSES:
        return SIMPLE_RESPONSES[user_query_lower]

    top_docs = retrieve_relevant_docs(user_query, top_k=top_k)

    if not top_docs:
        return "I'm sorry, I don't have information on that."

    prompt = build_prompt(user_query, top_docs)

    result = t5_pipeline(
        prompt,
        max_new_tokens=100,
        min_new_tokens=30,
        num_beams=5,  
        do_sample=True,  
        temperature=0.7,  
        repetition_penalty=1.3  
    )

    return result[0]["generated_text"].strip()

###############################################################################
# 4. FIXED TTS FUNCTION (THREAD-SAFE) & SPEAKER BUTTON
###############################################################################
def speak_text(text):
    """Runs text-to-speech (TTS) in a separate thread to prevent blocking."""
    if TTS_ENABLED and not mute_enabled[0]:
        tts_thread = threading.Thread(target=_speak, args=(text,))
        tts_thread.start()

def _speak(text):
    """Helper function to run TTS without blocking the main UI thread."""
    tts_engine = pyttsx3.init()
    tts_engine.say(text)
    tts_engine.runAndWait()

###############################################################################
# 5. BUILD THE CHATBOT UI (WITH SPEAKER & VOICE INPUT BUTTON)
###############################################################################
def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("MIC College Chatbot with Voice Input")
    app.geometry("700x600")
    app.minsize(700, 600)

    def toggle_mute():
        mute_enabled[0] = not mute_enabled[0]
        mute_button.configure(text="Unmute" if mute_enabled[0] else "Mute")

    top_frame = ctk.CTkFrame(app)
    top_frame.pack(padx=10, pady=10, fill="x")

    welcome_label = ctk.CTkLabel(
        master=top_frame, text="MIC College Chatbot with Voice Input/Output",
        font=ctk.CTkFont(size=18, weight="bold"),
        text_color="black"
    )
    welcome_label.pack(side="left", padx=10)

    mute_button = ctk.CTkButton(
        master=top_frame, text="Mute",
        command=toggle_mute, font=ctk.CTkFont(size=14)
    )
    mute_button.pack(side="right", padx=10)

    chat_frame = ctk.CTkScrollableFrame(app, height=400)
    chat_frame.pack(padx=10, pady=(0, 10), fill="both", expand=True)

    def add_message(sender, message_text):
        bg_color = "#25D366" if sender == "User" else "#ECE5DD"
        text_color = "white" if sender == "User" else "black"
        anchor_side = "e" if sender == "User" else "w"

    # Create a message frame with padding
        message_container = ctk.CTkFrame(chat_frame, fg_color=bg_color, corner_radius=5)
        message_container.pack(anchor=anchor_side, padx=10, pady=2, fill="none")

    # Create a label with text wrapping
        label = ctk.CTkLabel(
            message_container, text=message_text, wraplength=400, justify="left",
            padx=10, pady=5, text_color=text_color
        )
        label.pack(side="left", padx=5, pady=5)

    # Adjust the container size dynamically
        message_container.update_idletasks()
        label_width = label.winfo_reqwidth()
        message_container.configure(width=label_width + 20, height=label.winfo_reqheight() + 10)

    # Add speaker button for bot responses
        if sender == "Bot":
            ctk.CTkButton(
                message_container, text="🔊", width=30, height=30, command=lambda: speak_text(message_text)
            ).pack(side="right", padx=5, pady=2)


    def ask_bot():
        """Handles user input and sends it to the chatbot."""
        user_query = user_entry.get().strip()
        if not user_query:
            return

        add_message("User", user_query)
        user_entry.delete(0, "end")

        answer = answer_question(user_query)
        add_message("Bot", answer)
        speak_text(answer)

    def on_enter(event):
        ask_bot()

    def voice_input():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            add_message("Bot", "Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                user_query = recognizer.recognize_google(audio)
                user_entry.insert(0, user_query)
                ask_bot()
            except sr.UnknownValueError:
                add_message("Bot", "Sorry, I couldn't understand that.")

    user_entry = ctk.CTkEntry(master=app, placeholder_text="Type or use voice...", width=400)
    user_entry.pack(padx=10, pady=10, fill="x")
    user_entry.bind("<Return>", on_enter)
    
    button_frame = ctk.CTkFrame(app)
    button_frame.pack(fill="x", padx=10, pady=5)

    ctk.CTkButton(button_frame, text="Ask", command=ask_bot).pack(side="left", padx=5)
    ctk.CTkButton(button_frame, text="🎤 Speak", command=voice_input).pack(side="right", padx=5)

    app.mainloop()

if __name__ == "__main__":
    main()

