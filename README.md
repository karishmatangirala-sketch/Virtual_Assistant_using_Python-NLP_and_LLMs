# Virtual Assistant using Python, NLP & LLMs 🤖

## Overview
This project is a **Python-powered virtual assistant** that combines Natural Language Processing (NLP) and Large Language Models (LLMs) to provide intelligent, context-aware conversational support and task execution.

Built as a college/final-year capstone, the assistant can understand spoken or typed input, generate responses using LLMs, and perform actions like web search, reminders, weather lookup, and more.

---

## 🎯 Key Features
- **Natural Language Conversation**: Supports voice/text input using speech recognition and text-to-speech.
- **LLM Integration**: Utilizes models like GPT‑4, LLaMA, or OpenAI API for response generation.
---

## 🛠️ Tech Stack
| Component            | Details                                      |
|---------------------|-----------------------------------------------|
| Programming Language | Python 3.x                                   |
| Speech-to-Text      | `SpeechRecognition`                          |
| Text-to-Speech      | `pyttsx3` or `gTTS`                           |
| NLP & Intent Logic  | `NLTK`, `spaCy` or custom classifier          |
| LLM Backend         | Hugging Face model                             |
| Others              | `datetime`, `webbrowser`, `os`, etc.          |

---

## 📥 Installation

1. **Clone the repository**
   bash
   git clone https://github.com/VASU-GATTE/Virtual_Assistant_using_Python-NLP_and_LLMs.git
   cd Virtual_Assistant_using_Python-NLP_and_LLMs


2. **(Optional) Set up Python virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**

   * Create a `.env` file or update `config.py` with keys such as:

     * `OPENAI_API_KEY`
     * `WEATHER_API_KEY`
     * Optional: other service keys

---

## 🚀 Usage

To launch the assistant:

```bash
python mod.py
```

or

```bash
python mod.py
```

Then:

* Use **voice input** (speak when prompted) or **type text commands**
* The assistant will process your query via NLP/LLM and respond or act
* Sure! Here's the updated section with example commands tailored to **your college context**:

---

### 🗣️ Example Commands (College-related)

* **"What is the name of my college?"**
* **"Where is DVR & Dr. HS MIC College of Technology located?"**
* **"Who is the principal of my college?"**
* **"Tell me about the CSE department in my college"**
* **"When does the next semester start?"**
* **"What are the upcoming college events?"**
* **"Show me the academic calendar"**
* **"Open the college website"**


## 📁 Project Structure

```
Virtual_Assistant_py/
├── assistant.py          # Main entry point
├── modules/              # Auxiliary modules (weather, reminders, web search, etc.)
├── nlp/                  # NLP & intent classification code
├── data/
│   └── history.jsonl     # Chat logs
├── requirements.txt
└── README.md
```

---

## 💡 Contribution

This is an individual academic project. You're welcome to suggest enhancements via GitHub issues or submit pull requests for improvements.

---

## 📄 License

Please specify your license (e.g. MIT, GPL) or note "All Rights Reserved".

---

## ☎️ Contact

Questions or feedback?

* GitHub: [Karishma](https://github.com/karishmatangirala-sketch)
* Email :karishmatangirala@gmail.com

---

## Acknowledgments

Inspired by many open-source Python virtual assistants that blend speech-processing and web APIs such as the work by **AkshatShokeen**, **ashutoshkrris**, and others working in Python + NLP assistants ([github.com][1], [toptal.com][2], [github.com][3], [github.com][4], [github.com][5]).



[1]: https://github.com/AkshatShokeen/VIRTUAL-ASSISTANT-USING-PYTHON-USING-NLP/blob/main/README.md?utm_source=chatgpt.com "VIRTUAL-ASSISTANT-USING-PYTHON-USING-NLP/README.md at main ..."
[2]: https://www.toptal.com/openai/create-your-own-ai-assistant?utm_source=chatgpt.com "Using LLMs As Virtual Assistants for Python Programming | Toptal®"
[3]: https://github.com/AkshatShokeen/VIRTUAL-ASSISTANT-USING-PYTHON-USING-NLP?utm_source=chatgpt.com "AkshatShokeen/VIRTUAL-ASSISTANT-USING-PYTHON-USING-NLP - GitHub"
[4]: https://github.com/ab1ngeorge/VIRTUAL-ASSISTANT?utm_source=chatgpt.com "GitHub - ab1ngeorge/VIRTUAL-ASSISTANT: Developing a voice assistant in ..."
[5]: https://github.com/Paulescu/virtual-assistant-llm/blob/main/README.md?utm_source=chatgpt.com "virtual-assistant-llm/README.md at main - GitHub"

