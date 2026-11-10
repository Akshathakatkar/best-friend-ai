

# Best Friend AI

An AI-powered conversational companion fine-tuned to communicate in a friendly, natural, and supportive way.

## About the Project

Best Friend AI is built using **Qwen2.5-0.5B-Instruct** and fine-tuned with **LoRA (Low-Rank Adaptation)** using the Hugging Face ecosystem.

The goal is to create a conversational AI that feels more like talking to a close friend than interacting with a traditional assistant.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- LoRA
- Qwen2.5-0.5B-Instruct

## Model

The fine-tuned LoRA adapter is hosted on Hugging Face:

[Akshathakatkar/best-friend-qwen](https://huggingface.co/Akshathakatkar/best-friend-qwen)

The base model is downloaded from Hugging Face automatically when the application runs.

## Project Structure

```text
best-friend-ai/
├── test_bestfriend.py
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── fine_tuning.ipynb
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/best-friend-ai.git
cd best-friend-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Chatbot

```bash
python test_bestfriend.py
```

The first run requires an internet connection to download the base model and adapter. No additional fine-tuning is required.

## How It Works

1. Loads the Qwen2.5-0.5B-Instruct base model.
2. Downloads the fine-tuned LoRA adapter from Hugging Face.
3. Loads the tokenizer and conversation template.
4. Generates responses to user messages.
5. Maintains conversation history during the session.

## Future Improvements

- Build a web-based chatbot interface.
- Improve conversational memory.
- Enhance response quality through further fine-tuning.
- Deploy an interactive demo.

## Author

Akshatha Katkar


