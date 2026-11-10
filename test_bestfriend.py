import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "./bestfriend-qwen"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


# Load base Qwen
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32
)


# Load your trained LoRA adapter
model = PeftModel.from_pretrained(
    model,
    ADAPTER
)

model.eval()


# Conversation history
messages = [
    {
        "role": "system",
        "content": (
            "You are a close, supportive best friend. "
            "Speak naturally, casually, warmly, and honestly. "
            "Use light humor when it fits. "
            "Do not sound like customer support. "
            "Ask natural follow-up questions when useful, "
            "but do not interrogate the user."
        )
    }
]


while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    # Convert conversation into Qwen format
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]

    # Generate response
    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove the original prompt
    new_tokens = outputs[0][input_ids.shape[-1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    print("AI:", response)

    # Save response to conversation history
    messages.append({
        "role": "assistant",
        "content": response
    })
