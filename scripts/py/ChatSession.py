import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


class ChatSession:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.messages: list[dict[str, str]] = []
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def add_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self.messages.append({"role": "assistant", "content": message})

    def get_chat_history(self) -> list[dict[str, str]]:
        return self.messages

    def text(self) -> str:
        return self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

    def get_embeddings(self) -> BatchEncoding:
        return self.tokenizer([self.text()], return_tensors="pt").to(self.model.device)

    def get_response(self, max_tokens: int = 32768) -> str:
        inputs = self.get_embeddings()
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def __call__(self, prompt: str) -> str:
        self.add_user_message(prompt)
        response = self.get_response()
        self.add_assistant_message(response)
        return response


def main() -> None:
    print(type(AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")))

    model_name = "Qwen/Qwen3-0.6B"
    chat_session = ChatSession(model_name)

    print("---------- Starting Chat Session ----------")
    prompt = input("Enter your starting prompt: ")
    while prompt != "kill":
        print(f"\n{chat_session(prompt)}\n")
        prompt = input("Enter message: ")


main()