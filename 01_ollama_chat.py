from ollama import Client

MODEL = "qwen3:4b-instruct-2507-q4_K_M"
QUESTION = "대한민국에서 유행했던 미룬이 가사를 알려줘"

# 내 PC에서 실행 중인 Ollama에 연결합니다.
client = Client(host="http://127.0.0.1:11434", timeout=180)
print("Ollama에 질문을 보냈습니다. 답변을 기다려 주세요.")

response = client.chat(
    model=MODEL,
    messages=[{"role": "user", "content": QUESTION}],
    stream=False,
    options={"temperature": 0, "num_predict": 256},
)

print("\n[Ollama 답변]")
print(response.message.content)
