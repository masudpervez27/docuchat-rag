from services.config import configure_runtime
from services.llm import _get_client


def main() -> None:
    configure_runtime()
    client = _get_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": "what is the capital of France?", # Reply with exactly: Groq API works.
            }
        ],
        max_tokens=10,
        temperature=0,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()