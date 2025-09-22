from openai import OpenAI


def chat_with_openai(prompt: str = ""):

    prompt = prompt or "Hello, how are you?"

    # Create OpenAI client instance with your API key
    # client = OpenAI(api_key="your-api-key")
    client = OpenAI()

    # Single prompt without history
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo" etc.
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Print assistant response
    print(response.choices[0].message.content)


if __name__ == '__main__':
    prompt = input('Say hello to openai!'
                   ' (Just press enter to say "Hello, how are you?") : ')
    chat_with_openai(prompt)
