import openai

# Set your OpenAI API key
# openai.api_key = "your-api-key"

# Call the OpenAI chat completion API with a single prompt, no history
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",  # you can use gpt-4, gpt-3.5-turbo, or other available model
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

# Print the assistant's reply
print(response.choices[0].message.content)
