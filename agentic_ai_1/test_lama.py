import google.generativeai as genai
import os

genai.configure(
    api_key=os.getenv(
        "GEMINI_API_KEY"
    )
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

response = model.generate_content(
    "Return only JSON: {'test':1}"
)

print(response.text)