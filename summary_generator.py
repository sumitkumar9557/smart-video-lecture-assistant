import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_summary(transcript_text):
    max_chars = 12000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars]

    prompt = f"""Create a topic-wise summary of the lecture transcript below (in bullet points):

Transcript:
{transcript_text}

Summary:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    transcript_path =