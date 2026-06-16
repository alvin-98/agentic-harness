from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
import os

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# Define your Pydantic model
class Answer(BaseModel):
    country: str = Field(description="The country being asked about")
    capital: str = Field(description="The capital city of the country")
    fun_fact: Optional[str] = Field(description="A fun fact about the capital")

# Convert Pydantic model to JSON schema for guided_json
json_schema = Answer.model_json_schema()

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    extra_body={"nvext": {"guided_json": json_schema}},
    stream=False
)

# Parse JSON response into Pydantic model
content = completion.choices[0].message.content
print(f"Raw response: {content}")

try:
    parsed: Answer = Answer.model_validate_json(content)
    print("\n--- Parsed Result ---")
    print(f"Country  : {parsed.country}")
    print(f"Capital  : {parsed.capital}")
    print(f"Fun fact : {parsed.fun_fact}")
except Exception as e:
    print(f"\nFailed to parse response: {e}")