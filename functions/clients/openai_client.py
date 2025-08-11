import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
openai_client = OpenAI(api_key=OPEN_AI_KEY)