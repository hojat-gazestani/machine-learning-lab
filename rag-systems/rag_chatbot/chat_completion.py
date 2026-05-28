from dotenv import load_dotenv
from openai import OpenAI
from colorama import Fore
import warnings

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()

# Create client
client = OpenAI()

print(Fore.GREEN + "Requesting the model to generate a response..." + Fore.RESET + "\n")

completion = client.chat.completions.create(
    model="OpenAI/GPT-OSS-120b",
    messages=[
        {
            "role": "system",
            "content": "You are a funny assistant, skilled in telling 2-sentence jokes about the topic given",
        },
        {"role": "user", "content": "Tell me a joke about Python"},
    ],
)

print(Fore.BLUE + completion.choices[0].message.content)
