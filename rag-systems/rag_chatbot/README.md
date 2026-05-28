# RAG Chatbot (LangChain + Chroma + OpenAI)

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions based on a custom document using LangChain, vector embeddings, and an LLM.

---

## How It Works
1. Loads document
+ `TextLoader("./docs/user-manual.txt")`

2. Splits text into chunks
+ `RecursiveCharacterTextSplitter`

3. Creates embeddings
+ `sentence-transformers/all-MiniLM-L6-v2`

4. Stores vectors
+ `Chroma vector database`

5. Retrieves relevant context
+ `retriever`

6. Feeds context + question into LLM
+ `ChatOpenAI(model="OpenAI/GPT-OSS-120b")`

7. Generates answer


## Run the Project

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set environment variables
```bash
cat ~/.env            
OPENAI_BASE_URL=http://local.llm/v1
OPENAI_API_KEY=<your-api-key> 
```

3. Run the chatbot
```bash
python rag_chatbot.py
```

### Example
```bash
Q: What is error E2?
```
