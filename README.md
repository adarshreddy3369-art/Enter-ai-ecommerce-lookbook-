# AI E-Commerce Lookbook

A Streamlit catalog that combines standard product filters with semantic product discovery.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Optional OpenAI configuration

Copy the example below into a local `.env` file, then replace the value with a valid OpenAI API key. Do not commit this file.

```env
OPENAI_API_KEY=sk-your-key-here
```

Without an API key, the catalog remains usable with local semantic matching. With a valid key, it uses OpenAI embeddings (`text-embedding-3-small`) for richer search results.

