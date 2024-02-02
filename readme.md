# Welcome to SQL Agents AI-Hub! 🤖

## Steps

* Load your data dictionary in the data folder. Csv file with the following columns or similar:

```python
['table_name','field_name','description']
```

* ```pip install -r requirements.txt```

* Copy .env.example and rename it to .env

* Fill the values (LangSmith is optional but recommended)

* Run the app with

```chainlit run agent.py```

or

```chainlit run rag_analyst.py```

* Wait for embeddings to be created and parsed into the Chroma database.

* Open the bot at ```localhost:8000``` and start asking questions!
