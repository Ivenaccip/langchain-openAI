# Retriever with Langchain and OpenAI
In this notebook I customized a vector database in Chroma with arxiv papers, in order to generate a retriever with the OpenAI API.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the libraries.

```bash
!pip install langchain pypdf openai chromadb tiktoken langchain-community langchain-openai
```

## Import the documents

```python
import requests
from langchain.document_loaders import PyPDFLoader

urls = [
    'https://arxiv.org/pdf/2306.06031v1.pdf',
    '...',
    'https://arxiv.org/pdf/2306.13643v1.pdf'
]

ml_papers = []

for i, url in enumerate(urls):
    response = requests.get(url)
    filename = f'paper{i+1}.pdf'
    with open(filename, 'wb') as f:
        f.write(response.content)
        print(f'Descargado {filename}')

        loader = PyPDFLoader(filename)
        data = loader.load()
        ml_papers.extend(data)

print('Contenido de ml_papers:')
print()
```

## Split the documents

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
    )

documents = text_splitter.split_documents(ml_papers)
```

## Embedding and vector database

```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
    )
```


## Chat and queries

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)
```

## Example

```python
query = "qué es fingpt?"
qa_chain.run(query)
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
