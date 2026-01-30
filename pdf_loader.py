from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('Apex.pdf')
docs = loader.load()
print(len(docs))