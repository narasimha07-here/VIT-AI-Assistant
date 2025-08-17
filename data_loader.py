import os
from langchain.document_loaders import DirectoryLoader, TextLoader

def custom_text_loader(filepath):
    return TextLoader(file_path=filepath, encoding='utf-8', autodetect_encoding=True)

def load_documents(data_path):
    loader = DirectoryLoader(
             path=data_path,
             glob="**/*.txt",
             loader_cls=custom_text_loader,
             show_progress=True,
         )
    return loader.load()
     