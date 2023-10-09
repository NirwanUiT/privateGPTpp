from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain.llms import HuggingFacePipeline
from torch import cuda as torch_cuda

from flask_cors import CORS

import os
import argparse
import time
import sys
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import shutil
from fileinput import filename

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

load_dotenv()

embeddings_model_name = 'all-MiniLM-L6-v2'#os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = 'data/privateGPTpp/db'#os.environ.get('PERSIST_DIRECTORY')


model_n_batch = 8
target_source_chunks = 4
model_n_ctx = 2000

# Load environment variables
#persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = r'/data/privateGPTpp/source_documents'
#embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50

from constants import CHROMA_SETTINGS

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    # Print the name of the document that is currently being processed
    for file_name in os.listdir(source_directory):
        if file_name not in ignored_files:
            print(f"Processing {file_name}")
            break
    documents = load_documents(source_directory, ignored_files)
    print(documents)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #print(text_splitter)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def ingest():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(texts)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
    
def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch_cuda.mem_get_info()[0]/(1024**2))

def calculate_layer_count() -> int | None:
    """
    Calculates the number of layers that can be used on the GPU.
    """
    is_gpu_enabled = torch_cuda.is_available()
    if not is_gpu_enabled:
        return None
    LAYER_SIZE_MB = 120.6 # This is the size of a single layer on VRAM, and is an approximation.
    # The current set value is for 7B models. For other models, this value should be changed.
    LAYERS_TO_REDUCE = 6 # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
    if (get_gpu_memory()//LAYER_SIZE_MB) - LAYERS_TO_REDUCE > 32:
        return 32
    else:
        return (get_gpu_memory()//LAYER_SIZE_MB-LAYERS_TO_REDUCE)

def call_model(query, model_type, hide_source):
    # Parse the command line arguments
    #args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
    #callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    #callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM/mnt/nas1/nba055-2/privateGPTpp/models/llama-2-7b-chat.ggmlv3.q4_0.bin
    match model_type:
        case "LlamaCpp":
            #llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
            llm = LlamaCpp(model_path=r'/data/privateGPTpp/models/llama-2-7b-chat.ggmlv3.q4_0.bin', n_ctx=model_n_ctx, verbose=False, n_gpu_layers=calculate_layer_count())
        case "GPT4All":
            #llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
            llm = GPT4All(model="data/privateGPTpp/models/ggml-gpt4all-j-v1.3-groovy.bin", backend='gptj', verbose=False)
        case "MedLlama":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPTpp/models/medllama', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case "phi":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPTpp/models/phi-1_5',task="text-generation", 
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case "codegeex2":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPTpp/models/codegeex2-6b', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case "codellama":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPTpp/models/CodeLlama-7b-hf', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case "vicuna":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPTpp/models/vicuna-7b-v1.5', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not hide_source)
    # Interactive questions and answers
    

    # Get the answer from the chain
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], [] if hide_source else res['source_documents']
    end = time.time()

    # Print the result
    '''print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)'''

    # Print the relevant sources used for the answer
    sources = []
    for document in docs:
        #print("\n> " + document.metadata["source"] + ":")
        #print(document.page_content)
        # Append source and page content to sources list
        sources.append(document.metadata["source"] + ":" + document.page_content)
        
    return answer, sources


###################################################################################################################################################

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    #return "<p>Hello, World!</p>"
    return render_template('index.html')

@app.route("/", methods=['POST'])
def hello_post():
    return "<p>Hello, World!</p>"
    

@app.route("/upload", methods=['POST'])
def upload():
    #Get the filename
    filename = request.files['file'].filename
    
    # Upload the file to the source directory
     
    file = request.files['file']#.read().decode("latin-1")
    #print(file)
    # Save the file to the source directory
    '''os.chdir(source_directory)
    with open(filename, "w") as f:
        f.write(file)'''
    file.save('/data/privateGPTpp/source_documents/' +(file.filename))
    ingest()
    
    # Return a message to the json file
    #return {'message': 'File uploaded successfully'}
    # return a message to be displayed on the "/" webpage and not the "/upload" webpage
    return redirect(url_for('hello'))
    

@app.route("/predict", methods=['POST'])
def predict():
    #text = str(request.form['text'])
    # Get the text from the json file
    data = request.get_json()
    text = data['prompt']
    #text = data['input']
    # Select model from drop down list of index.html
    model_type = data['model']
    print(text)
    print(model_type)
    
    # Check if the source directory is empty
    if not os.listdir(source_directory):
        print("Source directory is empty. Please upload a file first.")
    
    answer, sources = call_model(text, model_type, hide_source=False)
    print(sources)
    # From each of the elements in the sources list, split the string at the first colon
    sources = [source.split(":", 1) for source in sources]
    # Concatenate the sources list to a string
    sources = '\n\n'.join([source[1] for source in sources])
    # Concatenate the sources string to the answer string and add Source: to the beginning of the sources string
    answer = answer + '\n\nSources :\n\n' + sources
    # Return the answer and sources as a dict which can be read in json format in javascript
    return {'answer': answer}

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'source_documents'
    app.run(port=4000, host='0.0.0.0', debug=True)
