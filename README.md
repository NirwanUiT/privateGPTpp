# privateGPT++
A repository that builds on the original privateGPT repository. Includes a functional webapp and a few other working models.

privateGPT is built on the repository created and maintained by imartinez. You can find the link to the original repository here: https://github.com/imartinez/privateGPT

## Function
privateGPT allows you to ask questions to your documents without needing an internet connection. It is completely secure as your data never leaves the execution environment.
Additionally we have introduced a frontend that allows you to host a simple webapp interface on your local server. 
Apart from the two models introduced in the original privateGPT repository, we have given you access to 6 different models, namely: ```opt-iml-max-1.3b, gpt-neo-1.3B, opt-2.7b, Minitron-4B-Base, h2o-danube3-4b-chat, TinyMistral-6x248M-Instruct```

The entire framework is divided into the frontend and the backend architectures. We shall go into them briefly in the following sections.

## Getting started

1. Open Visual Studio Code and connect to the server. For this, you need to install a few extensions. a) Remote-SSH b) Remote Explorer. After they are installed, press F1 and you will need to enter the details provided to you.
2. At first you will be prompted to enter your user id which should be in the form: ```ssh <host name>@vs-c2.cs.uit.no```
3. You will be asked to provide your config file.
4. This should add your host to the list of available servers. Steps 1-3 are only a first time requirement.
5. Now press F1 and you should see your host, namely "vs-c2.cs.uit.no". You will be prompted to type your password.
6. After typing your password, a new window shall open. Open a new terminal in the new window(which is basically the giving you access to the server) from the terminal tab above.
7. Enter the following code in the bash terminal: ```docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v /mnt/nas1/{your uit mail username}:/data -p 6006:6006/tcp -p 8888:8888/tcp nirwan1998/privategptpp:latest```
8. ```docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v $HOME/data:/data -p 6006:6006/tcp -
p 8888:8888/tcp nirwan1998/privategptpp:latest```
9. This will download the docker image onto your server. This will take some time.
10. After the image is downloaded, you will find yourself within the image environment. This is your working environment. If you make any changes to the libraries installed in the environment, make sure to commit the changes to the docker image. This can be done by: ```docker commit <container id> <your_image_name>```
11. Now you can clone this repository into your folder using ```git clone https://github.com/NirwanUiT/privateGPTpp.git```
12. Change directory into "privateGPTpp" using ```cd privateGPTpp```
13. Install the packages required by running ```pip install -r requirements.txt``` in the terminal.
14. Next run ```curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash```
15. And then ```apt-get install git-lfs```
16. Run the following commands<br>
   ```mkdir models```<br>
   ```cd models```<br>
   ```git clone https://huggingface.co/nvidia/Minitron-4B-Base```<br>
   This downloads the Minitron into your model directory.
17. Go back to the parent directory by ```cd ..```
18. Further change directory into "deploy-llm-project" by ```cd deploy-llm-project```
19. Run ```python app.py```
20. TO CHECK IF DOCKER ALLOWS DOWNLOADING OF LFS FILES PLEASE RUN THE FOLLOWING IN THE "models" FOLDER:
    <br>
    ```git clone https://huggingface.co/microsoft/phi-1_5```
    <br>
    It should download the repository of phi_1.5 along with the .bin file which is approximately 3 GB in size.

## Backend

![alt text](https://github.com/NirwanUiT/privateGPTpp/blob/master/Flowchart.png?raw=true)

The backend is built primarily on Langchain and the HuggingFace Hub.
Langchain is an open source framework that allows developers to combine LLMs with external sources of computation and data.
It allows you to use an LLM which has a good amount of general knowledge, but instead of using it's entire database to produce answers, we can use Langchain to direct the LLM to use the existing knowledge from a document to answer our queries. 
We will now go through parts of code in the file "app.py" to get a better understanding of the code.
The work done in the backend consists of primarily first processing the uploaded document/s and then ingesting them and creating a vectore database from which the LLM shall refer to for context.

### Loading and processing documents:
1. The functions "load_single_document", "load_documents" and "process_documents" are related to loading the uploaded documents and processing them.
2. There exists a ```source_documents``` folder in the parent directory which contains all the uploaded documents. "load_documents" loads these documents. "process_documents" processes these documents. It uses ```RecursiveCharacterTextSplitter``` to split the document into a number of chunks. The chunk size is a hyperparameter that can be set manually. It will be shown later.<br>

![alt text](https://github.com/NirwanUiT/privateGPTpp/blob/master/text_splitter.png?raw=true)

### Ingestion:
1. The function "ingest" creates vector embeddings of the text chunks using the ```HuggingFaceEmbeddings```. The embedding model can be changed. It is a hyperparameter.
2. Then we use the Chroma vectorstore from Langchain to create a database of these embeddings for generating relevant information for the model.
3. It also checks if there is an existing database, and if there is, it just appends the new embeddings to the existing database.
   
![alt text](https://github.com/NirwanUiT/privateGPTpp/blob/master/ingest.png?raw=true)

### Running inference:
1. The function "call_model" takes the query/prompt as input and calls your selected LLM model on the query to generate an output.
2. We will use a chain here. A chain in the context of Langchain is simply a pipeline that takes your LLM, prompt, relevant information, etc to predict an answer.
3. Here, we can select different kinds of models for our use case.<br>
   For Minitron - ```llm = llm = HuggingFacePipeline.from_model_id(model_id='/data/assgn_2/privateGPTpp/models/Minitron-4B-Base', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})```<br>
   The following models worked for me but you are free to try any other models<br>
   1. ```opt-iml-max-1.3b```
   2. ```gpt-neo-1.3B```
   3. ```opt-2.7b```
   4. ```Minitron-4B-Base```
   5. ```h2o-danube3-4b-chat```
   6. ```TinyMistral-6x248M-Instruct```
   <br>
   If you wish to use any other model from HuggingFaceHub, please follow the instructions below:<br>
   a. Go to the [https://huggingface.co/](https://huggingface.co/collections/adamelliotfields/small-language-models-65d8f8200733541e06582d44)<br>
   b. This website hosts a huge collection of models for LLMs and other purposes.<br>
   c. Search for any model (as an example I will be showcasing the phi_1.5 model). Click on "Clone this repository".<br>
   
   ![alt text](https://github.com/NirwanUiT/privateGPTpp/blob/master/huggingface_phi_clone.png?raw=true)<br>
   
   d. Enter the highlighted line into the terminal in your models directory.<br>
   
   ![alt text](https://github.com/NirwanUiT/privateGPTpp/blob/master/clone_huggingface_model.png?raw=true)<br>
   
   e. This should download all the necessary files into your models directory.<br>
   f. You can now use a code as shown below with the local path to your model to load your LLM using the ```HuggingFacePipeline```.
   
   ![alt text](https://github.com/NirwanUiT/privateGPTpp/blob/master/HuggingFacePipeline.png?raw=true)<br>
   
5. Since we want a question-answer system, in this example, we will be using RetrievalQA chain to chain the LLM and the retrieval database. Other chains include ExtractiveQA and GenerativeQA.<br>
6. Subsequently, we are now prepared to answer the query in context of the provided documents.

## Frontend

The frontend is created using Flask, HTML, CSS and Javascript.
It has four primary functions

## Miscellanous

In case you want to use the OpenAI backend, please access the ```openai``` folder. The key will be provided to you in class.<br>

For the purpose of hyperparameterization, you can tune the following hyperparameters:
1. The batch size: model_n_batch(8)
2. Number of chunks to be used while retrieval: target_source_chunks(4)
3. The number of tokens: model_n_ctx(2000)
4. The number of chunks for documents to be split into: chunk_size(500)
5. Specifies the number of characters that each chunk should overlap with the previous chunk. This can be useful for ensuring that important information is not missed if it spans across multiple: chunks.chunk_overlap(50)
