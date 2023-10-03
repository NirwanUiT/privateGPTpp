# privateGPT++
A repository that builds on the original privateGPT repository. Includes a functional webapp and a few other working models.

privateGPT is built on the repository created and maintained by imartinez. You can find the link to the original repository here: https://github.com/imartinez/privateGPT

## Function
privateGPT allows you to ask questions to your documents without needing an internet connection. It is completely secure as your data never leaves the execution environment.
Additionally we have introduced a frontend that allows you to host a simple webapp interface on your local server. 
Apart from the two models introduced in the original privateGPT repository, we have given you access to 4 other models, namely MedLlama, Microsoft phi_1.5, Vicuna 7b and CodeLlama.

The entire framework is divided into the frontend and the backend architectures. We shall go into them briefly in the following sections.

## Getting started

1. Open Visual Studio Code and connect to the server. It will open a new window.
2. After typing your credentials, open a new terminal from the terminal tab above.
3. Enter the following code in the bash terminal: docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v /mnt/nas1/{your uit mail username}:/{a custom new folder} -p 6006:6006/tcp -p 8888:8888/tcp nirwan1998/privategptpp:latest
4. This will download the docker image onto your server. This will take some time.
5. After the image is downloaded, you will find yourself within the image environment. This is your working environment. If you make any changes to the libraries installed in the environment, make sure to commit the changes to the docker image. This can be done by: docker commit <container id> <your_image_name>

## Backend

*Insert flowchart of workings*

The backend is built primarily on Langchain and the HuggingFace Hub.
The work done in the backend consists of primarily first processing the uploaded document/s and then ingesting them in a vector database.
