# privateGPT++
A repository that builds on the original privateGPT repository. Includes a functional webapp and a few other working models.

privateGPT is built on the repository created and maintained by imartinez. You can find the link to the original repository here: https://github.com/imartinez/privateGPT

## Function
privateGPT allows you to ask questions to your documents without needing an internet connection. It is completely secure as your data never leaves the execution environment.
Additionally we have introduced a frontend that allows you to host a simple webapp interface on your local server. 
Apart from the two models introduced in the original privateGPT repository, we have given you access to 4 other models, namely MedLlama, Microsoft phi_1.5, Vicuna 7b and CodeLlama.

The entire framework is divided into the frontend and the backend architectures. We shall go into them briefly in the following sections.

## Getting started

1. Open Visual Studio Code and connect to the server. For this, you need to install a few extensions. a) Remote-SSH b) Remote Explorer. After they are installed, press F1 and you will need to enter the details provided to you.
2. At first you will be prompted to enter your user id which should be in the form: ```ssh <host name>@vs-c2.cs.uit.no```
3. You will be asked to provide your config file.
4. This should add your host to the list of available servers. Steps 1-3 are only a first time requirement.
5. Now press F1 and you should see your host, namely "vs-c2.cs.uit.no". You will be prompted to type your password.
6. After typing your password, a new window shall open. Open a new terminal in the new window(which is basically the giving you access to the server) from the terminal tab above.
7. Enter the following code in the bash terminal: ```docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v /mnt/nas1/{your uit mail username}:/data -p 6006:6006/tcp -p 8888:8888/tcp nirwan1998/privategptpp:latest```
8. This will download the docker image onto your server. This will take some time.
9. After the image is downloaded, you will find yourself within the image environment. This is your working environment. If you make any changes to the libraries installed in the environment, make sure to commit the changes to the docker image. This can be done by: ```docker commit <container id> <your_image_name>```
10. Now you can clone this repository into your folder using ```git clone https://github.com/NirwanUiT/privateGPTpp.git```
11. Change directory into "privateGPTpp" using ```cd privateGPTpp```
12. Run the following commands<br>
   ```mkdir models```<br>
   ```cd models```<br>
   ``` ```
14. 
15. Further change directory into "deploy-llm-project" by ```cd deploy-llm-project```
16. Run ```python app.py```
17. TO CHECK IF DOCKER ALLOWS DOWNLOADING OF LFS FILES PLEASE RUN THE FOLLOWING IN THE "models" FOLDER:
    <br>
    ```git clone https://huggingface.co/microsoft/phi-1_5```
    <br>
    It should download the repository of phi_1.5 along with the .bin file which is approximately 3 GB in size.

## Backend

*Insert flowchart of workings*

The backend is built primarily on Langchain and the HuggingFace Hub.
The work done in the backend consists of primarily first processing the uploaded document/s and then ingesting them in a vector database.
