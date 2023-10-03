# privateGPT++
A repository that builds on the original privateGPT repository. Includes a functional webapp and a few other working models.

privateGPT is built on the repository created and maintained by imartinez. You can find the link to the original repository here: https://github.com/imartinez/privateGPT

## Function
privateGPT allows you to ask questions to your documents without needing an internet connection. It is completely secure as your data never leaves the execution environment.
Additionally we have introduced a frontend that allows you to host a simple webapp interface on your local server. 
Apart from the two models introduced in the original privateGPT repository, we have given you access to 4 other models, namely MedLlama, Microsoft phi_1.5, Vicuna 7b and CodeLlama.

The entire framework is divided into the frontend and the backend architectures. We shall go into them briefly in the following sections.

## Backend

*Insert flowchart of workings*

The backend is built primarily on Langchain and the HuggingFace Hub.
The work consists of primarily first processing the uploaded document/s and then ingesting them in a vector database.
