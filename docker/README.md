To build a docker image with nlp-architect installed, change the active directory to this folder and run the following command:

`docker build -t nlparchitect .`

('nlparchitect' is just an example, you can change it to any image name you like)

To run the docker image:

`docker run -it -p 8080:8080 nlparchitect`

('-p 8080:8080' will allow you to access the server  visualization on - http://localhost:8080)




