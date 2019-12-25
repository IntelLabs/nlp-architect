To build a docker image with nlp-architect installed, change the active directory to this folder and run the following command:

`docker build -t nlparchitect .`

('nlparchitect' is just an example, you can change it to any image name you like)

To run the docker image:

`docker run -it -p 8080:8080 nlparchitect`

To run the docker image and work with jupyter notebook:

'docker run -it -p 8888:8888 nlparchitect /bin/bash'

And then start a notebook from the container by running:
'jupyter notebook --ip=*'

In order to map data from the docker container to your host use -v option:

'docker run -it -p 8888:8888 -v /host_folder_path_to_create:/home/nlp_user/path_to_new_folder nlparchitect /bin/bash'

('-p 8080:8080' will allow you to access the server  visualization on - http://localhost:8080,
 '-p 8888:8888' will allow you to access the jupyter notebook from your local host)




