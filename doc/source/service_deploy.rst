.. ---------------------------------------------------------------------------
.. Copyright 2016-2018 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------
NLP Architect Server Deployment Tutorial
########################################

Overview
--------
This tutorial walks you through the multiple steps for deploying NLP Architect server locally.
Deployment allows the server to scale well based on user requests.


SW Stack
--------
Various layers of the software stack are as follows

.. image :: assets/service_deploy.png


Prerequisites
-------------
1. Ubuntu 16.04
2. Must have root privileges 
3. Virtualization must be enabled in your computer BIOS

Kubectl Installation
--------------------
kubectl is a command line interface for running commands against Kubernetes clusters.
Following are the installation instructions

.. code::

     sudo apt-get update && sudo apt-get install -y apt-transport-https
     curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
     sudo touch /etc/apt/sources.list.d/kubernetes.list 
     echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
     sudo apt-get update
     sudo apt-get install -y kubectl

Minikube Installation
---------------------
Minikube provides a simple way of running Kubernetes on your local machine.
Following are the installation instructions

.. code::

    curl -Lo minikube https://storage.googleapis.com/minikube/releases/v0.25.0/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/
    sudo minikube start --vm-driver=none
    Follow instructions posted by minikube


Docker Installation
-------------------
.. code::

   sudo apt-get install ca-certificates curl gnupg2 software-properties-common
   sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) stable"
   sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7EA0A9C3F273FCD8
   sudo apt-get update && sudo apt-get install docker-ce
   sudo usermod -aG docker $USER
   # if the following command doesn't work, re-login again
   exec su -l $USER



Launch Docker Registry
----------------------
.. code::
 
    Follow https://docs.docker.com/config/daemon/systemd/#httphttps-proxy to fix proxy things if need be
    docker run -d -p 5000:5000 --restart=always --name registry registry:2

Build DockerFile
-----------------
Create a Dockerfile with the following content and save it in your deployment directory.

.. code::

    FROM python:3.6 AS builder
    
    RUN apt-get update
    RUN apt-get install -y git
    
    ARG GITHUB_ACCESS_TOKEN
    # check out project at current location, hopefully this is a tag eventually
    # right now this is latest commit from https://github.com/NervanaSystems/nlp-architect/pull/243/commits/ at 12:21pm 8/6/18
    RUN git clone https://x-access-token:"${GITHUB_ACCESS_TOKEN}"@github.com/NervanaSystems/nlp-architect.git
    
    # prevent keeping token in final image
    FROM python:3.6
    
    COPY --from=builder /nlp-architect /src/nlp-architect
    
    ARG NLP_ARCH_VERSION=v0.3
    
    WORKDIR /src/nlp-architect
    RUN git fetch
    RUN git checkout ${NLP_ARCH_VERSION}
    
    # install nlp-architect project itself
    RUN pip3 install .

    # run NLP Architect server
    CMD [ "nlp_architect", "server", "-p", "8080"]
    
Run the following commands to build the docker file

.. code::

    docker build --build-arg GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN} --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -t nlp_architect .
    docker tag nlp_architect localhost:5000/nlp_architect
    docker push localhost:5000/nlp_architect
    docker run --rm -it -p 8080:8080 localhost:5000/nlp_architect

Deploy Kubernetes
-----------------
Create a depolyment.yaml file in the same directory as your deployment. Fill the deployment.yaml file with the following contents

.. code::

	apiVersion: extensions/v1beta1
	kind: Deployment
	metadata:
	  name: nlp-server
	spec:
	  replicas: 1
	  template:
	    metadata:
	      labels:
	        run: nlp-server
		id: "0"
		app: nlp-server
	    spec:
	      containers:
	      - name: nlp-server
	        image: localhost:5000/nlp_architect
	        imagePullPolicy: Always
	        resources:
    		  limits:
    		    cpu: 1300m
    		    memory: 1600Mi
    		  requests:
    		    cpu: 1100m
    		    memory: 1300Mi
		ports:
		- containerPort: 8080

	---

	apiVersion: v1
	kind: Service
	metadata:
	  name: nlp-server
	spec:
	  type: NodePort
	  selector:
	    app: nlp-server
	  ports:
	  - name: http
	    port: 8080
	    targetPort: 8080

	---

	apiVersion: autoscaling/v2beta1
	kind: HorizontalPodAutoscaler
	metadata:
	  name: nlp-server
	spec:
	  scaleTargetRef:
	    apiVersion: apps/v1
	    kind: Deployment
	    name: nlp-server
	  minReplicas: 3
	  maxReplicas: 10
	  metrics:
	  - type: Resource
	    resource:
	      name: cpu
	      targetAverageUtilization: 50

Run the following commands to create a deployment on the kubernetes cluster

.. code::

     kubectl create -f deployment.yaml
     # run the following command to see your pods spin up; there will be 3 of them if your machine has enough resources
     watch -n1 kubectl get pods
     # this next command gives you the {nodeportvalue} below, it'll be in the format `8080:{nodeportvalue}`
     kubectl get svc
     # this next command will show you the hpa created with this deployment
     kubectl get hpa
     # if you ever want to see everything at once, run this:
     kubectl get all
     # if there is a problem, run this:
     kubectl logs {podname}
     # if there is a problem with the deployment itself, run this:
     kubectl describe pod {podname}
     # to redeploy, run this, and then rerun the `kubectl create -f deployment.yaml` command
     kubectl delete -f deployment.yaml
     
     

To test the server 

.. code::

    curl --noproxy "*" $(sudo minikube ip):{nodeportvalue}
    Where nodeportvalue is from kubectl get svc
    
Now you can browse nlp architect at the following url: http://{operating_system_ip}:8080
    
