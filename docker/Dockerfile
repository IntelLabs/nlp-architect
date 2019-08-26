
FROM ubuntu
RUN apt-get upgrade && apt-get update && apt-get install -y python3 python3-pip git
RUN export NLP_ARCHITECT_BE=CPU
RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN git clone https://github.com/NervanaSystems/nlp-architect.git
WORKDIR "nlp-architect"
RUN pip3 install -e .
RUN pip3 install -r dev-requirements.txt
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
EXPOSE 8080
