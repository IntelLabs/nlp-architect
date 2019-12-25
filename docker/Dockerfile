FROM ubuntu
RUN apt-get upgrade && apt-get update && apt-get install -y python3 python3-pip git
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN pip3 install --no-cache-dir jupyter
RUN useradd -ms /bin/bash nlp_user
USER nlp_user
WORKDIR /home/nlp_user
RUN export NLP_ARCHITECT_BE=CPU
RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN git clone https://github.com/NervanaSystems/nlp-architect.git
RUN export PYTHONPATH="$PYTHONPATH:/Users/nlp_user/nlp_architect"
RUN export JUPYTER_PATH="${JUPYTER_PATH}:/Users/nlp_user/nlp_architect"
WORKDIR "/home/nlp_user/nlp-architect"
RUN pip3 install --no-cache-dir -e .
RUN pip3 install --no-cache-dir -r dev-requirements.txt
EXPOSE 8080
EXPOSE 8888