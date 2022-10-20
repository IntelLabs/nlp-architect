FROM ubuntu:18.04
LABEL maintainer="Intel AI Lab NLP"

RUN apt update && \
    apt install -y bash build-essential ca-certificates python3 python3-pip git vim && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools && \
    python3 -m pip install --no-cache-dir jupyter

WORKDIR /workspace
RUN git clone https://github.com/IntelLabs/nlp-architect.git
RUN cd nlp-architect/ && \
    python3 -m pip install -e .[all,dev] && \
    python3 -m pip install -r server/requirements.txt && \
    python3 -m pip install -r examples/requirements.txt && \
    python3 -m pip install -r solutions/trend_analysis/requirements.txt && \
    python3 -m pip install -r solutions/absa_solution/requirements.txt && \
    python3 -m pip install -r solutions/set_expansion/requirements.txt
RUN python3 -m spacy download en

WORKDIR /workspace/nlp-architect
CMD ["/bin/bash"]

EXPOSE 8080
EXPOSE 8888
