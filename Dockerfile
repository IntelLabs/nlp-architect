FROM ubuntu:18.04
LABEL maintainer="Intel AI Lab NLP [Steve's ABSA working solution]"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y bash build-essential ca-certificates python3 python3-pip git vim && \
    rm -rf /var/lib/apt/lists && \
    apt-get clean

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools

ENV LC_ALL=C.UTF-8

WORKDIR /workspace

COPY . /workspace/nlp-architect/

WORKDIR /workspace/nlp-architect/

RUN pip install -e . && \
    pip install -r solutions/absa_solution/requirements.txt

RUN python3 -m spacy download en

ENV BOKEH_ALLOW_WS_ORIGIN=localhost:5006
#ENV BOKEH_ALLOW_WS_ORIGIN=[Your desired ip]:5006

WORKDIR /workspace/nlp-architect/solutions/absa_solution/

CMD ["python3","ui.py"]

EXPOSE 5006

