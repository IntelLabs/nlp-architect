FROM python:3.6.3-stretch

EXPOSE 8000

WORKDIR /var/lib/nlp-architect
COPY . .
RUN pip install -r requirements.txt
CMD ["/bin/bash", "-c", "gunicorn --bind 0.0.0.0:8000 --workers 3 server.serve:__hug_wsgi__"]