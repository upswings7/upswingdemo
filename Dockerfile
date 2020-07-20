FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python-dev
RUN pip3 install --upgrade pip
RUN apt-get install portaudio19-dev

COPY . /app
WORKDIR /app
ENV STATIC_URL /static
ENV STATIC_PATH /static
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app