FROM continuumio/anaconda3:latest

RUN mkdir src
WORKDIR /src
COPY . /src

RUN conda install tensorflow
RUN conda install -c conda-forge librosa
RUN pip install -r requirements.txt

EXPOSE 5000

ARG SERVICE_FILE=service.py

ENV FLASK_APP=$SERVICE_FILE
ENV FLASK_DEBUG=0

ENTRYPOINT ["python", "-m", "flask", "run", "--host", "0.0.0.0"]