FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

EXPOSE 5000

RUN pip install -r requirements.txt &&\ 
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html &&\
mkdir -p /app/utils/{data,model} &&\
wget YOUR_INTENTS_FILE -P /app/utils/data &&\
wget YOUR_MODEL_FILE -P /app/utils/model

ENV FLASK_APP=app.py

CMD ["flask", "app.py", "--host=0.0.0.0"]