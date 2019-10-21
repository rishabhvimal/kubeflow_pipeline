FROM python:3.7.3-slim

WORKDIR /home/rishabhvimal10/random_forest/Pipelines/Training/

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python","src/training.py"]
