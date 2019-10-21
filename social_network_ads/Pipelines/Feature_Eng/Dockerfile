FROM python:3.7.3-slim

WORKDIR /home/rishabhvimal10/social_network_ads/Pipelines/Feature_Eng/

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python","src/feature_eng.py"]