FROM python:3.8
COPY ./src /usr/local/python/
EXPOSE 5000
WORKDIR /usr/local/python/
RUN pip install -r requirements.txt
CMD python text_cluster_api.py