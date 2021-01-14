FROM tensorflow/tensorflow:latest-gpu-py3


WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN pip3 install tensorflow \
    && pip3 install keras 

CMD ["python", "./linear.py"]
