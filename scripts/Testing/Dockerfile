FROM python:3

WORKDIR /usr/src/app
RUN mkdir /usr/src/Data
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "./data_scraper.py"]