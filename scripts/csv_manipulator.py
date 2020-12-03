import csv
from datetime import datetime, timedelta

def load_csv(x):
    csv_data = []
    with open('../Data/'+ x +'.csv', 'r', encoding='utf-8') as scraped:
        reader = csv.reader(scraped, delimiter=',')
        time = date_time()
        for row in reader:
            if row[0] == time:
                columns = [row[0], row[1],
                           row[2], row[3], row[4]]
                Open = row[1]
                High = row[2]
                Low = row[3]
                Close = row[4]
                csv_data.append(columns)
    #last_row = csv_data[-1]
    #print(csv_data)
    print(Open, High, Low, Close)

def date_time():
    yesterday = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
    return yesterday