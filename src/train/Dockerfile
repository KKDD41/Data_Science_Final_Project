FROM python:3.10-alpine3.20


RUN apk update && apk upgrade
RUN apk add gcc g++ python3-dev musl-dev linux-headers

WORKDIR /usr/dsapp

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt

COPY ./src/train/train.py ./src/
COPY ./utils ./src/utils/
COPY ./sources.cfg ./src/

CMD python3 ./src/train.py