# Python base image
FROM python:2.7.18-alpine

# setup and compile tools
RUN apt install build-essential
RUN apt install libssl-dev
RUN apt install zlib1g-dev
RUN apt install libncurses5-dev
RUN apt install libncursesw5-dev
RUN apt install libreadline-dev
RUN apt install libsqlite3-dev
RUN apt install libgdbm-dev
RUN apt install libdb5.3-dev
RUN apt install libbz2-dev
RUN apt install libexpat1-dev
RUN apt install liblzma-dev
RUN apt install libffi-dev

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

COPY ./requirements.txt /usr/src/app
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app
