# Python base image
FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update

# setup and compile tools
RUN apt install -y build-essential
RUN apt install -y libssl-dev
RUN apt install -y zlib1g-dev
RUN apt install -y libncurses5-dev
RUN apt install -y libncursesw5-dev
RUN apt install -y libreadline-dev
RUN apt install -y libsqlite3-dev
RUN apt install -y libgdbm-dev
RUN apt install -y libdb5.3-dev
RUN apt install -y libbz2-dev
RUN apt install -y libexpat1-dev
RUN apt install -y liblzma-dev
RUN apt install -y libffi-dev
RUN apt install -y libblas-dev
RUN apt install -y liblapack-dev
RUN apt install -y libatlas-base-dev
RUN apt install -y gfortran

# Install python2 and pip
RUN apt install -y python
RUN apt install -y python-pip

# Install dependencies-- has to be in this order
RUN pip install ez_setup
RUN easy_install -U setuptools

# Running into error installing from requirements directly.
# This work:
RUN pip install future
RUN pip install decorator==3.4.0
RUN pip install networkx==1.11
RUN pip install numpy==1.11.2
RUN pip install scipy==0.8.0
RUN pip install gensim==0.13.3

# commented out for now, for testing purpose we are only mounting it
# Set work directory
# WORKDIR /usr/src/app

# Copy project
# COPY . /usr/src/app
