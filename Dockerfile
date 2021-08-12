# Ubuntu base image
FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update

# Install setup and compile tools for python builds
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

# Install dependencies-- needs to be installed in this order
RUN pip install ez_setup
RUN easy_install -U setuptools

# Running into error installing directly from requirements.txt,
# This is due to package version and python2.7 clashes.
# Below requirements works-- needs to be installed in this order
RUN pip install future
RUN pip install decorator==3.4.0
RUN pip install networkx==1.11
RUN pip install numpy==1.11.2
RUN pip install scipy==0.8.0
RUN pip install six==1.5.0
RUN pip install smart_open==1.2.1
RUN pip install gensim==0.13.3

# To load gensim without error, remove line 7 from gensim/summarization/pagerank_weighted.py
# Caution: This breaks gensim for pagerank_weighted applications.
# But to use Word2Vec and to run node2vec code it should be fine.
RUN sed -i '7d' /usr/local/lib/python2.7/dist-packages/gensim/summarization/pagerank_weighted.py

# If commented out: Mounting node2vec project directory onto the container
# Set work directory
# #WORKDIR /usr/src/app

# Copy project
# #COPY . /usr/src/app
