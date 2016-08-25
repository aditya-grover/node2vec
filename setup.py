#! /usr/bin/env python
#
# Copyright (C) 2016 Russell Poldrack <poldrack@stanford.edu>
# some portions borrowed from https://github.com/mwaskom/lyman/blob/master/setup.py


descr = """node2vec: algorithm for learning continuous representations for nodes in any (un)directed, (un)weighted graph"""

import os
from setuptools import setup
from sys import version

if version > '2.8.0':
    raise Exception('Currently only works in Python 2.7')

DISTNAME="node2vec"
DESCRIPTION=descr
MAINTAINER='node2vec team'
LICENSE='MIT'
URL='http://snap.stanford.edu/node2vec/'
DOWNLOAD_URL='https://github.com/aditya-grover/node2vec'
VERSION='0.1'

if __name__ == "__main__":

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    import sys

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        description=DESCRIPTION,
        include_package_data=True,
        package_data={'node2vec.tests':['emb/karate.emb','graph/karate.edgelist']},
        license=LICENSE,
        version=VERSION,
        url=URL,
        download_url=DOWNLOAD_URL,
	install_requires=['gensim','networkx'],
        packages=['node2vec'],
        scripts=['scripts/run_node2vec.py'],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'License :: OSI Approved :: BSD License',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
    )
