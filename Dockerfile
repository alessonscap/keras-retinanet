####################################################################
# DASA-FIDI-IARA - Keras Retinanet Dockerfile
####################################################################
#############################
# General setup
#############################
FROM tensorflow/tensorflow:1.10.1-gpu-py3

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

# create root directory for our project in the container
RUN mkdir /keras-retinanet

# Copy the current directory contents into the container at /keras-retinanet
ADD . /keras-retinanet/

#############################
# Building keras-retinanet
#############################
# Set the working directory to /keras-retinanet
WORKDIR /keras-retinanet

# Builds the external dependencies
RUN python setup.py build_ext --inplace