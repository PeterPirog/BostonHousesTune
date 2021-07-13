#Use existing docker image as a base
FROM  tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer="peterpirogtf@gmail.com"

#Download and install dependencies
RUN export python=python3
RUN apt-get update -y
RUN apt-get install --no-install-recommends -y libgl1-mesa-dev
RUN apt-get install --no-install-recommends -y rsync
RUN apt-get install --no-install-recommends -y apt-utils
RUN apt-get autoremove -y

#Upgrade pip and setuptools
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

RUN pip install gym[atari]
RUN pip install box2d
RUN pip install gym[box2d]
RUN pip install scikit-learn
RUN pip install feature-engine
RUN pip install category-encoders

RUN pip install modin
RUN pip install -U ray[all]
RUN ray install-nightly

RUN pip install argparse

#install Gaussian Process Framework
RUN pip install GPy

# Container start command
CMD ["/bin/bash"]


#command to build new image:
#sudo docker build -t peterpirogtf/ray_tf2:gpu .

#How to push docker image to hub

#login by:
# docker login
# docker push peterpirogtf/ray_tf2:gpu