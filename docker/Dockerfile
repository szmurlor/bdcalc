# The base-deps Docker image installs main libraries needed to run Ray

FROM ubuntu:bionic
RUN apt-get update \
    && apt-get install -y \
        git \
        wget \
        cmake \
        build-essential \
        curl \
        unzip \
	nfs-common \
        libsm6 libgl1-mesa-glx \
	vim \
        libxt-dev \
        libgl1-mesa-glx \
        iputils-ping \
        iproute2 \
    && apt-get clean \
    && echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
    && wget \
        --quiet 'https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh' \
        -O /tmp/anaconda.sh \
    && /bin/bash /tmp/anaconda.sh -b -p /opt/conda \
    && rm /tmp/anaconda.sh \
    && /opt/conda/bin/pip install --upgrade pip \
    && /opt/conda/bin/conda install -y libgcc \
    && /opt/conda/bin/conda clean -y --all \
    && /opt/conda/bin/pip install \
        flatbuffers \
        cython==0.29.0 \
        numpy==1.15.4 \
	ray \
	matplotlib \
	vtk \
	aiohttp psutil \
	pydicom

ENV PATH "/opt/conda/bin:$PATH"
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"

WORKDIR bdcalc
COPY ./src/*.py /bdcalc/
COPY ./vmc_c/create_pareto_vmc_c.pyx /bdcalc/vmc_c/
COPY ./vmc_c/setup.py /bdcalc/vmc_c/
RUN cd /bdcalc/vmc_c \
    && python setup.py install

COPY ./docker/start-head.sh /bdcalc/
COPY ./docker/start-node.sh /bdcalc/
RUN mkdir /bdcalc/ray_session

#RUN apt-get install -y openssh-server
#RUN mkdir /var/run/sshd
#RUN echo 'root:HasloDoDoses2019' | chpasswd
#RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
#RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

#ENV NOTVISIBLE "in users profile"
#RUN echo "export VISIBLE=now" >> /etc/profile

#EXPOSE 22

CMD ["bash"]
