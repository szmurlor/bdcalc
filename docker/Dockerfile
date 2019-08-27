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
        --quiet 'https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh' \
        -O /tmp/anaconda.sh \
    && /bin/bash /tmp/anaconda.sh -b -p /opt/conda \
    && rm /tmp/anaconda.sh \
    && /opt/conda/bin/pip install --upgrade pip \
    && /opt/conda/bin/conda install -y libgcc \
    && /opt/conda/bin/conda clean -y --all \
    && /opt/conda/bin/pip install \
        flatbuffers \
        cython \
        numpy \
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

COPY ./docker/docker-cluster.py /bdcalc/

CMD ["bash"]
