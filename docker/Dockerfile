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
    && apt-get clean \
    && echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
    && wget \
        --quiet 'https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh' \
        -O /tmp/anaconda.sh \
    && /bin/bash /tmp/anaconda.sh -b -p /opt/conda \
    && rm /tmp/anaconda.sh \
    && /opt/conda/bin/conda install -y \
        libgcc \
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
RUN /opt/conda/bin/pip install --upgrade pip
RUN apt-get install -y libsm6 \
      	libgl1-mesa-glx \
	vim

ENV PATH "/opt/conda/bin:$PATH"
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"

WORKDIR bdcalc
COPY ./src/*.py /bdcalc/
COPY ./vmc_c/create_pareto_vmc_c.pyx /bdcalc/vmc_c/
COPY ./vmc_c/setup.py /bdcalc/vmc_c/
RUN cd /bdcalc/vmc_c \
    && python setup.py install

RUN apt install -y nfs-common

CMD ["bash"]
