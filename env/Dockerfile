FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.2

# config
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# dependency
RUN apt-get update && apt-get upgrade -y

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends\     
        build-essential \
        libfreetype6-dev \
        libpng-dev \
        libzmq3-dev \
        python3-dev \
        # libspatialindex-dev \
        # gdal-bin \
        # libgdal-dev \
        libsm6 \
        vim \
        git \
        curl \
        wget \
        zip \
        zsh \
        pip \
        neofetch \
        openssh-server \
        htop \
        screen \
        python3-venv \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# path
RUN mkdir /workspace
WORKDIR /workspace

# Create a virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools
COPY env/requirements.txt .
RUN pip install -r requirements.txt
ENV PYTHONPATH $PYTHONPATH:/workspace

# SSH
RUN mkdir /var/run/sshd
RUN echo 'root:sar' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication/PasswordAuthentication/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd 
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# run
ENTRYPOINT ["/usr/sbin/sshd", "-D"]
# ENTRYPOINT ["/bin/bash", "-c", "sleep infinity"]