FROM ubuntu:22.04

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    git \
    unzip \
    tmux \
    vim \
    wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN git clone https://github.com/brmazzoni/config.git
RUN cp config/.tmux.conf .
RUN cp config/.vimrc .
RUN pip install tensorflow keras akida akida-models jupyter pandas
RUN wget https://cernbox.cern.ch/remote.php/dav/public-files/BYSfwyJIfiQOYYo/think-master.zip
RUN wget https://cernbox.cern.ch/remote.php/dav/public-files/xnsdLstTrgXbPbR/archive_arthur.tar
RUN unzip think-master.zip
RUN tar -xvf archive_arthur.tar
RUN git clone https://github.com/brmazzoni/think.git
WORKDIR think
CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0"]

EXPOSE 8888
