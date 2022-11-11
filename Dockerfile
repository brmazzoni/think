FROM ubuntu:22.04

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    wget \
    git \
    unzip \
    tmux \
    vim \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN git clone https://github.com/brmazzoni/config.git
RUN cp config/.tmux.conf .
RUN cp config/.vimrc .
RUN pip install tensorflow keras akida akida-models jupyter pandas
RUN wget https://cernbox.cern.ch/remote.php/dav/public-files/BYSfwyJIfiQOYYo/think-master.zip
RUN unzip think-master.zip
RUN git clone https://github.com/brmazzoni/think.git
WORKDIR think
CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0"]

EXPOSE 8888
