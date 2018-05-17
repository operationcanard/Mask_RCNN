FROM gcr.io/tensorflow/tensorflow:latest-gpu

WORKDIR /workspace

ADD Docker/debian-requirements.txt .

RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && \
    DEBIAN_PACKAGES=$(egrep -v "^\s*(#|$)" debian-requirements.txt) && \
    apt-get install -y --force-yes $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm debian-requirements.txt

ADD Docker/requirements.txt .

RUN pip3 --no-cache-dir install -U \
    	setuptools \
    	virtualenv \
    	wheel \
    	six \
  && pip3 --no-cache-dir install -r requirements.txt \
  && rm requirements.txt
