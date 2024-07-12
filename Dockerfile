FROM python:3
WORKDIR .
COPY . .

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    libopencv-dev \
    cmake \
    git


RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]