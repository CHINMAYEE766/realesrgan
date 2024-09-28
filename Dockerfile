FROM python:3.8.20-bookworm

# Fix: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt-get update -y && apt-get -y install ffmpeg libsm6 libxext6

# Copy files from source to the container
COPY . /Real-ESRGAN

# Set root at Real-ESRGAN
WORKDIR /Real-ESRGAN

# Create the outputs folder
RUN mkdir -p outputs

# Install pip ffmpeg
RUN pip install ffmpeg-python

# Install PyTorch
RUN pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install torchvision
RUN pip install https://download.pytorch.org/whl/cu110/torchvision-0.8.0-cp38-cp38-linux_x86_64.whl

# Install basicsr
RUN pip install basicsr

# Install facexlib
RUN pip install facexlib

# Install gfpgan
RUN pip install gfpgan

# Install all remain packages
RUN pip install -r requirements.txt

# Setup develop
RUN python setup.py develop

# Clean up all cached files
RUN pip cache purge && apt-get clean autoclean && apt-get autoremove --yes && rm -rf /var/lib/{apt,dpkg,cache,log}/

ENTRYPOINT [ "python" ]
