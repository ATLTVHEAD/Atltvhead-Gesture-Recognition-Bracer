#define base container
ARG BASE_CONTAINER=jupyter/scipy-notebook:
FROM $BASE_CONTAINER
#create some metadata for the image
LABEL maintainer = "Nate Damen <nate.damen@gmail.com>" \
      version="0.1" \
      description = " Notebooks and data for creating tensorflowlite model of arm gesture recognition from Accl and Gyro data"

#make a working directory
WORKDIR /app

COPY . /app

#mount point
VOLUME /app/data

#expose the port
EXPOSE 8888

#install tensorflow 2.2
RUN pip install --quiet --no-cache-dir \
    'tensorflow==2.2.0' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

#run notebook upon container launch
CMD ["jupyter","notebook","--ip='*'","--port=8888","--no-browser","--allow-root"]

#to build the docker file, cd to the appropriate directory and copy past the following into the terminal: docker build -t atltvhead/gesture_bracer . 
#to run the this docker file use the following:  docker run -p 8888:8888 -v "C:/Users/nated/Documents/Python Scripts/Atltvhead-Gesture-Recognition-Bracer:/app/data" --name testbracer atltvhead/gesture_bracer:latest
