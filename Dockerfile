FROM apache/spark:3.5.3

USER root

 # Instal other modules
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    psutil \
    pandas \
    tabulate

USER spark