FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
COPY requirements_docker.txt /opt
RUN pip install -r /opt/requirements_docker.txt \
    && pip install tensorboard \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html \
    && pip install torch_geometric==2.2.0 \
    && pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html \
    && pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
ENV PATH=$PATH:/usr/local/cuda-11.6/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64:/usr/lib/x86_64-linux-gnu

# Copy the contents of the random-walk directory
COPY . /GraphSSLScaling

# Set the working directory in the container
WORKDIR /GraphSSLScaling

CMD ["/bin/bash"]