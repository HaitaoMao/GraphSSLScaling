# Code Base for "Graph SSL Scaling"

## Quick Start
1. Install the dependencies via Dockerfile or 'pip install -r requirements.txt'
2. Execute data_scaling.sh or model_scaling.sh to use the pipeline for experiments.

## Install via Dockerfile
```Bash
git clone https://github.com/HaitaoMao/GraphSSLScaling.git ./GraphSSLScaling
cd GraphSSLScaling
docker build --no-cache --tag ssl:latest .
docker run -it --gpus all --ipc host --name ssl -v /home:/home ssl:latest bash
# upon completion, you should be at /GraphSSLScaling inside the container
```

## Install via Singularity
```Bash
git clone https://github.com/HaitaoMao/GraphSSLScaling.git ./GraphSSLScaling
cd GraphSSLScaling
singularity build ssl.sif Singularity.def
singularity shell --nv -B /home:/home ssl.sif
# upon completion, you should be at /GraphSSLScaling inside the container
```

## Instructions

### Why we build this pipeline?

1. There is a lot of parameters to set. It's not practical to set all of them manually in main function.
2. There is a lot of results to record. It's necessary to record the results with corresponding settings.

### How can this pipeline help us?

1. The **ConfigParser** will read the config file of your settings(e.g., learning rate,hidden_size) or default settings and set them for datasets or the corresponding executor of the model chosed by you.

2. The **./libgptb/log folder** will restore the config settings and the **./libgptb/cache folder** will restore the evaluate results with the exp_id. Then you can easily collect them.

### How is this pipeline organised?

The whole process is outlined in `libgptb/pipeline/pipeline.py`

In brief the whole process will be like

1. **ConfigParser** will load the config file and default config file for the **Dataset** and **Executor**
2. **Dataset** will load the data and get some features(e.g., input feature dims)
3. **Executor** will load the chosen model and the model will be trained and evaluated with stored evaluation results.


### The default config

You check the default configs for our **Executor** and **Model**. Check `libgptb/config/executors/XXXExecutor.json` and `libgptb/config/model/SSGCL/XXX.json` .

The config parameters specified by the user have the highest priority, which can be included in a json file. In other words the default config will be overwrittern.

## The execution environment

### Docerfile_x86
```dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
COPY requirements_docker.txt /opt
RUN pip install -r /opt/requirements.txt \
    && pip install tensorboard \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html \
    && pip install torch_geometric\
    && pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html \
    && pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
ENV PATH=$PATH:/usr/local/cuda-11.6/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64:/usr/lib/x86_64-linux-gnu
```


### Usage

1. You can build from source with command `docker build -f $Dockerfile path -t $YourImageName`.

3. Execute the following command in the parent path of the project.

   ```shell
   docker run -v ./GraphSSLScaling/:/GraphSSLScaling -w /GraphSSLScaling --gpus all $YourImageName /bin/bash /GraphSSLScaling/command.sh
   # -v $HOST_PATH:$Container_Path means mounting the localfile to the container. 
   # -w means setting the working dir
   # --gpus all means using all GPU
   # /bin/bash /GraphSSLScaling/command.sh means using the /bin/bash of Container to execute the customized command.sh. You could combine it with the model_scaling.sh or data_scaling.sh
   ```


###  Singularity.def file

```bash
#Bootstrap is used to specify the agent,where the base image from,here localimage means to build from a local image
Bootstrap: localimage
## This is something like 'From' in DOCKERFILE to indicate the base image
From: ./pytorch_1.13.1-cuda11.6-cudnn8-devel.sif

# %files can be used to copy files from host into the image
# like 'COPY' in DOCKERFILE
# Here we copy the requirements.txt into the image, then we can use it to install the required dependencies.
%files
    requirements_docker.txt /opt

# %post is used to build the new image
# Usage is same to shell.Here we used pip to install dependencies.
%post
    pip install -r /opt/requirements.txt
    pip install tensorboard
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
    pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html 
    pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html 

 
#% environment is used to set env_variables once the image starts
# These lines are necessary to load cuda
%environment
    export PATH=$PATH:/usr/local/cuda-11.6/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64:/usr/lib/x86_64-linux-gnu

# The usage of singularity is similar to docker
# You can use 'singularity exec --writable-tmpfs --nv --nvccli' to enable the NVIDIA gpu for executing with sigularity 
```

