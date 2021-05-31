# PreCA-Net
PreCA-Net : Prediction for CRISPR-Cas9 variants’ Activities on Deep Neural Network

    how to set env
    non_GPU
    if window
        conda create -n astroboi_tf_23 python=3.8 tensorflow=2.3.0 h5py=2.10.0
        conda activate astroboi_tf_23
    
    
        conda create -n astroboi_tf_2 python=3.6 tensorflow=2.1.0 h5py=2.10.0
        conda activate astroboi_tf_2
    
    elif Linux
        conda create -n astroboi_tf_22 python=3.6 tensorflow=2.2.0 h5py=2.10.0
        conda activate astroboi_tf_22
    
    with CUDA
    CUDA 10.1
    cudnn 7.6.5
    if window
        conda create -n astroboi_cuda_23 python=3.8 tensorflow-gpu=2.3.0 h5py=2.10.0
        conda activate astroboi_cuda_23
    
        conda create -n astroboi_cuda_22 python=3.6 h5py=2.10.0
        conda activate astroboi_cuda_22
        pip install tensorflow-gpu==2.2.0
    
    
    elif Linux
        conda create -n astroboi_cuda_24 python=3.8 tensorflow-gpu=2.4.1 h5py=2.10.0
        conda activate astroboi_cuda_24
    
        conda create -n astroboi_cuda_22 python=3.6 tensorflow-gpu=2.2.0 h5py=2.10.0
        conda activate astroboi_cuda_22
    
    
    conda install -c anaconda pandas=1.1.3 xlrd=1.2.0 pydot=1.4.1 pydotplus=2.0.2 scikit-learn=0.23.2
    conda install -c conda-forge matplotlib=3.3.3
