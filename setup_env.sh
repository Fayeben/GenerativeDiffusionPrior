# conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install tqdm h5py munch pandas #mpi4py
pip install lmdb PyYAML tensorflow-gpu==2.8 torch-fidelity lpips pytorch-ignite