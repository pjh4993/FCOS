pip install ninja yacs cython matplotlib tqdm
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
python setup.py build develop --no-deps
cd cocoapi/PythonAPI
python setup.py build_ext install
