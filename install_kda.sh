# install torch==2.9.1
pip install --trusted-host pypi.nvidia.cn --trusted-host pypi.nvidia.com --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# install FLA
pip install -e third_party/flash-linear-attention

# install flashla
pip install -v -e . --no-build-isolation