# Install dependencies / requirements
pip install torch==1.4.0 torchvision==0.4.0

pip install torch-geometric \
  torch-sparse==latest+cu101 \
  torch-scatter==latest+cu101 \
  torch-cluster==latest+cu101 \
  -f https://pytorch-geometric.com/whl/torch-1.4.0.html 
  
# Install RDKit
pip install rdkit-pypi==2021.3.1.5 wandb