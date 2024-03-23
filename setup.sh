pip install torch_geometric
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-2.1.0+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html

mkdir -p -- "./DATA"
unzip /content/drive/MyDrive/RESEARCH/DATA/CIFARv2Mini.zip -d ./DATA

mkdir -p -- "./GRAPHDATA"
unzip /content/drive/MyDrive/RESEARCH/DATA/CifarGraphData.zip -d ./GRAPHDATA