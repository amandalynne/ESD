# run from ESD/
mkdir -p data/paths
cd data/paths

wget https://zenodo.org/record/3459420/files/part-ii-dependency-paths-chemical-disease-sorted.txt.gz
wget https://zenodo.org/record/3459420/files/part-ii-dependency-paths-chemical-gene-sorted.txt.gz
wget https://zenodo.org/record/3459420/files/part-ii-dependency-paths-gene-disease-sorted.txt.gz
wget https://zenodo.org/record/3459420/files/part-ii-dependency-paths-gene-gene-sorted.txt.gz

gunzip part-ii-dependency-paths-chemical-disease-sorted.txt.gz
gunzip part-ii-dependency-paths-chemical-gene-sorted.txt.gz
gunzip part-ii-dependency-paths-gene-disease-sorted.txt.gz
gunzip part-ii-dependency-paths-gene-gene-sorted.txt.gz

cd ..
