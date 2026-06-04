This project summaries the creation of 3 HPC nodes, 1 master and 2 worker nodes then allowing passwordless SSH communication and commmunication via MPI.
Afterwards running randoomforest ML on 2 datasets the leukemia dataset and the digits dataset.
Task 2 is about docker swarm and apache spark.
To access the digits dataset which is from the scikit library this code is used load_digits().
before applying the spark script the files need to be copied into the container utilising this code:
docker cp data_set_ALL_AML_train.csv       <spark-master-id>:/opt/spark/data/
docker cp data_set_ALL_AML_independent.csv <spark-master-id>:/opt/spark/data/
docker cp actual.csv                       <spark-master-id>:/opt/spark/data/
All the other codes and files are available in the repository.
The gene expression data was from kaggle as well.
The video link:
https://drive.google.com/file/d/1pTQZGBAdweq8jzZnkQRk_BfpcWXiSg8V/view?usp=sharing
Manar Abdelmonem Reyad   221001939
Soad Mohamed Adel        221001449
