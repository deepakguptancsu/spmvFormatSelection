The installation steps for required packages are as given below. These installation steps are for "Ubuntu 16.04 LTS Base".

sudo apt-get install python3-pip
sudo python3.5 -m pip install numpy
sudo python3.5 -m pip install scipy
sudo python3.5 -m pip install tqdm
sudo python3.5 -m pip install xgboost
sudo python3.5 -m pip install scikit-learn

The sample input matrices are kept for testing in folder named "sparseMatrixdataSet".
If training data is required to be generated for more number of sparse matrices then those matrices shall be kept in
the same folder, without changing the name of the folder.

The training data generated for 355 sample matrices is kept in file spmvData.csv and in file spmvData_for355Matrices.csv, as a backup.

The python script can be run by using the following command:
python3.5 spmv.py

It then prompts for entering either 1 or 2.
Press 1: To create new training data/spmvData.csv
Press 2: To use existing training data/spmvData.csv

If you want to create training data for new set of matrices then press 1 and enter. This will generate a new csv file and overrite the old csv file.
If you want to use existing csv file then press 2 and enter.

The model will be trained by using the file spmvData.csv
After using five fold cross validation, the script will output the average accuracy of the model.

NOTE:
1. The current code for "calculating average number of NZ neighbors of an element" takes a lot of time. So it is currently commented.
And was not used while calculating the features of the matrices.
2. Most of the sparse matrices take a lot of space when converted to DIA format. This causes memory error and code crashes due to this.
Therefore, this code is also commented currently and was also not used while calculating the fearures of the matrices.

Code Architecture:
- Whenever spmv.py is executed, it asks for input from user. If input is 1 then it creates the training data by calling createTrainingData()
- This function reads all matrices in 'sparseMatrixdataSet' folder one by one. 
- For each matrix it calls calAttributes() to extract attributes from the given matrix.
- calAttributes() apart from extracting features from the matrix, also calls findLabel() to findout which format performs best on the given matrix.
- After calculating all features for the given matrices, these extracted features are then stored in spmvData.csv file in comma seperated format
- The data is then read from spmvData.csv file
- The read data is then divided into attributes and labels. The last column of training data corresponds to the class label, representing format suitable for the given matrix.
- This data is then divided into training and testing data by using five fold cross validation using KFold(n_splits=5)
- This training data is then fit into xgboost model
- The trained model's accuracy is then calculated by using testing data.
- Average accuracy is then calculated to get overall accuracy of the model
