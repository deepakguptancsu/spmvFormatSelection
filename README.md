# Program Dynamic Optimization via Machine Learning

Scalable Approach to Sparse Matrix Format Selection

Sparse matrix vector multiplication (SpMV) is one of the most important kernel in many scientific computations and is often a major performance bottleneck. Performance of SpMV highly depends upon the format in which sparse matrix is represented. This project addresses this problem and focuses on selecting suitable SpMV format for a given sparse matrix by analyzing its features.

This selection is done by a pre-trained predictive model. Various features of a sparse matrix like number of non-zeros, number of diagonals, average number of non-zero per row, average number of non-zero per column etc., are used for training the model. For a given input sparse matrix, the model predicts a suitable format.

Implementation: Project is implemented in Python. It has two main modules-
(a)	Preprocessing – Extracts key features of sample sparse matrices for 7 important sparse matrix formats 
(b)	Training – XGBoost model is trained by using extracted features

Results: For 355 sample sparse matrices taken from SuiteSparse, the overall accuracy of the trained model by using five-fold cross validation is 84.19%

More detailed explanation can be found in spmvSelection.pdf
