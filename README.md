# Linear Regression- Iris dataset

Problem: To train a model using linear regression, perform 
classification using the trained model and cross validate it with 
k-fold validation for the Iris data set.

Iris DataSet:
The data set contains 3 classes of 50 instances each, where each class 
refers to a type of iris plant. The attribute to be predicted is class of iris 
plant.

Attribute Information:
1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. class: 
-- Iris Setosa 
-- Iris Versicolour 
-- Iris Virginica

Method Used:
1) Load the dataset into the program
2) Shuffle the data so that it is in the random order.
3) Using k fold validation, split the data into test and train folds.
4) Input your train data into the linear regression algorithm to 
get weights of the line equation learned.
5) Using the weights learned, validate the predictions using the 
test data and compute the accuracy for each of the iterations.
6) Compute the mean of the accuracies obtained.
