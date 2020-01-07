import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR


def linear_regression(list_data):
    df = pd.DataFrame(list_data)
    df.columns = ['one', 'sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    x_value = df[['one', 'sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].values
    x_matrix = np.matrix(np.array(x_value))
    # print(x_value)

    # print(x_matrix)
    y_value = df['class'].values
    class_values = y_value
    class_values = class_values.tolist()
    # print(y_value)
    y_matrix = np.matrix(np.array(y_value))
    # print(y_matrix)
    # print(type(y_matrix))
    # print(x_matrix)
    # print('************************')
    x_transpose = np.transpose(x_matrix)
    # print(x_transpose)
    y_matrix = np.transpose(y_matrix)
    # print(y_matrix)
    b = np.linalg.inv(x_transpose.dot(x_matrix)).dot(x_transpose).dot(y_matrix)
    # print('****************************************************')
    # print('Beta value:')
    # print(b)
    return b, x_matrix, class_values, x_value
    # x_inverse = np.linalg.inv(x)

    # print(y)
    # x_matrix = y.values
    # print(x_matrix)
    # print(list_data)
    # plt.figure(figsize=(15, 10))
    # plt.tight_layout()
    # seabornInstance.distplot(dataset['class'])
    # print('x:')
    # print(len(x_value))
    # print('y:')
    # print(len(y_value))

def classification(x_matrix,b, class_values):
    # print('****************************************************')
    y_hat = x_matrix.dot(b)

    y_hat = y_hat.tolist()
    c1 = c2 = c3 = 0
    for yh in y_hat:

        if (float(yh[0]) > 2.5):
            yh[0] = 3
            c1 = c1 + 1
        elif (float(yh[0]) > 1.5):
            yh[0] = 2
            c2 = c2 + 1
        else:
            yh[0] = 1
            c3 = c3 + 1
    # print('Prediction Values:')
    # print(y_hat)
    # print('class values for test:')
    # print(class_values)
    # print(c1)
    # print(c2)
    # print(c3)
    count = 0
    length = len(class_values)
    for i in range(length):
        if (y_hat[i][0] == class_values[i]):
            count = count + 1
    accuracy = count / 150 * 100
    # print(accuracy)
    return accuracy


if __name__ == '__main__':
    dataset = pd.read_csv('iris.csv')
    list_data = dataset.values.tolist()
    for x in list_data:
        x.insert(0, 1)
        # print(x)
        # print(x[4])
        if x[5] == 'Iris-setosa':
            x[5] = 1
        elif x[5] == 'Iris-versicolor':
            x[5] = 2
        elif x[5] == 'Iris-virginica':
            x[5] = 3
    # print(list_data)

    # *************************************************
    # Linear-regression
    # *************************************************
    print('Applying Linear Regression for the entire dataset:')
    b, x_matrix, class_values, x_value = linear_regression(list_data)
    print("Beta value is:")
    print(b)


    # *************************************************
    # Classification
    # *************************************************

    accuracy = classification(x_matrix, b, class_values)
    print("After classification of entire dataset, the accuracy is:")
    print(accuracy)

    # *************************************************
    # Cross-validation
    # *************************************************

    # prepare cross validation
    kfold = KFold(3, True, 1)
    # enumerate splits
    scores = []
    best_svr = SVR(kernel='rbf')
    dataset = dataset.sample(frac=1)
    # print(dataset)
    myarray = np.asarray(dataset)
    accuracy_list = []
    print("Applying k-fold validation using k=3:")
    for train, test in kfold.split(myarray):
        # print('train:')
        # print(myarray[train])
        # print('test:')
        # print(myarray[test])
        df_cr = myarray[train].tolist()
        df_test = myarray[test].tolist()

        # list_data = dataset.values.tolist()

        for x in df_cr:
            x.insert(0, 1)
            # print(x)
            # print(x[4])
            if x[5] == 'Iris-setosa':
                x[5] = 1
            elif x[5] == 'Iris-versicolor':
                x[5] = 2
            elif x[5] == 'Iris-virginica':
                x[5] = 3

        for x in df_test:
            x.insert(0, 1)
            # print(x)
            # print(x[4])
            if x[5] == 'Iris-setosa':
                x[5] = 1
            elif x[5] == 'Iris-versicolor':
                x[5] = 2
            elif x[5] == 'Iris-virginica':
                x[5] = 3

        b_value, x_matrix1, class_values, x_value = linear_regression(df_cr)
        print('**************************************')
        print('Beta value(weight) for iteration:')
        print(b_value)

        df_test = pd.DataFrame(df_test)
        # print('df_test')
        df_test.columns = ['one', 'sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
        x_value = df_test[['one', 'sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].values

        x_matrix = np.matrix(np.array(x_value))

        y_value = df_test['class'].values
        # print('yvalue:')
        # print(y_value)

        y_value = y_value.tolist()
        accuracy = classification(x_matrix, b, y_value)
        print("After classification for this iteration, the accuracy is:")
        print(accuracy)
        accuracy_list.append(accuracy)
    print('**************************************')
    acc_sum = 0
    # print(accuracy_list)
    for x in accuracy_list:
        acc_sum = acc_sum+x
    print('The average accuracy after cross-validation is:')
    print(acc_sum/len(accuracy_list))
    # x_train, x_test, y_train, y_test = x_value[train], x_value[test], y_value[train], y_value[test]
    # best_svr.fit(x_train, y_train)
    # scores.append(best_svr.score(x_test, y_test))

