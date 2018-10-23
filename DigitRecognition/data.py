"""
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
"""

import numpy as np
import re

class Data:
    """------------------------------------------------------
    Use this to pre-process data for Neural Network training
    ------------------------------------------------------"""

    def load_csv(self, file_name, row_del, col_del, target_col=-1, numeric_target=True, ignore_cols=None):
        """
        Assumes the first row to contain header information
        i.e., column names
        :param file_name: Absolute path to the file to be loaded
        :param row_del: delimiter used to separate rows
        :param col_del: delimiter used to separate columns
        :param target_col: Optional, an integer that represents the index of the target column
        :param ignore_cols: Optional, an integer or sequence of integers that represents the index of the columns to be
        ignored
        :return:
        """
        file = open(file_name,'r')
        data = file.read()
        file.close()

        data = data.split(row_del)[1:]
        N = len(data)
        temp = data[0].split(col_del)
        M = len(temp)-1 # subtract 1 to exclude target column
        del_cols = None
        if ignore_cols is None:
            del_cols = 0
        else:
            del_cols = len(ignore_cols)

        A = np.zeros([N,M-del_cols])
        Y = []
        col_indices = np.array(range(M+1),dtype=np.int)
        col_indices = np.argwhere(col_indices != target_col)
        col_indices = np.array(list(filter(lambda x: all(x != ignore_cols),col_indices)))

        remove_rows = []
        r = re.compile("\d+\.?\d*")
        r_char = re.compile("[a-zA-Z\-\_]*")
        for i in range(N):
            if data[i] != '':
                sample = np.array(data[i].split(col_del))
                temp_list = (sample[col_indices].ravel())
                for j in range(len(temp_list)):
                    match = r.search(temp_list[j])
                    if match is None:
                        print("Debug...")
                    A[i,j] = np.float(match.group(0))

                if numeric_target:
                    match = r.search(sample[target_col])
                    Y.append(np.float(match.group(0)))
                else:
                    match = r_char.search(sample[target_col])
                    Y.append(np.str(match.group(0)))
            else:
                remove_rows.append(i)

        remove_rows = np.array(remove_rows,dtype=np.int)
        rows_indices = np.array(range(N),dtype=np.int)
        rows_indices = np.array(list(filter(lambda x: all(x != remove_rows),rows_indices)))
        A = A[rows_indices]

        Y = np.array(Y)
        return A,Y

    def partition_data(self, X, y, tr_size, shuffle=False):
        """
        Splits the data into training, and validation set
        :param X: Array must be in shape: [Samples,Features]
        :param y: Array must be in shape: [Samples]
        :param tr_size: Int that represent the size of training set
        :param shuffle: (Optional) Boolean flag that allow the data to be shuffled prior to partitioning
        :return: X_tr, y_tr, X_vl, y_vl after shuffling (if enabled), and partitioning
        """
        N, M = X.shape
        if shuffle:
            indices = np.arange(0,N)
            np.random.shuffle(indices)
            X = X[indices,:]
            y = y[indices]

        lim = tr_size
        X_tr = X[:lim, :]
        y_tr = y[:lim]
        X_vl = X[lim:, :]
        y_vl = y[lim:]
        return X_tr, y_tr, X_vl, y_vl

    def get_sample(self, X, y, size=1):
        """
        As the name suggests, this function simply returns samples
        from given data
        :param X: Array must be of shape: [Samples, Features]
        :param y: Array must be of shape: [Samples]
        :param size: (Optional) Int that controls the number of samples returned
        :return: Sample input X, Sample target y
        """
        N, M = X.shape
        indices = np.random.randint(0,N,size)
        X = X[indices,:]
        y = y[indices]
        return X, y

if __name__ == '__main__':
    data = Data()
    X,Y = data.load_csv(file_name='/Users/Rajkumar/Downloads/energydata_complete.csv', \
                        row_del='\n', \
                        col_del=',', \
                        target_col=1, \
                        numeric_target=False, \
                        ignore_cols=[0,27,28])
    print("Debug...")