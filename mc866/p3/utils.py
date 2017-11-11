import numpy as np

# This function binarizes a vector
# Example:
# In: v = [1,2,3]
# Out: v = [1,0,0;
#                 0,1,0;
#                 0,0,1]
def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)


# This function sets 1 to the the maximum label and 0 for the other ones
# The input is a matrix of lables, one per row
# Example:
#         [0.1 0.3  0.9
# In:     0.3 0.01 0.2
#          0.9 0.8  0.1]
#
#         [0 0 1
# Out:  1 0 0
#          1 0 0]
def get_max_label (vin):
      [m,n] = vin.shape
      vout = np.zeros([m,n])

      mx = vin.max(axis=1)
      for i in range(m):
            for j in range (n):
                  vout[i,j] = int (mx[i] == vin[i,j])

      return vout


# This function counts the number of the miss classification by comparing the label matrix obtaineg by a classifier
# and the real label matrix
def cont_error (vreal, vclass):
      # Getting the matrix binarized
      vclass = get_max_label (vclass)
      [m,n] = vreal.shape
      dif =vreal - vclass
      err = 0

      for i in range(m):
            flag = 0
            for j in range (n):
                  if dif[i,j] != 0:
                        flag = 1

            if flag == 1:
                  err = err + 1

      return err

# This function computes the sigmoid
def sigmoid (v):
    return 1/(1+np.exp(-v))








