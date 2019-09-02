from numpy import *
from numpy import linalg as la

df = mat(array([[1,1],[1,7]]))
U,Sigma,VT = la.svd(df)
print(U)
# [[ 0.16018224  0.98708746]
#  [ 0.98708746 -0.16018224]]
print(Sigma)
# [7.16227766 0.83772234]
print(VT)
# [[ 0.16018224  0.98708746]
#  [ 0.98708746 -0.16018224]]
