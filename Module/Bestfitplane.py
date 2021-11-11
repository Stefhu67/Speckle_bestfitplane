import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

class bestfitplane():
    def __init__(self, CSV_file):
        self.CSV_file = CSV_file
    def findbestfitplane(self):
        df = pd.read_csv(self.CSV_file, delimiter=',', header=None, dtype=str)
        df = df.drop([0])
        df = df.astype(float)
        df = df.drop(df[df[12]==-1].index)
        Xm = df[0]
        Ym = df[1]
        Zm = df[2]
        Um = df[3]
        Vm = df[4]
        Wm = df[5]

        X = Xm+Um
        Y = Ym+Vm
        Z = Zm+Wm

        XS=X.to_numpy()
        YS=Y.to_numpy()
        ZS=Z.to_numpy()

        # do fit
        tmp_A = []
        tmp_b = []
        for i in range(len(XS)):
            tmp_A.append([XS[i], YS[i], 1])
            tmp_b.append(ZS[i])
        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)

        # Manual solution: pseudoinverse
        fit = (A.T * A).I * A.T * b
        errors = b - A * fit
        residual = np.linalg.norm(errors)
        return fit.item(0), fit.item(1), fit.item(2)


'''#read csv file
df = pd.read_csv('./Data/00000_0.csv', delimiter=',', header=None, dtype=str)

df = df.drop([0])
df = df.astype(float)
print(df)

#delete row sigma -1
df = df.drop(df[df[12]==-1].index)
 
Xm = df[0]
Ym = df[1]
Zm = df[2]
Um = df[3]
Vm = df[4]
Wm = df[5]

X = Xm+Um
Y = Ym+Vm
Z = Zm+Wm

XS=X.to_numpy()
YS=Y.to_numpy()
ZS=Z.to_numpy()

# do fit
tmp_A = []
tmp_b = []
for i in range(len(XS)):
    tmp_A.append([XS[i], YS[i], 1])
    tmp_b.append(ZS[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)

# Manual solution: pseudoinverse
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

# Or use Scipy
#from scipy.linalg import lstsq
#fit, residual, rnk, s = lstsq(A, b)

print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
print("errors: \n", errors)
print("residual:", residual)'''

'''
# plot raw data
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(XS, YS, ZS, color='b')

# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X_plane,Y_plane = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z_plane = np.zeros(X.shape)
for r in range(X_plane.shape[0]):
    for c in range(X_plane.shape[1]):
        Z_plane[r,c] = fit[0] * X_plane[r,c] + fit[1] * Y_plane[r,c] + fit[2]
ax.plot_wireframe(X_plane,Y_plane,Z_plane, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()'''