import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt(f"../cmake-build-debug/bin/eigenvalues.csv", delimiter=",")
plt.figure()
omegas = []
for i in range(len(A[0, :])):
    omegas.append(0.01 + i * (10. - 0.01) / 100)
plt.plot(omegas, A[0, :], label="log(abs(det(A)))")
print(A.shape)
plt.plot(omegas, 30 * A[1, :] - 35, label="70 * interior nerumann eigenvalue - 80")
# plt.hlines([0.5], 0, 20)
plt.legend()
plt.show()