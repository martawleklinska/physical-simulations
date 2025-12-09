import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def interpolate_polynomial(x, *a):
    result = 0
    for i, coeff in enumerate(a):
        result += coeff * x ** i
    return result

x_points = np.linspace(0, 15, 16)
y_points = np.array([4, 16, 22, 24, 22, 19, 20, 20, 22, 28, 35, 45, 55, 65, 75, 85])

coefficients, _ = curve_fit(interpolate_polynomial, x_points, y_points, p0=[1]*5)

plt.plot(x_points, y_points, 'o', label='data points')
plt.plot(x_points, interpolate_polynomial(x_points, *coefficients), label='polynomial fit')
plt.ylim(0, 100)
plt.legend()
# plt.show()
print(*coefficients)