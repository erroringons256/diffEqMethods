import numpy as np
import pdb
np.set_printoptions(precision = 15, suppress = True)

# function of ODE y^(n)=f(y, y', y'', ..., y^(n-1))
def f(x,y):
    return - y[0] * y[0] * y[0]

# Perform an Euler iteration on all derivatives at once. (Also this is why the array gets split into 2 subarrays.) https://en.wikipedia.org/wiki/Euler_method
def eulerIter(iters, arr): # arr is a python list in this format: [x, dx, y, y^(1), y^(2), ..., y^(m-1)]
    results = []
    xStart, dx, y = np.array(arr[0]), np.array(arr[1]), np.array(arr[2:])
    x = xStart
    y = np.append(y, f(x,y))
    eulerStepArr = [y[:-1], y[1:]]
    results.append(y[0])
    i = 0
    while i < iters:
        i += 1
        x = xStart + iters * dx
        y[:-1] = eulerStepArr[0] + eulerStepArr[1] * dx
        y[-1] = f(x,y[:-1])
        eulerStepArr[0], eulerStepArr[1] = y[:-1], y[1:]
        results.append(y[0])
    return results

# Perform an RK2 iteration on all derivatives at once. https://en.wikipedia.org/wiki/Midpoint_method
def rk2Iter(iters, arr):
    results = []
    xStart, dx, y = np.array(arr[0]), np.array(arr[1]), np.array(arr[2:])
    x = xStart
    y = np.append(y, f(x,y))
    halfDx = 0.5 * dx
    halfStepArr = np.empty_like(y)
    eulerStepArr = [y[:-1], None]
    results.append(y[0])
    i = 0
    while i < iters:
        i += 1
        halfEulerStepArr = [y[:-1], y[1:]]
        halfStepX = xStart + (i - 0.5) * dx
        halfStepArr[:-1] = halfEulerStepArr[0] + halfEulerStepArr[1] * halfDx
        halfStepArr[-1] = f(halfStepX, halfStepArr[:-1])
        eulerStepArr[1] = halfStepArr[1:]
        x = xStart + i * dx
        y[:-1] = eulerStepArr[0] + eulerStepArr[1] * dx
        y[-1] = f(x,y[:-1])
        eulerStepArr[0] = y[:-1]
        results.append(y[0])
    return results

# Perform an RK4 iteration on all derivatives at once. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
def rk4Iter(iters, arr):
    results = []
    xStart, dx, y = np.array(arr[0]), np.array(arr[1]), np.array(arr[2:])
    x = xStart
    y = np.append(y, f(x,y))
    y1 = y
    y2 = np.empty_like(y)
    y3 = np.empty_like(y)
    y4 = np.empty_like(y)
    a1 = 0 # 0 * dx
    a2 = 0.5 * dx
    a3 = 0.5 * dx
    a4 = dx # 1 * dx
    b1 = b4 = 0.166666666666667
    b2 = b3 = 0.333333333333333
    results.append(y[0])
    i = 0
    while i < iters:
        i += 1
        c1 = x # x + a1
        c2 = x + a2
        c3 = x + a3
        c4 = x + a4
        x = xStart + i * dx
        y2[:-1] = y1[:-1] + y1[1:] * a2
        y2[-1] = f(y2, y2[:-1])
        y3[:-1] = y1[:-1] + y2[1:] * a3
        y3[-1] = f(y3, y3[:-1])
        y4[:-1] = y1[:-1] + y3[1:] * a4
        y4[-1] = f(y4, y4[:-1])
        y[:-1] = y[:-1] + (b1 * y1[1:] + b2 * y2[1:] + b3 * y3[1:] + b4 * y4[1:]) * dx
        y[-1] = f(x,y[:-1])
        results.append(y[0])
    return results

def rk6Iter(iters, arr):
    results = []
    xStart, dx, y = np.array(arr[0]), np.array(arr[1]), np.array(arr[2:])
    x = xStart
    y = np.append(y, f(x,y))
    y1 = y
    y2 = np.empty_like(y)
    y3 = np.empty_like(y)
    y4 = np.empty_like(y)
    y5 = np.empty_like(y)
    y6 = np.empty_like(y)
    y7 = np.empty_like(y)
    a21 = 1 / 3.0 # = a42 = c2 = c4
    # a31 = 0
    a32 = 2 / 3.0 # = c3
    a41 = 1 / 12.0
    a43 = -1 / 12.0
    a51 = 25 / 48.0
    a52 = -55 / 24.0
    a53 = 35 / 48.0
    a54 = 15 / 8.0
    a61 = 3 / 20.0
    a62 =  -11 / 20.0
    a63 = -1 / 8.0
    a64 = 1 / 2.0
    a65 = 1 / 10.0
    a71 = -261 / 260.0
    a72 = 33 / 13.0
    a73 = 43 / 156.0
    a74 = -118 / 39.0
    a75 = 32 / 195.0
    a76 = 80 / 39.0
    b1 = 13 / 200.0 # = b7
    # b2 = 0
    b3 = 11 / 40.0 # = b4
    b5 = 4 / 25.0 # = b6
    c5 = 5 / 6.0
    c6 = 2 * a41
    # c7 = 1


#for el in eulerIter(200, [0, 0.1, 1.0, 0.0]):
#    print("%.6f,"% el, end="")
#print()
#for el in eulerIter(400, [0, 0.05, 1.0, 0.0]):
#    print("%.6f,"% el, end="")
#print()
#for el in rk2Iter(200, [0, 0.1, 1.0, 0.0]):
#    print("%.6f,"% el, end="")
#print()
#for el in rk4Iter(200, [0, 0.1, 1.0, 0.0]):
#    print("%.6f,"% el, end="")
print(rk4Iter(25, [0,0.2, 1.0, 0.0, 0.0]))
print(rk2Iter(50, [0,0.1, 1.0, 0.0, 0.0]))
print(eulerIter(100, [0,0.05, 1.0, 0.0, 0.0]))