#code is from Appendix C of the paper A Variational Quantum Attack for AES-like
#Symmetric Cryptography (Wang et al., 2022)
import numpy as np

def gradient_descent(x0,function, r, xerr):
    #x0: initial point (minimum expectation of the Hamiltonian from N+1 points) -- int
    #function: the function
    #r: the learning rate -- float
    count = 0
    length = len(x0)
    for ii in range(1024):
        cost = function(x0)
        count += 1
        if cost < xerr:
            break
        gd = np.zeros(length)
        for i in range(length):
            x = x0.copy()
            x[i] += 0.01
            cost_prime = function(x)
            count += 1
            gd[i] = (cost_prime - cost) / 0.01
        # Generate a random number r0 in range [0, 1]
        r0 = np.random.uniform(0, 1)
        x0 -= (r / abs(cost) + np.log(count) / count * r0) * gd
        if gd < 0.8:
            x0 = np.random.uniform(-1,1,length)
    return x0

def n_m_method(function, x0, alpha, xerr):
    N = len(x0)
    points = [x0]
    for i in range(N):
        xi = x0.copy()
        if x0[i] == 0:
            xi[i] = 0.8
        else:
            xi[i] = x0[i] * alpha
        points.append(xi)
    times = N+1
    while times<1024:
        points.sort(key = lambda x:function(x))
        if function(points[0]) < xerr:
            break
        if function(points[-1]) - function(points[1]) < 0.15:
            for i in range(N):
                xi = x0.copy()
                if x0[i] == 0:
                    xi[i] = 0.8
                else:
                    xi[i] = x0[i] * alpha
                points[i+1] = xi
            continue
        m = np.mean(points[:-1], axis=0)
        r = 2 * m - points[-1]
        times += 1
        if(function(points[0]) <= function(r) < function(points[-2])):
            points[-1] = r
            continue
        if function(r) < function(points[0]):
            s = m + 2 * (m - points[-1])
            times += 1
            if function(s) < function(r):
                points[-1] = s
                continue
            else:
                points[-1] = r
                continue
        if function(points[-2]) <= function(r) < function(points[-1]):
            c1 = m + (r - m) / 2
            times += 1
            if function(c1) < function(r):
                points[-1] = c1
                continue
            else:
                for i in range(1, N+1):
                    points[i] = x0 + (points[i] - x0) / 2.0
                times += N
                continue
        if function(points[-1]) <= function(r):
            c2 = m + (points[-1] - m) / 2.0
            times += 1
            if function(c2) < function(points[-1]):
                points[-1] = c2
                continue
            else:
                for i in range(1, N+1):
                    points[i] = x0 + (points[i] - x0) / 2.0
                times += N
                continue
    return points[0]


            
            





