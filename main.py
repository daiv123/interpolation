import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

from torch import linspace

# Langrange interpolation 
def langrange_alpha(x,y):
    n = len(x)
    alpha = np.reshape(x, (-1, 1))
    alpha = np.repeat(alpha, n, axis=1)
    alpha = alpha - np.transpose(alpha)

    # set diagonal to 1 to fix product of zero
    np.fill_diagonal(alpha, 1)

    alpha = np.prod(alpha, axis=1)
    alpha = 1/alpha
    return alpha

def langrange_alpha_extend(x, y, alpha, x_hat, y_hat):
    n = len(x)
    n_hat = len(x_hat)
    
    x = np.reshape(x, (-1, 1))
    x_hat = np.reshape(x_hat, (-1, 1))

    x_repeat = np.repeat(x, n_hat, axis=1)
    x_hat_repeat = np.repeat(x_hat, n, axis=1)
    x_hat_repeat = np.transpose(x_hat_repeat)
    
    alpha_hat_top = x_repeat - x_hat_repeat
    alpha_hat_top = np.prod(alpha_hat_top, axis=1)
    alpha_hat_top = alpha/alpha_hat_top

    x_concat = np.concatenate((x, x_hat), axis=0)
    x_concat_repeat = np.repeat(x_concat, n_hat, axis=1)
    x_concat_repeat = np.transpose(x_concat_repeat)
    x_hat_repeat = np.repeat(x_hat, n+n_hat, axis=1)
    alpha_hat_bottom = x_concat_repeat - x_hat_repeat

    alpha_hat_bottom = 1/np.prod(alpha_hat_bottom, axis=1, where=alpha_hat_bottom!=0)

    alpha_hat = np.concatenate((alpha_hat_top, alpha_hat_bottom), axis=0)

    return alpha_hat


def langrange_s_t(x,y, new_x):
    n = len(x)
    m = len(new_x)
    s = np.ones((n,m))
    t = np.ones((n,m))
    for i in range(1,n):
        s[i] = s[i-1]*(new_x-x[i-1])
        j = n - i - 1
        t[j] = t[j+1]*(new_x-x[j+1])
    return s, t

def lagrange_s_t_extend(x, y, new_x, s, t, x_hat, y_hat):
    n = len(x)
    n_hat = len(x_hat)
    m = len(new_x)

    s_hat = np.ones((n+n_hat,m))
    s_hat[:n, :] = s

    t_hat = np.ones((n+n_hat,m))
    t_hat[:n, :] = t

    s_hat[n] = s_hat[n-1]*(new_x-x[n-1])

    for i in range(1, n_hat):
        s_hat[i+n] = s_hat[i+n-1]*(new_x-x_hat[i-1])

    for i in range(n + n_hat - 2, n - 2, -1):
        t_hat[i] = t_hat[i+1]*(new_x - x_hat[i-n+1])

    t_hat[:n-1, :] = t_hat[:n-1, :] * t_hat[n-1:n, :]

    return s_hat, t_hat


def langrange_interpolation(x,y, new_x, alpha=None, s=None, t=None):
    if alpha is None:
        alpha = langrange_alpha(x,y)
    alpha = np.reshape(alpha, (-1, 1))
    n = len(x)
    m = len(new_x)

    if (s is None) or (t is None):
        s, t = langrange_s_t(x,y, new_x)

    l = np.ones((n,m))
    l = s * t * alpha

    y_new = l * np.reshape(y, (-1, 1))
    y_new = np.sum(y_new, axis=0)

    # print("s:\n", s)
    # print("t:\n", t)
    # print("s*t\n", s*t)
    # print("l:\n", l)
    # print("y:\n", y)

    return y_new
    
def newton_interpolation(x, y, new_x):
    n = len(x)
    m = len(new_x)
    A = np.zeros((n,n))
    A[:,0] = 1
    for i in range(1,n):
        for j in range(1,i+1):
            A[i,j] = A[i,j-1]*(x[i]-x[j-1])

    w = la.solve_triangular(A, y, lower=True)
    w = np.reshape(w, (-1, 1))

    pi = np.ones((n,m))
    for i in range(1,n):
        pi[i] = pi[i-1]*(new_x-x[i-1])


    w = np.repeat(w, m, axis=1)
    p = w * pi
    new_y = np.sum(p, axis=0)
    return new_y

def newton_a(x, y):
    n = len(x)
    a = np.zeros((n,n))
    a[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            a[i,j] = (a[i+1,j-1] - a[i,j-1]) / (x[i+j] - x[i])
    return a

def newton_a_extend(x, y, a, x_hat, y_hat):
    n = len(x)
    n_hat = len(x_hat)
    a_hat = np.zeros((n+n_hat,n+n_hat))
    a_hat[:n, :n] = a
    x_extend = np.concatenate((x, x_hat), axis=0)
    a_hat[n:,0] = y_hat
    for j in range(1,n+n_hat):
        for i in range(n-j, n+n_hat-j):
            a_hat[i,j] = (a_hat[i+1,j-1] - a_hat[i,j-1]) / (x_extend[i+j] - x_extend[i])
    return a_hat

def newton_pi(x, y, new_x):
    n = len(x)
    m = len(new_x)
    pi = np.ones((n,m))
    for i in range(1,n):
        pi[i] = pi[i-1]*(new_x-x[i-1])
    return pi

def newton_pi_extend(x, y, new_x, pi, x_hat, y_hat):
    n = len(x)
    n_hat = len(x_hat)
    m = len(new_x)
    pi_hat = np.ones((n+n_hat,m))
    pi_hat[:n, :] = pi
    pi_hat[n] = pi_hat[n-1]*(new_x-x[n-1])
    for i in range(1,n_hat):
        pi_hat[i+n] = pi_hat[i+n-1]*(new_x-x_hat[i-1])
    return pi_hat

def newton_divided_difference(x, y, new_x, a=None, pi=None):
    n = len(x)
    m = len(new_x)
    if a is None:
        a = newton_a(x, y)
    w = a[0,:]
    w = np.reshape(w, (-1, 1))

    if pi is None:
        pi = newton_pi(x, y, new_x)

    w = np.repeat(w, m, axis=1)
    p = w * pi
    new_y = np.sum(p, axis=0)
    return new_y

def newton_divided_difference_extend(x, y, new_x, new_y, x_hat, y_hat, a, pi):
    n = len(x)
    n_hat = len(x_hat)
    m = len(new_x)
    new_y_hat = new_y[:]
    w = a[0,n:n+n_hat]
    w = np.reshape(w, (-1, 1))
    pi = pi[n:n+n_hat,:]
    p = w * pi
    new_y_hat = new_y_hat + np.sum(p, axis=0)
    return new_y_hat


def f(x):
    # return x**2
    return np.sin(x)

def genearate_data():
    x = np.linspace(0, 10, 11)
    y = f(x)

    x_hat = np.linspace(11, 20, 10)
    y_hat = f(x_hat)

    x_extend = np.concatenate((x, x_hat), axis=0)
    y_extend = np.concatenate((y, y_hat), axis=0)

    x_new = np.linspace(0, 20, 100)

    return x, y, x_hat, y_hat, x_extend, y_extend, x_new


def main():
    x = np.linspace(0,100,100)
    y = f(x)
    x_new = np.linspace(0,100,1000)
    lagrange_base_times = []
    lagrange_extend_times = []
    newton_times = []
    newton_divided_difference_times = []
    newton_divided_difference_extend_times = []
    for i in range(1,len(x)):
        start = time.time()
        langrange_interpolation(x[:i], y[:i], x_new)
        end = time.time()
        lagrange_base_times.append(end-start)


    alpha = langrange_alpha(x[:1], y[:1])
    s,t = langrange_s_t(x[:1], y[:1], x_new)
    for i in range(1,len(x)):
        start = time.time()
        alpha = langrange_alpha_extend(x[:i], y[:i], alpha, x[i:i+1], y[i:i+1])
        s,t = lagrange_s_t_extend(x[:i], y[:i], x_new, s, t, x[i:i+1], y[i:i+1])
        langrange_interpolation(x[:i+1], y[:i+1], x_new, alpha, s, t)
        end = time.time()
        lagrange_extend_times.append(end-start)

    for i in range(1,len(x)):
        start = time.time()
        newton_interpolation(x[:i], y[:i], x_new)
        end = time.time()
        newton_times.append(end-start)

    for i in range(1,len(x)):
        start = time.time()
        a = newton_a(x[:i], y[:i])
        pi = newton_pi(x[:i], y[:i], x_new)
        newton_divided_difference(x[:i], y[:i], x_new, a, pi)
        end = time.time()
        newton_divided_difference_times.append(end-start)

    a = newton_a(x[:1], y[:1])
    pi = newton_pi(x[:1], y[:1], x_new)
    y_new = newton_divided_difference(x[:i], y[:i], x_new, a, pi)
    for i in range(1,len(x)):
        start = time.time()
        a = newton_a_extend(x[:i], y[:i], a, x[i:i+1], y[i:i+1])
        pi = newton_pi_extend(x[:i], y[:i], x_new, pi, x[i:i+1], y[i:i+1])
        y_new = newton_divided_difference_extend(x[:i], y[:i], x_new, y_new, x_new, y_new, a, pi)
        end = time.time()
        newton_divided_difference_extend_times.append(end-start)


    # plot times
    plt.plot(np.linspace(1,len(lagrange_base_times), len(lagrange_base_times)), lagrange_base_times, label="lagrange_base")
    plt.plot(np.linspace(1,len(lagrange_extend_times), len(lagrange_extend_times)), lagrange_extend_times, label="lagrange_extend")
    # plt.plot(np.linspace(1,len(newton_times), len(newton_times)), newton_times, label="newton")
    # plt.plot(np.linspace(1,len(newton_divided_difference_times), len(newton_divided_difference_times)), newton_divided_difference_times, label="newton_divided_difference")
    plt.plot(np.linspace(1,len(newton_divided_difference_extend_times), len(newton_divided_difference_extend_times)), newton_divided_difference_extend_times, label="newton_divided_difference_extend")
    plt.legend()
    plt.show()
        

# def main():
#     x, y, x_hat, y_hat, x_extend, y_extend, x_new = genearate_data()

#     a = newton_a(x, y)
#     pi = newton_pi(x, y, x_new)
#     y_new = newton_divided_difference(x, y, x_new, a, pi)

#     a_extend = newton_a_extend(x, y, a, x_hat, y_hat)
#     pi_extend = newton_pi_extend(x, y, x_new, pi, x_hat, y_hat)
#     y_new_extend = newton_divided_difference_extend(x, y, x_new, y_new, x_hat, y_hat, a=a_extend, pi=pi_extend)

#     plt.plot(x, y, 'o', label='original')
#     plt.plot(x_hat, y_hat, 'o', label='extended data')
#     plt.plot(x_new, y_new, 'r', label='newton_interpolation')
#     plt.plot(x_new, y_new_extend, 'g', label='newton_interpolation extended')
#     plt.legend()
#     plt.ylim(-3, 3)
#     plt.show()



#langrange main function
# def main():

#     # sample x and y from a function
#     x = np.linspace(0, 10, 11)
#     y = f(x)

#     x_hat = np.linspace(11, 20, 10)
#     y_hat = f(x_hat)

#     x_new = np.linspace(0, 20, 100)

#     alpha = langrange_alpha(x,y)
#     s, t = langrange_s_t(x,y, x_new)

#     y_new = langrange_interpolation(x,y, x_new, alpha, s, t)

#     alpha_extend = langrange_alpha_extend(x, y, alpha, x_hat, y_hat)
#     s_extened, t_extend = lagrange_s_t_extend(x, y, x_new, s, t, x_hat, y_hat)

#     x_extend = np.concatenate((x, x_hat), axis=0)
#     y_extend = np.concatenate((y, y_hat), axis=0)

#     y_new_extend = langrange_interpolation(x_extend, y_extend, x_new, alpha_extend, s_extened, t_extend)

#     # alpha_base = langrange_alpha(x,y)
#     # s_base, t_base = langrange_s_t(x,y, x_new)

#     # y_new = langrange_interpolation(x,y, x_new, alpha=alpha, s=s, t=t)
#     # y_new_base = langrange_interpolation(x,y, x_new, alpha=alpha_base, s=s_base, t=t_base)

#     # print("s:\n", s)
#     # print("t:\n", t)
#     # print("s_base:\n", s_base)
#     # print("t_base:\n", t_base)

#     # print("y_new:\n", y_new)
#     # print("y_new_base:\n", y_new_base)


#     plt.plot(x, y, 'o', label='original data')
#     plt.plot(x_hat, y_hat, 'o', label='extended data')
#     plt.plot(x_new, y_new, 'r', label='interpolated data')
#     plt.plot(x_new, y_new_extend, 'g', label='interpolated data with extended data')
#     plt.ylim(-3, 3)
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    main()
