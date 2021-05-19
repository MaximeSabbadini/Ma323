import numpy as np
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits import mplot3d
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pylab as pl

def u0(x):
    return (np.sin(np.pi*x))**10

def dec_am(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    print(c)
    if np.abs(V)*tau<=h:
        print(f"Attention, la CFL n'est pas respectée pour h = {h} et tau={tau}")
        # return
    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Mda = np.eye(N)*(1-c)

    for i in range(N-1):
        Mda[i+1, i] = c
    Mda[0, N-1] = c

    # print(Mda)

    Ut = np.zeros((len(T), N))
    Un = u0(X)
    Ut[0, :] = Un
    for i in range(1, len(T)):
        Un = Mda@Un
        Ut[i, :] = Un

    return T, X, Ut

def lax_friedrichs(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    print(c)
    if np.abs(V)*tau<=h:
        print(f"Attention, la CFL n'est pas respectée pour h = {h} et tau={tau}")
        # return
    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Mlf = np.zeros((N, N))

    for i in range(N-1):
        Mlf[i+1, i] = (1+c)/2
        Mlf[i, i+1] = (1-c)/2
    Mlf[0, N-1] = (1+c)/2
    Mlf[N-1, 0] = (1-c)/2

    # print(Me)

    Ut = np.zeros((len(T), N))
    Un = u0(X)
    Ut[0, :] = Un
    for i in range(1, len(T)):
        Un = Mlf@Un
        Ut[i, :] = Un

    return T, X, Ut

def lax_wendroff(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    print(c)
    if np.abs(V)*tau<=h:
        print(f"Attention, la CFL n'est pas respectée pour h = {h} et tau={tau}")
    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Mlw = np.eye(N)*(1-c**2)

    for i in range(N-1):
        Mlw[i+1, i] = (c+c**2)/2
        Mlw[i, i+1] = (c**2-c)/2
    Mlw[0, N-1] = (c+c**2)/2
    Mlw[N-1, 0] = (c**2-c)/2

    # print(Me)

    Ut = np.zeros((len(T), N))
    Un = u0(X)
    Ut[0, :] = Un
    for i in range(1, len(T)):
        Un = Mlw@Un
        Ut[i, :] = Un

    return T, X, Ut


def explicite_centree(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    print(c)
    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Me = np.eye(N)

    for i in range(N-1):
        Me[i+1, i] = c/2
        Me[i, i+1] = -c/2
    Me[0, N-1] = c/2
    Me[N-1, 0] = -c/2

    # print(Me)

    Ut = np.zeros((len(T), N))
    Un = u0(X)
    Ut[0, :] = Un
    for i in range(1, len(T)):
        Un = Me@Un
        Ut[i, :] = Un

    return T, X, Ut

def Impl_centree(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    print(c)
    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Mi = np.eye(N)

    for i in range(N-1):
        Mi[i+1, i] = -c/2
        Mi[i, i+1] = c/2
    Mi[0, N-1] = -c/2
    Mi[N-1, 0] = c/2

    # print(Mi)

    Ut = np.zeros((len(T), N))
    Un = u0(X)
    Ut[0, :] = Un

    for i in range(1, len(T)):
        Un = np.linalg.solve(Mi, Un)
        Ut[i, :] = Un

    return T, X, Ut

V = 2


tau = .01
h = .02
N = int(1/h-1)
T1, X1, Ut1 = Impl_centree(N, tau)

h = .002
tau = .005
N = int(1/h-1)
T2, X2, Ut2 = Impl_centree(N, tau)

h = .002
tau = .002
N = int(1/h-1)
T3, X3, Ut3 = Impl_centree(N, tau)

h = .005
tau = .0002
N = int(1/h-1)
T4, X4, Ut4 = Impl_centree(N, tau)

plt.figure(figsize=(15,5))
plt.suptitle('Schéma de Implicite centré')
plt.subplot(141)
plt.title('h = 0.02, $\\tau = 0.01$')
plt.plot(X1, Ut1[0, :], label='t=0')
plt.plot(X1, Ut1[10, :], label=f't = {T1[10]}')
plt.plot(X1, Ut1[20, :], label=f't = {T1[20]}')
plt.plot(X1, Ut1[-1, :], label=f't = {T1[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(142)
plt.title('h = 0.002, $\\tau = 0.005$')
plt.plot(X2, Ut2[0, :], label='t=0')
plt.plot(X2, Ut2[10, :], label=f't = {T2[100]}')
plt.plot(X2, Ut2[20, :], label=f't = {T2[200]}')
plt.plot(X2, Ut2[-1, :], label=f't = {T2[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(143)
plt.title('h = 0.002, $\\tau = 0.002$')
plt.plot(X3, Ut3[0, :], label='t=0')
plt.plot(X3, Ut3[10, :], label=f't = {T3[10]}')
plt.plot(X3, Ut3[20, :], label=f't = {T3[20]}')
plt.plot(X3, Ut3[-1, :], label=f't = {T3[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(144)
plt.title('h = 0.005, $\\tau = 0.0002$')
plt.plot(X4, Ut4[0, :], label='t=0')
plt.plot(X4, Ut4[10, :], label=f't = {T4[10]}')
plt.plot(X4, Ut4[20, :], label=f't = {T4[20]}')
plt.plot(X4, Ut4[-1, :], label=f't = {T4[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.show()

tau = .01
h = .02
N = int(1/h-1)
T1, X1, Ut1 = explicite_centree(N, tau)

h = .002
tau = .005
N = int(1/h-1)
T2, X2, Ut2 = explicite_centree(N, tau)

h = .002
tau = .002
N = int(1/h-1)
T3, X3, Ut3 = explicite_centree(N, tau)

h = .005
tau = .0002
N = int(1/h-1)
T4, X4, Ut4 = explicite_centree(N, tau)

plt.figure(figsize=(15,5))
plt.suptitle('Schéma Explicite centré')
plt.subplot(141)
plt.title('h = 0.02, $\\tau = 0.01$')
plt.plot(X1, Ut1[0, :], label='t=0')
plt.plot(X1, Ut1[10, :], label=f't = {T1[10]}')
plt.plot(X1, Ut1[20, :], label=f't = {T1[20]}')
plt.plot(X1, Ut1[-1, :], label=f't = {T1[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(142)
plt.title('h = 0.002, $\\tau = 0.005$')
plt.plot(X2, Ut2[0, :], label='t=0')
plt.plot(X2, Ut2[10, :], label=f't = {T2[100]}')
plt.plot(X2, Ut2[20, :], label=f't = {T2[200]}')
plt.plot(X2, Ut2[-1, :], label=f't = {T2[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(143)
plt.title('h = 0.002, $\\tau = 0.002$')
plt.plot(X3, Ut3[0, :], label='t=0')
plt.plot(X3, Ut3[10, :], label=f't = {T3[10]}')
plt.plot(X3, Ut3[20, :], label=f't = {T3[20]}')
plt.plot(X3, Ut3[-1, :], label=f't = {T3[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(144)
plt.title('h = 0.005, $\\tau = 0.0002$')
plt.plot(X4, Ut4[0, :], label='t=0')
plt.plot(X4, Ut4[10, :], label=f't = {T4[10]}')
plt.plot(X4, Ut4[20, :], label=f't = {T4[20]}')
plt.plot(X4, Ut4[-1, :], label=f't = {T4[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.show()


tau = .01
h = .02
N = int(1/h-1)
T1, X1, Ut1 = lax_friedrichs(N, tau)

h = .002
tau = .005
N = int(1/h-1)
T2, X2, Ut2 = lax_friedrichs(N, tau)

h = .002
tau = .002
N = int(1/h-1)
T3, X3, Ut3 = lax_friedrichs(N, tau)

h = .005
tau = .0002
N = int(1/h-1)


T4, X4, Ut4 = lax_friedrichs(N, tau)
plt.figure(figsize=(15,5))
plt.suptitle('Schéma de Lax-Friedrichs')
plt.subplot(141)
plt.title('h = 0.02, $\\tau = 0.01$')
plt.plot(X1, Ut1[0, :], label='t=0')
plt.plot(X1, Ut1[10, :], label=f't = {T1[10]}')
plt.plot(X1, Ut1[20, :], label=f't = {T1[20]}')
plt.plot(X1, Ut1[-1, :], label=f't = {T1[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(142)
plt.title('h = 0.002, $\\tau = 0.005$')
plt.plot(X2, Ut2[0, :], label='t=0')
plt.plot(X2, Ut2[10, :], label=f't = {T2[100]}')
plt.plot(X2, Ut2[20, :], label=f't = {T2[200]}')
plt.plot(X2, Ut2[-1, :], label=f't = {T2[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(143)
plt.title('h = 0.002, $\\tau = 0.002$')
plt.plot(X3, Ut3[0, :], label='t=0')
plt.plot(X3, Ut3[10, :], label=f't = {T3[10]}')
plt.plot(X3, Ut3[20, :], label=f't = {T3[20]}')
plt.plot(X3, Ut3[-1, :], label=f't = {T3[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(144)
plt.title('h = 0.005, $\\tau = 0.0002$')
plt.plot(X4, Ut4[0, :], label='t=0')
plt.plot(X4, Ut4[10, :], label=f't = {T4[10]}')
plt.plot(X4, Ut4[20, :], label=f't = {T4[20]}')
plt.plot(X4, Ut4[-1, :], label=f't = {T4[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.show()


tau = .01
h = .02
N = int(1/h-1)
T1, X1, Ut1 = lax_wendroff(N, tau)

h = .002
tau = .005
N = int(1/h-1)
T2, X2, Ut2 = lax_wendroff(N, tau)

h = .002
tau = .002
N = int(1/h-1)
T3, X3, Ut3 = lax_wendroff(N, tau)

h = .005
tau = .0002
N = int(1/h-1)
T4, X4, Ut4 = lax_wendroff(N, tau)

plt.figure(figsize=(15,5))
plt.suptitle('Schéma de Lax-Wendroff')
plt.subplot(141)
plt.title('h = 0.02, $\\tau = 0.01$')
plt.plot(X1, Ut1[0, :], label='t=0')
plt.plot(X1, Ut1[10, :], label=f't = {T1[10]}')
plt.plot(X1, Ut1[20, :], label=f't = {T1[20]}')
plt.plot(X1, Ut1[-1, :], label=f't = {T1[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(142)
plt.title('h = 0.002, $\\tau = 0.005$')
plt.plot(X2, Ut2[0, :], label='t=0')
plt.plot(X2, Ut2[10, :], label=f't = {T2[100]}')
plt.plot(X2, Ut2[20, :], label=f't = {T2[200]}')
plt.plot(X2, Ut2[-1, :], label=f't = {T2[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(143)
plt.title('h = 0.002, $\\tau = 0.002$')
plt.plot(X3, Ut3[0, :], label='t=0')
plt.plot(X3, Ut3[10, :], label=f't = {T3[10]}')
plt.plot(X3, Ut3[20, :], label=f't = {T3[20]}')
plt.plot(X3, Ut3[-1, :], label=f't = {T3[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(144)
plt.title('h = 0.005, $\\tau = 0.0002$')
plt.plot(X4, Ut4[0, :], label='t=0')
plt.plot(X4, Ut4[10, :], label=f't = {T4[10]}')
plt.plot(X4, Ut4[20, :], label=f't = {T4[20]}')
plt.plot(X4, Ut4[-1, :], label=f't = {T4[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.show()

tau = .01
h = .02
N = int(1/h-1)
T1, X1, Ut1 = dec_am(N, tau)

h = .002
tau = .005
N = int(1/h-1)
T2, X2, Ut2 = dec_am(N, tau)

h = .002
tau = .002
N = int(1/h-1)
T3, X3, Ut3 = dec_am(N, tau)

h = .005
tau = .0002
N = int(1/h-1)
T4, X4, Ut4 = dec_am(N, tau)

plt.figure(figsize=(15,5))
plt.suptitle('Schéma décentré amont')
plt.subplot(141)
plt.title('h = 0.02, $\\tau = 0.01$')
plt.plot(X1, Ut1[0, :], label='t=0')
plt.plot(X1, Ut1[10, :], label=f't = {T1[10]}')
plt.plot(X1, Ut1[20, :], label=f't = {T1[20]}')
plt.plot(X1, Ut1[-1, :], label=f't = {T1[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(142)
plt.title('h = 0.002, $\\tau = 0.005$')
plt.plot(X2, Ut2[0, :], label='t=0')
plt.plot(X2, Ut2[10, :], label=f't = {T2[100]}')
plt.plot(X2, Ut2[20, :], label=f't = {T2[200]}')
plt.plot(X2, Ut2[-1, :], label=f't = {T2[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(143)
plt.title('h = 0.002, $\\tau = 0.002$')
plt.plot(X3, Ut3[0, :], label='t=0')
plt.plot(X3, Ut3[10, :], label=f't = {T3[10]}')
plt.plot(X3, Ut3[20, :], label=f't = {T3[20]}')
plt.plot(X3, Ut3[-1, :], label=f't = {T3[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.subplot(144)
plt.title('h = 0.005, $\\tau = 0.0002$')
plt.plot(X4, Ut4[0, :], label='t=0')
plt.plot(X4, Ut4[10, :], label=f't = {T4[10]}')
plt.plot(X4, Ut4[20, :], label=f't = {T4[20]}')
plt.plot(X4, Ut4[-1, :], label=f't = {T4[-1]}')
plt.legend()
plt.grid()
plt.xlabel('Espace')
plt.ylabel('Amplitude')
plt.show()

"""
TT, XX = np.meshgrid(T1, X1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(TT, XX, Ut1.T, cmap='jet')
ax.set_xlabel('Temps')
ax.set_ylabel('Espace')
ax.set_title('')
plt.show()

plt.figure()
for i in range(len(Ut1)):
    plt.plot(X1, Ut1[i, :])
    plt.pause(.1)
    plt.clf()
plt.show()
"""

