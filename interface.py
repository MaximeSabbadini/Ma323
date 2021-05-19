import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar
import numpy as np
import time
from tkinter import messagebox

class Application(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('Visualisation')
        self.master.geometry("{0}x{1}+0+0".format(self.master.winfo_screenwidth(), self.master.winfo_screenheight()-70))
        # print(self.master.winfo_screenheight(), self.master.winfo_screenwidth())
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure, master=self.master)

        self.method = "Explicite centré"
        self.status = True
        self.sol_theorique = tk.IntVar(value=1)

        self.b1 = tk.Button(self.master, text='Update', command=self.update_plot)
        self.l1 = tk.Label(self.master, text='h :')
        self.l2 = tk.Label(self.master, text='tau :')
        self.e1 = tk.Entry(self.master)
        self.e2 = tk.Entry(self.master)
        self.c1 = tk.Checkbutton(self.master, text='Solution théorique', variable=self.sol_theorique, onvalue=1, offvalue=0, command=lambda:print(self.sol_theorique.get()))

        self.stop_button = tk.Button(self.master, text='Stop', command=self.stop, state="disabled")
        self.combo = ttk.Combobox(self.master, values=['Explicite centré', 'Implicite centré', 'Lax-Friedrichs', 'Lax-Wendroff', 'Explicite décentré amont'], justify=tk.CENTER)
        self.combo.current(0)
        self.combo.bind('<<ComboboxSelected>>', self.combo_changed)
        self.combo.config(state='readonly')


        self.toolbar = NavigationToolbar(self.canvas, self.master)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.l1.place(x=10, y=self.master.winfo_screenheight()-240) #660
        self.e1.place(x=40, y=self.master.winfo_screenheight()-242) #658
        self.l2.place(x=330, y=self.master.winfo_screenheight()-240) #660
        self.e2.place(x=365, y=self.master.winfo_screenheight()-242) #658
        self.c1.place(x=self.master.winfo_screenwidth()-140, y=self.master.winfo_screenheight()-240) #1300, 660
        self.stop_button.pack()
        self.combo.pack(fill=tk.BOTH)
        self.b1.pack(fill=tk.BOTH)
        self.toolbar.pack()

    def combo_changed(self, evt):
        self.method = self.combo.get()

    def update_plot(self):
        h = float(self.e1.get())
        N = int(1/h-1)
        tau = float(self.e2.get())

        if self.method == 'Explicite centré':
            T, X, Ut = explicite_centree(N, tau)
        elif self.method == 'Implicite centré':
            T, X, Ut = Impl_centree(N, tau)
        elif self.method == 'Lax-Friedrichs':
            try:
                T, X, Ut = lax_friedrichs(N, tau)
            except TypeError:
                return
        elif self.method == 'Lax-Wendroff':
            try:
                T, X, Ut = lax_wendroff(N, tau)
            except TypeError:
                return
        elif self.method == 'Explicite décentré amont':
            try:
                T, X, Ut = dec_am(N, tau)
            except TypeError:
                return

        T_t, X_t, Ut_t = solution_theorique(N, tau)

        self.stop_button["state"] = "normal"
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        for i in range(len(Ut)):
            self.ax.clear()
            self.ax.set_xlabel('Espace')
            self.ax.set_ylabel('Amplitude')
            if self.status:
                self.ax.set_title('Schéma ' + self.method + f': h = {h}, $\\tau$={tau}' +'. t={0:.2f}'.format(T[i]))
                self.ax.plot(X, Ut[i, :], label='Solution calculée')
                self.figure.canvas.start_event_loop(.1)

            else:
                self.status = True
                self.stop_button["state"] = "disabled"
                return
            if self.sol_theorique.get() == 1:
                self.ax.plot(X_t, Ut_t[i, :], label='Solution théorique', ls='--')
            self.ax.grid()
            self.ax.legend(loc=1)
            self.canvas.draw()

    def stop(self):
        self.status = False


def u0(x):
    return (np.sin(np.pi*x))**10

def dec_am(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    # print(c)
    if np.abs(V)*tau>=h:
##        # print(f"Attention, la CFL n'est pas respectée pour h = {h} et tau={tau}")
        messagebox.showinfo("Attention", "La CFL n'est pas respectée, merci de changer tau et h.")
##        return
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

def Impl_centree(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    # print(c)
    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Mi = np.eye(N)

    for i in range(N-1):
        Mi[i+1, i] = -c/2
        Mi[i, i+1] = c/2
    Mi[0, N-1] = -c/2
    Mi[N-1, 0] = c/2

    Ut = np.zeros((len(T), N))
    Un = u0(X)
    Ut[0, :] = Un

    for i in range(1, len(T)):
        Un = np.linalg.solve(Mi, Un)
        Ut[i, :] = Un

    return T, X, Ut

def lax_friedrichs(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    # print(c)
    if np.abs(V)*tau>=h:
##        # print(f"Attention, la CFL n'est pas respectée pour h = {h} et tau={tau}")
        messagebox.showinfo("Attention", "La CFL n'est pas respectée, merci de changer tau et h.")
##        return

    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Mlf = np.zeros((N, N))

    for i in range(N-1):
        Mlf[i+1, i] = (1+c)/2
        Mlf[i, i+1] = (1-c)/2
    Mlf[0, N-1] = (1+c)/2
    Mlf[N-1, 0] = (1-c)/2


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
    # print(c)
    if np.abs(V)*tau>=h:
##        # print(f"Attention, la CFL n'est pas respectée pour h = {h} et tau={tau}")
        messagebox.showinfo("Attention", "La CFL n'est pas respectée, merci de changer tau et h.")
##        return

    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Mlw = np.eye(N)*(1-c**2)

    for i in range(N-1):
        Mlw[i+1, i] = (c+c**2)/2
        Mlw[i, i+1] = (c**2-c)/2
    Mlw[0, N-1] = (c+c**2)/2
    Mlw[N-1, 0] = (c**2-c)/2


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
    # print(c)
    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)
    Me = np.eye(N)

    for i in range(N-1):
        Me[i+1, i] = c/2
        Me[i, i+1] = -c/2
    Me[0, N-1] = c/2
    Me[N-1, 0] = -c/2


    Ut = np.zeros((len(T), N))
    Un = u0(X)
    Ut[0, :] = Un
    for i in range(1, len(T)):
        Un = Me@Un
        Ut[i, :] = Un

    return T, X, Ut

def solution_theorique(N, tau):
    h = 1/(N+1)
    c = V*tau/h
    X = np.linspace(0, 1, N)
    T = np.arange(0, 2, tau)

    Ut = np.zeros((len(T), N))
    Un = u0(X)
    Ut[0, :] = Un
    for i in range(1, len(T)):
        Un = u0(X-V*T[i])
        Ut[i, :] = Un

    return T, X, Ut


if __name__ == '__main__':
    V = 2
    root = tk.Tk()
    app = Application(root)
    app.mainloop()
