#!/usr/bin/env python
# coding: utf-8

import tkinter as tk
from os import listdir,system
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

mypath = './data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
file_name = ''

def run():
    original_data,dataset, exp_ans, dim = load_data()
    W = init_weight(dim)
    total = 0
    total = len(dataset)
    num_column = len(dataset[0])
    len_train = int(2*total/3)
    len_test = int(total-len_train)
    new_W, trainErrors = training(W,dataset,exp_ans)
    testErrors = testing(new_W,dataset,exp_ans)
    acc_rate = accuracy(total,testErrors)
    return original_data, dataset, exp_ans,new_W,acc_rate

def load_data():
    with open(file_name,'r') as f :
        load = []
        load.clear()
        for line in f.readlines():
            load.append(list(map(float,line.strip().split(' '))))
        np.random.shuffle(load)
        load = np.array(load)
    load_X = []
    exp = []
    load_X.clear()
    exp.clear()
    for arr in load:
        temp = arr[:-1]
        tempX = [-1.0]
        exp.append(arr[-1])
        for element in temp:
            tempX.append(element)
        load_X.append(tempX)
        del tempX
    dim = len(arr)-1
    return load,load_X, exp, dim

def get_weight():
    return round(random.uniform(-1, 1), 2)

def init_weight(Dim):
    weight = [-1]
    for i in range(0,Dim):
        tmp = get_weight()
        weight.append(tmp)
    return weight

def calculate(W,X):
    np1 = np.array(W)
    np2 = np.array(X)
    mul = np1*np2
    ans = 0
    for i in mul:
        ans += i
    return ans

def check_sgn(sgn,label_0,label_1):
    if sgn > 0:
        sgn = label_1
    else:
        sgn = label_0
    return sgn

def set_labels(exp_ans):
    label_min = min(exp_ans)
    label_max = max(exp_ans)
    return label_min, label_max

def update_weight(W_vector,X_vector,learning_rate,sgn,exp_ans):
    np_W = np.array(W_vector)
    np_X = np.array(X_vector)
    if sgn < exp_ans:
        new_W_vector = np_W + learning_rate * np_X
    else:
        new_W_vector = np_W - learning_rate * np_X
    #print("new_W_vector = W_vector -+ learning_rate * X_vector:", new_W_vector)
    return new_W_vector

def testing(W_vector,datasets,exp_ans):
    label_0,label_1 = set_labels(exp_ans)
    test_errors = 0
    len_data = len(datasets)
    for i in range(0, len_data):
        Y = calculate(W_vector,datasets[i])
        sgn = check_sgn(Y,label_0,label_1)
        if sgn != exp_ans[i]:
            test_errors += 1
    return test_errors

def training(W_vector,datasets,exp_ans):
    label_0,label_1 = set_labels(exp_ans)
    count = 0
    len_data = len(datasets)
    while (count < int(Iteration)):
        train_errors = 0
        for i in range(0, len_data):
            Y = calculate(W_vector,datasets[i])
            sgn = check_sgn(Y,label_0,label_1)
            if sgn != exp_ans[i]:
                train_errors += 1
                W_vector = update_weight(W_vector,datasets[i],float(learningRate),sgn,exp_ans[i])
        count += 1
    return W_vector, train_errors

def accuracy(total,errors_num):
    return round(float((total - errors_num) / total) * 100, 3)

class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.windows = master
        self.grid()
        self.mypath = './data/'
        self.files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.create_windows()

    def get_list(self,event):
        global file_name
        self.index = self.listbox.curselection()[0]
        self.selected = self.listbox.get(self.index)
        file_name = 'data/'+self.selected

    def create_windows(self):
        self.windows.title("Perceptron Homework1")

        self.listbox = tk.Listbox(windows, width=20, height=6)
        self.listbox.grid(row=0, column=0,columnspan=2,stick=tk.W+tk.E)

        self.yscroll = tk.Scrollbar(command=self.listbox.yview, orient=tk.VERTICAL)
        self.yscroll.grid(row=0, column=2, sticky=tk.W+tk.E)
        self.listbox.configure(yscrollcommand=self.yscroll.set)

        for item in self.files:
            self.listbox.insert(tk.END, item)

        self.listbox.bind('<ButtonRelease-1>', self.get_list)
        
        self.learning = tk.Label(windows, text="Learning rate:").grid(row=1,column=0, sticky=tk.W+tk.E)
        self.iteration = tk.Label(windows, text="Iteration:").grid(row=2,column=0, sticky=tk.W+tk.E)
        self.weight = tk.Label(windows, text="weight result:").grid(row=3,column=0, sticky=tk.W+tk.E)
        self.draw_iteration = tk.Label(windows, text="Accuracy rate:").grid(row=4,column=0, sticky=tk.W+tk.E)
        self.e1 = tk.Entry(windows)
        self.e2 = tk.Entry(windows)
        self.e3 = tk.Entry(windows)
        self.e4 = tk.Entry(windows)
        self.e1.grid(row=1, column=1, sticky=tk.W+tk.E)
        self.e2.grid(row=2, column=1, sticky=tk.W+tk.E)
        self.e3.grid(row=3, column=1, sticky=tk.W+tk.E)
        self.e4.grid(row=4, column=1, sticky=tk.W+tk.E)
        self.e1.delete(0,'end')
        self.e2.delete(0,'end')
        self.e1.insert(10,0.01)
        self.e2.insert(10,100)
        self.quit = tk.Button(windows, text='Quit', command=windows.quit).grid(row=5, column=0, sticky=tk.W+tk.E)
        self.show = tk.Button(windows, text='Show', command=self.show_entry_fields).grid(row=5, column=1, sticky=tk.W+tk.E)
        
        self.result_figure = Figure(figsize=(5,4), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, self.windows)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().grid(row=6, column=0, columnspan=3, sticky=tk.W+tk.E)
        
    def show_entry_fields(self):
        global learningRate
        global Iteration
        learningRate = self.e1.get()
        Iteration = self.e2.get()
        original_data,datasets,exp_ans,new_W,acc_rate = run()
        self.plot_data(original_data,exp_ans,new_W,acc_rate)        
    def plot_data(self,inputs,targets,weights,acc_rate):
        self.result_figure.clf()
        self.result_figure.a = self.result_figure.add_subplot(111)
        #self.plt.figure(figsize=(10,6))
        target_0,target_1 = set_labels(targets)
        total = len(inputs)
        len_train = int(2*total/3)
        train_inputs = inputs[:len_train]
        train_targets = targets[:len_train]
        test_inputs = inputs[len_train:]
        test_targets = targets[len_train:]
        for input,target in zip(train_inputs,train_targets):
            self.result_figure.a.plot(input[0],input[1],'ro' if (target == target_0) else 'bo')
        for input,target in zip(test_inputs,test_targets):
            self.result_figure.a.plot(input[0],input[1],'gx' if (target == target_0) else 'gx')
        for i in np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1])):
            slope = -1*(weights[1]/weights[2])
            intercept = weights[0]/weights[2]
            y = (slope*i) + intercept
            self.result_figure.a.plot(i, y, 'ko')
        self.e3.delete(0,'end')
        self.e4.delete(0,'end')
        self.e3.insert(10,str(weights))
        self.e4.insert(10,str(acc_rate))
        self.result_figure.a.set_title('Training Data')
        self.result_canvas.draw()
        
if __name__ == "__main__":
    windows = tk.Tk()
    app = Application(windows)
    windows.mainloop()

