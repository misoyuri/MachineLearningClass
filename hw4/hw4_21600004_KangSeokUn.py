from sys import getdefaultencoding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv("./weight-height.csv")

all_height = df["Height"].to_numpy()
male_height = df.loc[df.Gender == "Male", "Height"].to_numpy()
female_height = df.loc[df.Gender == "Female", "Height"].to_numpy()
all_weight = df["Weight"].to_numpy()
male_weight = df.loc[df.Gender == "Male", "Weight"].to_numpy()
female_weight = df.loc[df.Gender == "Female", "Weight"].to_numpy()

print("all height::", all_height.shape)
print("male height::", male_height.shape)
print("female height::", female_height.shape)

print("all weight::", all_weight.shape)
print("male weight::", male_weight.shape)
print("female weight::", female_weight.shape)

all_data = {"All Height" : all_height, "Male Height" : male_height, "Female Height" : female_height, "ALL Weight" : all_weight, "Male Weight" : male_weight, "Female Weight" : female_weight}

# Histogram
for key in all_data:
    print(key)
    counts, bins = np.histogram(all_data[key])
    plt.hist(bins[:-1], bins, weights=counts, label="# of bin:"+ str(len(bins)))
    plt.legend(loc='best')
    plt.title("Histogram: " + key)
    plt.savefig("./hist/" + key + ".png")
    plt.clf()


# Gaussian with MLE
for key in all_data:
    x = np.linspace(np.min(all_data[key]), np.max(all_data[key]), np.size(all_data[key]))
    mean = np.sum(all_data[key]) / np.size(all_data[key])
    var2 = np.sum((all_data[key]-mean)**2) / np.size(all_data[key])
    gau = np.exp(pow(x-mean, 2)/(-2*var2)) / pow(2 * np.pi * var2, 0.5) 
    plt.plot(x, gau, label="mean: {:.3f}\nvariance: {:.3f}".format( mean, var2))
    plt.legend(loc='best')
    plt.title("Gaussian with MLE: " + key)
    plt.savefig("./MLE/"+key+".png")
    plt.clf()


# KDE
for key in all_data:
    h_list = [1.0, 2.5, 5, 10]
    
    N = np.size(all_data[key])
    x = np.linspace(np.min(all_data[key]), np.max(all_data[key]), N)

    sig_k = 0.0
    
    for h in h_list:
        for x_n in all_data[key]:
            z = (x-x_n) / h
            sig_k = sig_k + np.exp(-0.5 * z.T * z) / pow(2 * np.pi, all_data[key].ndim/2)
            
        p_KDE = sig_k / (N * pow(h, x.ndim))
    
        plt.plot(x, p_KDE, label="h:"+str(h))
        plt.legend(loc='best')
        plt.title("KDE: " + key)
    plt.savefig("./KDE/"+key+".png")
    plt.clf()