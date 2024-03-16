import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from corrTz import normalize_vector


def normalize_vector(vector):
    min_value = np.min(vector)
    max_value = np.max(vector)
    
    normalized_vector = (vector - min_value) / (max_value - min_value) * 2 - 1
    
    return normalized_vector

# load data from OF_outputs directory for a sample, and ground truth
# movement 1
fib_dat_exp1 = pd.read_csv("rotX/fibrescope/fib1_experimental_results.csv", delimiter=",", usecols=["lk_gray_x"])
trim1 = int(0.5*len(fib_dat_exp1))
fib_dat_exp1 = fib_dat_exp1[12:trim1]
fib_dat_gnd1 = pd.read_csv("rotX/fibrescope/fib1euler.csv", delimiter=",", usecols=["roll_x"]) 
fib_dat_gnd1 = fib_dat_gnd1[12:trim1]
# movement 2
fib_dat_exp2 = pd.read_csv("transZ/Tz_raw_experimental_results_ALL.csv", delimiter=",", usecols=["fib1"])
trim2 = int(0.5*len(fib_dat_exp2))
fib_dat_exp2 = fib_dat_exp2[10:trim2]
fib_dat_gnd2 = pd.read_csv("transZ/ground_truth/fibrescope/fibrescope1.csv", delimiter=",", usecols=["Franka Tz"]) * (-1)
fib_dat_gnd2 = fib_dat_gnd2[10:trim2]

# normalize values

fib_dat_exp1 = normalize_vector(fib_dat_exp1)
fib_dat_gnd1 = normalize_vector(fib_dat_gnd1)
fib_dat_exp2 = normalize_vector(fib_dat_exp2)
fib_dat_gnd2 = normalize_vector(fib_dat_gnd2)

print('sizes: ', len(fib_dat_exp1), len(fib_dat_gnd1), len(fib_dat_exp2), len(fib_dat_gnd2))

# plot graphs. 1: rotation - movement 1. 2: translation - movement 2
t1 = np.linspace(0,30,len(fib_dat_exp1))
t2 = np.linspace(0,30,len(fib_dat_exp2))
print('sizes t: ', len(t1), len(t2))

plt.figure()
plt.plot(t1,fib_dat_exp1, 'r')
plt.plot(t1,fib_dat_gnd1, 'r--')
plt.plot(t2,fib_dat_exp2, 'b')
plt.plot(t2,fib_dat_gnd2, 'b--')
plt.legend(["Ry prediction","Ry ground truth","Tz prediction","Tz ground truth"], loc="upper right")
plt.ylabel("Normalised motion")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
