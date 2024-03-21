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
max_gnd1 = np.rad2deg(np.max(fib_dat_gnd1))
min_gnd1 = np.rad2deg(np.min(fib_dat_gnd1))
med_gnd1 = np.rad2deg(np.median(fib_dat_gnd1))
resolution1_vec = np.abs(fib_dat_gnd1.diff())
resolution1_min = np.rad2deg(np.min(resolution1_vec))
resolution1_max = np.rad2deg(np.max(resolution1_vec))
resolution1_med = np.rad2deg(np.median(resolution1_vec))
resolution1_mean = np.rad2deg(np.mean(resolution1_vec))
# mid_gnd1 = (max_gnd1 + min_gnd1) / 2 # rejected. median being used instead. small variation between the two.
# print('max, min, med, mid: ', max_gnd1, min_gnd1, med_gnd1, mid_gnd1)
print('rot: max, min, med: ', max_gnd1, min_gnd1, med_gnd1)
print('rot deviations: ', max_gnd1 - med_gnd1, med_gnd1 - min_gnd1)
print('resolution rot [min max median mean]: ', resolution1_min, resolution1_max, resolution1_med, resolution1_mean)

# movement 2
fib_dat_exp2 = pd.read_csv("transZ/Tz_raw_experimental_results_ALL.csv", delimiter=",", usecols=["fib1"])
trim2 = int(0.5*len(fib_dat_exp2))
fib_dat_exp2 = fib_dat_exp2[10:trim2]
fib_dat_gnd2 = pd.read_csv("transZ/ground_truth/fibrescope/fibrescope1.csv", delimiter=",", usecols=["Franka Tz"]) * (-1)
fib_dat_gnd2 = fib_dat_gnd2[10:trim2]
max_gnd2 = np.max(fib_dat_gnd2) * 1e3
min_gnd2 = np.min(fib_dat_gnd2) * 1e3
med_gnd2 = np.median(fib_dat_gnd2) * 1e3
resolution1_vec = np.abs(fib_dat_gnd2.diff())
resolution1_min = np.min(resolution1_vec) * 1e3
resolution1_max = np.max(resolution1_vec) * 1e3
resolution1_med = np.median(resolution1_vec) * 1e3
resolution1_mean = np.mean(resolution1_vec) * 1e3

# mid_gnd2 = (max_gnd2 + min_gnd2) / 2 # rejected. median being used instead. small variation between the two. 
# print('max, min, med, mid: ', max_gnd2, min_gnd2, med_gnd2, mid_gnd2)
print('z: max, min, med: ', max_gnd2, min_gnd2, med_gnd2)
print('z deviations: ', max_gnd2 - med_gnd2, med_gnd2 - min_gnd2)
print('resolution z [min max median mean]: ', resolution1_min, resolution1_max, resolution1_med, resolution1_mean)

# normalize values

fib_dat_exp1 = normalize_vector(fib_dat_exp1)
fib_dat_gnd1 = normalize_vector(fib_dat_gnd1)
fib_dat_exp2 = normalize_vector(fib_dat_exp2)
fib_dat_gnd2 = normalize_vector(fib_dat_gnd2)

# print('sizes: ', len(fib_dat_exp1), len(fib_dat_gnd1), len(fib_dat_exp2), len(fib_dat_gnd2))

# plot graphs. 1: rotation - movement 1. 2: translation - movement 2
t1 = np.linspace(0,30,len(fib_dat_exp1))
t2 = np.linspace(0,30,len(fib_dat_exp2))
# print('sizes t: ', len(t1), len(t2))

plt.figure()
plt.plot(t1,fib_dat_exp1, 'r')
plt.plot(t1,fib_dat_gnd1, 'r--')
plt.plot(t2,fib_dat_exp2, 'b')
plt.plot(t2,fib_dat_gnd2, 'b--')
plt.legend(["Ry pred","Ry gnd","Tz pred","Tz gnd"], loc="upper right")
plt.ylabel("Normalised motion")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
