import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EPS = 1e-4

def load_csv(filename):
    data1 = [] 
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            data1.append([float(row[0]), float(row[0])])
    return data1

def load_octave(filename):
    A = 0
    B = 0
    w = []
    with open(filename) as f:
        A, B = [float(t) for t in f.readline().split()]
        for line in f.readlines():
            w.append(float(line))
    return A, B, w

def plot_interval(y, x, color='b', label1=""):
    if (x == 1):
        plt.vlines(x, y[0], y[1], color, lw=1, label = label1)
    else:
        plt.vlines(x, y[0], y[1], color, lw=1)
        
def countJakkar(all_data):
        min_inc = list(all_data[0])
        max_inc = list(all_data[0])
        for interval in all_data:
            min_inc[0] = max(min_inc[0], interval[0])
            min_inc[1] = min(min_inc[1], interval[1])
            max_inc[0] = min(max_inc[0], interval[0])
            max_inc[1] = max(max_inc[1], interval[1])
        JK = (min_inc[1] - min_inc[0]) / (max_inc[1] - max_inc[0])
        return JK

def plot_interval_hist(x, color='b', label1=""):
    hist = [(t[1] + t[0]) / 2 for t in x]

    plt.hist(hist, color = color, label = label1, rwidth=0.8)
    
def calc_left_right(array, index):
    left = 0
    right = 0
    for i in range(index):
        left += array[i]
    for i in range(index + 1, len(array)):
        right += array[i]
    return (left, right)
    
def plotRect(plt, x1, y1, x2, y2, label, color):
    plt.add_patch(Rectangle((x1,y1),x2-x1,y2-y1, label= label, color=color))

if __name__ == "__main__":
    data1 = load_csv('lab3/data/Ch1_800nm_0.23mm.csv')
    data1_A, data1_B, data1_w = load_octave('lab3/data/Ch1.txt')
    data_n = [t for t in range(1, len(data1) + 1)]
    data1 = [[data1[i][0] - EPS, data1[i][1] + EPS] for i in range(len(data1))]

    for i in range(len(data1)):
       plot_interval(data1[i], data_n[i], 'C1', "$I_1$")
    plt.legend()
    plt.title('Data from experiment')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("lab3/report/pics/input.png")
    plt.figure()
    
    data1 = load_csv('lab3/data/Ch1_800nm_0.23mm.csv')
    data1 = [[data1[i][0] - EPS - data1_B * i, data1[i][1] - data1_B * i + EPS] for i in range(len(data1))]
    for i in range(len(data1)):
       plot_interval(data1[i], data_n[i], 'C1', "$I_1$")
    plt.legend()
    plt.title('Data from experiment without linear regression')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("lab3/report/pics/input_minus_linear.png")
    plt.figure()    

    borders = set()
    for it in data1:
        borders.add(it[0])
        borders.add(it[1])

    borders = list(borders)
    borders.sort()
    
    data_z = [[borders[i], borders[i+1]] for i in range(len(borders) - 1)]
    data_mu = [0] * len(data_z)
    
    for i, z_i in enumerate(data_z):
        for x_i in data1:
            if (x_i[0] <= z_i[0] and x_i[1] >= z_i[1]):
                data_mu[i] += 1
    
    mu = max(data_mu)
    
    moda = []
    median = None
    prev_left, prev_right = 0, 0
    for i, z_i in enumerate(data_z):
        if (data_mu[i] == mu):
            moda.append(z_i)
        left, right = calc_left_right(data_mu, i)
        if (median == None):
            if left == right:
                median = z_i
            elif left + data_mu[i] == right:
                median = [(z_i[0] + data_z[i + 1][0]) / 2, (z_i[1] + data_z[i + 1][1]) / 2]
            elif prev_left < prev_right and left > right:
                median = [(z_i[0] + data_z[i + 1][0]) / 2, (z_i[1] + data_z[i + 1][1]) / 2]
        prev_left, prev_right = left, right
    print("Moda", moda)
    print("Median", median)
    
    plt.stairs(data_mu, borders, label="$\mu_i$")
    for md in moda:
        plt.stairs([mu], md, label="moda", color="red")
    plt.stairs([mu], median, label="median")
    plt.legend()
    plt.title('$\mu_i$ histogram')
    plt.savefig("lab3/report/pics/mu_hist.png")
    plt.figure()

    print(countJakkar(data1))
    
    for i in range(len(data1)):
       plot_interval(data1[i], data_n[i], 'C1', "$I_1$")
    for md in moda:
        plotRect(plt.gca(), data_n[0], md[0], data_n[-1], md[1], "moda", "red")
    plotRect(plt.gca(), data_n[0], median[0], data_n[-1], median[1], "median", "green")
    plt.legend()
    plt.title('Data with median and moda')
    plt.savefig("lab3/report/pics/data_med_mod.png")
    
    plt.show()