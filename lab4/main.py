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

def plot_interval(y, x, color='b', label1=""):
    if (x == 1):
        plt.vlines(x, y[0], y[1], color, lw=1, label = label1)
    else:
        plt.vlines(x, y[0], y[1], color, lw=1)

def load_octave(filename):
    A = 0
    B = 0
    w = []
    with open(filename) as f:
        A, B = [float(t) for t in f.readline().split()]
        for line in f.readlines():
            w.append(float(line))
    return A, B, w

def plotRect(plt, x1, y1, x2, y2, label, color):
    plt.add_patch(Rectangle((x1,y1),x2-x1,y2-y1, label= label, color=color))

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

if __name__ == "__main__":
    data1 = load_csv('lab4/data/Ch1_800nm_0.23mm.csv')
    data1_A, data1_B, data1_w = load_octave('lab4/data/Ch1.txt')
    data_n = [t for t in range(1, len(data1) + 1)]
    
    data1_notequal = [[data1[i][0] - data1_w[i] * EPS, data1[i][1] + data1_w[i] * EPS] for i in range(len(data1))]
        
    for i in range(len(data1_notequal)):
       plot_interval(data1_notequal[i], data_n[i], 'C1', "$I_1$")
    plt.legend()
    plt.title('Data from experiment')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("lab4/report/pics/input.png")
    plt.figure()
    max_w = max(data1_w)
    
    data1 = [[data1[i][0] - max_w * EPS, data1[i][1] + max_w * EPS] for i in range(50, 151)]
    data_n = [t for t in range(50, 151)]
    
    for i in range(len(data1)):
       plot_interval(data1[i], data_n[i], 'C1', "$I_1$")
    plt.legend()
    plt.title('Data with constant coefficient eps * w')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("lab4/report/pics/max_w.png")
    plt.figure()

    acceptable_b1 = []
    acceptable_b0 = []
    for i in range(len(data1)):
        for j in range(i + 1, len(data1)):
            for k in range(0, 2):
                for m in range(0, 2):
                    p1 = [data_n[i], data1[i][k]]
                    p2 = [data_n[j], data1[j][m]]
                    
                    b1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    b0 = p1[1] - b1 * data_n[i]
                    
                    good = True
                    for n in range(len(data1)):
                        if b1 * data_n[n] + b0 < data1[n][0] or b1 * data_n[n] + b0 > data1[n][1]:
                            good = False
                            break
                    if good:
                        print(i, j, k, m, b0, b1)
                        acceptable_b1.append(b1)
                        acceptable_b0.append(b0)
    
    tmp = zip(acceptable_b0, acceptable_b1)
    tmp = list(set(tmp))
    acceptable_b0, acceptable_b1 = list(zip(*tmp))
    acceptable_b0 = list(acceptable_b0)
    acceptable_b1 = list(acceptable_b1)
    print(acceptable_b1)
    print(acceptable_b0)
        
    n_0 = [acceptable_b0[2],acceptable_b0[3],acceptable_b0[0],acceptable_b0[4],acceptable_b0[7],acceptable_b0[9],acceptable_b0[1],acceptable_b0[5],acceptable_b0[8],acceptable_b0[6]]
    n_1 = [acceptable_b1[2],acceptable_b1[3],acceptable_b1[0],acceptable_b1[4],acceptable_b1[7],acceptable_b1[9],acceptable_b1[1],acceptable_b1[5],acceptable_b1[8],acceptable_b1[6]]
    acceptable_b0 = n_0
    acceptable_b1 = n_1
    
    polygon_b0 = acceptable_b0.copy()
    polygon_b1 = acceptable_b1.copy()
    polygon_b0.append(polygon_b0[0])
    polygon_b1.append(polygon_b1[0])
    mx_b0 = max(acceptable_b0)
    mx_b1 = max(acceptable_b1)
    mn_b0 = min(acceptable_b0)
    mn_b1 = min(acceptable_b1)
    rect_x = [mn_b0, mn_b0, mx_b0, mx_b0, mn_b0]
    rect_y = [mn_b1, mx_b1, mx_b1, mn_b1, mn_b1]
    
    plt.plot(rect_x, rect_y, label="Bound Box", color="red")
    plt.plot(polygon_b0, polygon_b1, color="k")
    plt.fill(polygon_b0, polygon_b1, color="blue")
    plt.scatter(acceptable_b0, acceptable_b1, label = "Informative set", color="k")
    plt.xlabel('b0')
    plt.ylabel('b1')
    plt.savefig("lab4/report/pics/I.png")
    plt.figure()
    
    for i in range(len(data1)):
       plot_interval(data1[i], data_n[i], 'C1', "$I_1$")
       
    for i in range(len(acceptable_b0)):
        plt.plot([data_n[0], data_n[len(data1) - 1]], [acceptable_b0[i] + data_n[0] * acceptable_b1[i], acceptable_b0[i] + data_n[len(data1) - 1] * acceptable_b1[i]], label = "vertex " + str(i) + " from Informative set")
    plt.legend()
    plt.title('Corridor')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("lab4/report/pics/corridor.png")
    plt.figure()
    
    center_b0 = sum(acceptable_b0) / len(acceptable_b0)
    center_b1 = sum(acceptable_b1) / len(acceptable_b1)
    
    print(center_b1, center_b0)
    
    for i in range(len(data1)):
       plot_interval(data1[i], data_n[i], 'C1', "$I_1$")
       
    plt.plot([data_n[0], data_n[len(data1) - 1]], [center_b0 + data_n[0] * center_b1, center_b0 + data_n[len(data1) - 1] * center_b1], label = "center of mass of informative set")
    plt.legend()
    plt.title('center of mass')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("lab4/report/pics/center_of_mass.png")
    plt.figure()
    
    predict_points = [25, 175, 101.5]
    
    for point in predict_points:
        positions = []
        for i in range(len(acceptable_b0)):
            x_arr = [point-2, point+2]
            y_arr = [[acceptable_b0[i] + x * acceptable_b1[i]] for x in x_arr]
            positions.append(acceptable_b0[i] + point * acceptable_b1[i])
            plt.plot(x_arr, y_arr, label = "vertex " + str(i) + " from Informative set")        
        
        mid_p = (max(positions) + min(positions)) / 2
        rad_p = max(abs(max(positions) - mid_p), abs(min(positions) - mid_p))
        
        print(point, mid_p, rad_p, mid_p - rad_p, mid_p + rad_p)
        
        print("Jakkar", countJakkar([data1_notequal[int(point - 1)], [mid_p - rad_p, mid_p + rad_p]]))
        mid_p = (max(data1_notequal[int(point - 1)]) + min(data1_notequal[int(point - 1)])) / 2
        rad_p = max(abs(max(data1_notequal[int(point - 1)]) - mid_p), abs(min(data1_notequal[int(point - 1)]) - mid_p))
        print(data1_notequal[int(point - 1)], mid_p, rad_p)

        
        plt.vlines(point, mid_p-rad_p, mid_p+rad_p, "k", lw=1, label = "predicted interval")
        plt.scatter(point, mid_p, label = "predicted mid")
        
        plt.legend()
        plt.title("predict " + str(point))
        plt.xlabel('n')
        plt.ylabel('mV')
        plt.savefig("lab4/report/pics/predict" + str(point) + ".png")
    
    plt.show()