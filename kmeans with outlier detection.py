import csv
from random import sample
from math import sqrt
from numpy import zeros, linspace, zeros_like
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

def assign_closest_centroid(data, centroids):
    clusters = []
    count = 0
    for item in data:
        d_min = 1e9
        closest = 0
        c=0
        for centroid in centroids:
            d = 0
            for i in range(len(centroid)):
                d += (item[i] - centroid[i])**2
            d = sqrt(d)
            if d_min > d:
                d_min = d
                closest = c
            c+=1
        clusters.append([closest, d_min, count])
        count+=1
    return clusters

def compute_d_avg(clusters, k, gamma):
    c = 0
    d = 0
    for item in clusters:
        if item[0] != k:
            d = (d / (c+1)) * c + item[1]/(c+1)
            c+=1
    return d*gamma

def sort_by_second(item):
    return item[1]

def sort_by_third(item):
    return item[2]

def update_clusters_kmor(data, centroids, n_zero, k, gamma):
    clusters = assign_closest_centroid(data, centroids)
    d_avg = compute_d_avg(clusters, k, gamma)
    clusters.sort(key=sort_by_second, reverse=True)
    for i in range(n_zero):
        if clusters[i][1] > d_avg:
            clusters[i][0] = k
            clusters[i][1] = -1
    clusters.sort(key=sort_by_third)
    return clusters

def update_clusters_odc(data, centroids, k, gamma, old_clusters):
    clusters = assign_closest_centroid(data, centroids)
    d_avg = compute_d_avg(clusters, k, gamma)
    for i in range(len(clusters)):
        if clusters[i][1] > d_avg or old_clusters[i][0] == k:
            clusters[i][0] = k
            clusters[i][1] = -1
    return clusters

# update centroids given data and their assigned cluster
def update_centroids(data, clusters, k):
    centroids = []
    for i in range(k):
        centroid = []
        for w in range(len(data[0])):
            centroid.append(0)
        c = 0
        for j in range(len(clusters)):
            if clusters[j][0] == i:
                for f in range(len(data[0])):
                    centroid[f]= ((centroid[f] /(c+1))*c + data[j][f]/(c+1))
                c+=1
        centroids.append(centroid)
    return centroids

def update_distance(data, clusters, centroids):
    for item in range(len(data)):
        d = 0
        if clusters[item][0] == len(centroids):
            continue
        for i in range(len(data[0])):
            d += (data[item][i] - centroids[clusters[item][0]][i])**2
        d = sqrt(d)
        clusters[item][1] = d
    return clusters

def compute_p(clusters, d_avg, k):
    P = 0
    for item in clusters:
        if item[0] != k:
            P += item[1]
        else:
            P += d_avg
    return P

def get_outliers(clusters, k):
    outliers = []
    for item in clusters:
        if item[0] == k:
            outliers.append(item)
    return outliers

def permutation(lst):
    if len(lst) == 0:
        return []
    if len(lst) == 1:
        return [lst]
    l = []
    for i in range(len(lst)):
        m = lst[i]
        remLst = lst[:i] + lst[i+1:]
        for p in permutation(remLst):
            l.append([m] + p)
    return l

def best_ME(predicted, labels):
    dummy = []
    for i in range(1,k+1):
        dummy.append(i)
    permutations = permutation(dummy)
    new_clusters = zeros_like(predicted)
    M_best =2
    for perm in permutations:
        M_total = 0
        classes = []
        for item in predicted:
            f = 1
            for i in range(k):
                if item == i:
                    classes.append(perm[i])
                    f = 0
            if f == 1:
                classes.append(k+1)
        mat = confusion_matrix(labels, classes)
        for a in range(k+1):
            M = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            count = 0
            for b in range(k+1):
                for c in range(k+1):
                    if a == c:
                        count += mat[b][c]
                    if a == b and a == c:
                        tp += mat[b][c]
                    elif c == a:
                        fn += mat[b][c]
                    elif b == a:
                        fp += mat[b][c]
                    else:
                        tn += mat[b][c]
            #print(tp,fp,tn,fn)
            M = sqrt((1-(tp/(tp+fp)))**2+(fp/(tn+fp))**2)
            M_total += M*count/len(labels)
        if M_total < M_best:
            M_best= M_total
    return M_best

def KMOR(dataset, gamma, k, max_outlier_ratio, number_of_executions):
    data = []
    labels = []
    # Split data and labels
    with open(dataset, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for item in csvreader:
            dummylist = [float(i) for i in item]
            data.append(dummylist[1:-1])
            labels.append(dummylist[-1])

    labels = [int(i) for i in labels]
    # Set Model variables
    n_zero = int(len(data)/max_outlier_ratio)
    max_iteration = 100
    theta = 1e-6
    T = number_of_executions

    R_avg = 0
    M_avg = 0
    O_avg = 0
    for s in range(T):
        # Inintiate centroids randomly
        centroids = sample(data,k)

        P = 0
        t = 0
        clusters = assign_closest_centroid(data, centroids)
        for i in range(max_iteration):
            clusters = update_clusters_kmor(data, centroids, n_zero, k, gamma)
            centroids = update_centroids(data, clusters, k)
            clusters = update_distance(data, clusters, centroids)
            d_avg = compute_d_avg(clusters, k, gamma)
            P_new = compute_p(clusters, d_avg, k)
            if abs(P_new-P) < theta:
                break
            P = P_new
            t+=1

        O = len(get_outliers(clusters, k))
        O_avg += O
        predicted = []
        for i in clusters:
            predicted.append(i[0])
        R = adjusted_rand_score(predicted, labels)
        R_avg += R
        M_best = best_ME(predicted,labels)
        M_avg += M_best
    print(R_avg/T, M_avg/T, O_avg/T)

def kmeansmm(dataset, k, max_outlier_ratio, number_of_executions):
    KMOR(dataset, 0, k, max_outlier_ratio, number_of_executions)

def ODC(dataset, gamma, k, number_of_executions):
    data = []
    labels = []
    # Split data and labels
    with open(dataset, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for item in csvreader:
            dummylist = [float(i) for i in item]
            data.append(dummylist[1:-1])
            labels.append(dummylist[-1])

    labels = [int(i) for i in labels]
    # Set Model variables
    max_iteration = 100
    theta = 1e-6
    T = number_of_executions

    R_avg = 0
    M_avg = 0
    O_avg = 0
    for s in range(T):
        # Inintiate centroids randomly
        centroids = sample(data,k)

        P = 0
        t = 0
        clusters = assign_closest_centroid(data, centroids)
        for i in range(max_iteration):
            clusters = update_clusters_odc(data, centroids, k, gamma, clusters)
            centroids = update_centroids(data, clusters, k)
            clusters = update_distance(data, clusters, centroids)
            d_avg = compute_d_avg(clusters, k, gamma)
            P_new = compute_p(clusters, d_avg, k)
            if abs(P_new-P) < theta:
                break
            P = P_new
            t+=1

        O = len(get_outliers(clusters, k))
        O_avg += O
        predicted = []
        for i in clusters:
            predicted.append(i[0])
        R = adjusted_rand_score(predicted, labels)
        R_avg += R
        M_best = best_ME(predicted,labels)
        M_avg += M_best
    print(R_avg/T, M_avg/T, O_avg/T)


if __name__ == '__main__':
    dataset = 0
    gamma = 0
    k = 0
    while True:
        inp = input("Enter 1 for BCW and 2 for shuttle: ")
        if inp == '1':
            dataset = 'datasets/breast-cancer-wisconsin-no-miss.csv'
            gamma = 1
            k = 1
            max_outlier_ratio = 2
            number_of_executions = 10
            break
        elif inp == '2':
            dataset = "datasets/shuttle_normal.csv"
            gamma = 4.1
            k = 3
            max_outlier_ratio = 10
            number_of_executions = 10
            break

    while True:
        inp = input("Enter 1 for KMOR and 2 for k-means-- and 3 for ODC: ")
        if inp == '1':
            KMOR(dataset, gamma, k, max_outlier_ratio, number_of_executions)
            break
        elif inp == '2':
            kmeansmm(dataset, k, max_outlier_ratio, number_of_executions)
            break
        elif inp == '3':
            ODC(dataset, gamma, k, number_of_executions)
            break
