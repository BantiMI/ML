import numpy as np

def sum_after_filt(p1, p2):
    sum = 0
    for i in range(len(p1[0])):
        for j in range(len(p1)):
            sum += p1[i][j] * p2[i][j]
    return sum

def relu_activation(x):
    return 0 if x < 0 else x

def padding_start_img(matrix, n = 2):
    if n == 2:
        for i in matrix:
            i.insert(0, 0)
            i.append(0)
        matrix.insert(0, [0 for _ in range(len(matrix[1]))])
        matrix.append([0 for _ in range(len(matrix[1]))])
        return matrix
    if n == 3:
        for i in matrix:
            for j in i:
                j.insert(0, 0)
                j.append(0)
            i.insert(0, [0 for _ in range(len(i[1]))])
            i.append([0 for _ in range(len(i[1]))])
        return matrix

def padding_feature_maps(matrix):
    for k in range(len(matrix)):
        x = len(matrix[k])
        y = len(matrix[k][0])
        if x % 2 != 0: matrix[k].append([0 for _ in range(len(matrix[k][1]))])
        if y % 2 != 0:
            for i in matrix[k]:
                i.append(0)
    return matrix

def get_raw_feature_map(image, filts, n = 2):
    if n == 2:
        neob_k_p = []
        for h in range(len(filts)):
            vr_neob_k_p = []
            for i in range(len(image) - 2):
                a = []
                for j in range(len(image[0]) - 2):
                    p2 = []
                    for ch in range(3):
                        p2.append(image[i+ch][j:j+3])
                    a.append(sum_after_filt(p2, filts[h]))
                vr_neob_k_p.append(a)
            neob_k_p.append(vr_neob_k_p)
        return neob_k_p
    if n == 3:
        neob_k_p2 = []
        for h in range(len(filts)):
            vr_neob_k_p = []
            for cloy in range(10):
                vr2_neob_k_p2 = []
                for i in range(len(image[cloy]) - 2):
                    a = []
                    for j in range(len(image[cloy][0]) - 2):
                        p2 = []
                        for ch in range(3):
                            p2.append(image[cloy][i+ch][j:j+3])
                        a.append(sum_after_filt(p2, filts[h][cloy]))
                    vr2_neob_k_p2.append(a)
                vr_neob_k_p.append(vr2_neob_k_p2)
            neob_k_p2.append(vr_neob_k_p)
        return neob_k_p2

def get_processed_feature_map(neob_k_p):
    ob_k_p = []
    for i in range(len(neob_k_p)):
        vr_ob_k_p =[]
        for k in range(len(neob_k_p[i])):
            a = [relu_activation(x) for x in neob_k_p[i][k]]
            vr_ob_k_p.append(a)
        ob_k_p.append(vr_ob_k_p)
    return ob_k_p

def maxpooling(ob_k_p):
    posle_maxpooling = []
    for k in range(len(ob_k_p)):
        vr_posle_maxpooling = []
        for i in range(0, len(ob_k_p[k]), 2):
            a2 = []
            for j in range(0, len(ob_k_p[k][0]), 2):
                a = []
                a.append(ob_k_p[k][i][j:j + 2])
                a.append(ob_k_p[k][i + 1][j:j + 2])
                a = np.array(a)
                a = a.flatten()
                a2.append(float(max(a)))
            vr_posle_maxpooling.append(a2)
        posle_maxpooling.append(vr_posle_maxpooling)
    return posle_maxpooling


