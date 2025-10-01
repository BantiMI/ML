import numpy as np
import CNN_lib as lib
image = [[1, 1, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 1, 1],
         [0, 0, 1, 1, 0],
         [0, 1, 1, 0, 0]]

lib.padding_start_img(image)

filts = []
for i in range(10):
    W1 = []
    for k in range(3):
        W1.append(np.random.randint(-5, 5, size=3))

    filts.append(W1)

neob_k_p = lib.get_raw_feature_map(image, filts)

# for k in range(len(neob_k_p)):
#     print(neob_k_p[k])

ob_k_p = lib.get_processed_feature_map(neob_k_p)

# print("до padding_obr2")
# for k in range(len(ob_k_p)):
#     print(ob_k_p[k])

lib.padding_feature_maps(ob_k_p)

# print("До maxpool список")
# for k in range(len(ob_k_p)):
#     print(ob_k_p[k])

posle_maxpooling = lib.maxpooling(ob_k_p)


lib.padding_start_img(posle_maxpooling, 3)

# print("Уже maxpool список")
# for k in range(len(posle_maxpooling)):
#     print(posle_maxpooling[k])

# print("второй слой: \n")
# print("Фильтры второго слоя: ")
filts2 = []
for _ in range(20):
    w2 = []
    for i in range(10):
        W1 = []
        for k in range(3):
            W1.append([int(x) for x in np.random.randint(-5, 5, size=3)])
        w2.append(W1)
    filts2.append(w2)

# for i in range(len(filts2)):
#     for j in range(len(filts2[i])):
#         print(filts2[i][j])


neob_k_p2 = lib.get_raw_feature_map(posle_maxpooling, filts2, 3)
# print("\n 3 мерная матрица фильтров")
# for i in range(len(neob_k_p2[0])):
#     print(neob_k_p2[0][i])

neob_k_p2_2 = []

for k_p2 in neob_k_p2:
    vr_neob_k_p2_2 = []
    for i in range(len(k_p2[0])):
        a = []
        for j in range(len(k_p2[0][i])):
            sum_el = 0.0
            for cloy in range(len(k_p2)):
                sum_el+=k_p2[cloy][i][j]
            a.append(sum_el)
        vr_neob_k_p2_2.append(a)
    neob_k_p2_2.append(vr_neob_k_p2_2)
# print("\n итоговая двумерная карта признаков 2 слоя")
# for i in range(len(neob_k_p2_2)):
#     print(neob_k_p2_2[i])


ob_k_p2 = lib.get_processed_feature_map(neob_k_p2_2)
# print("\n итоговая обработанная двумерная карта признаков 2 слоя")
# for i in range(len(ob_k_p2)):
#     print(ob_k_p2[i])


lib.padding_feature_maps(ob_k_p2)

# print("\nДо maxpool список")
# for k in range(len(ob_k_p2)):
#     print(ob_k_p2[k])

posle_maxpooling2 = lib.maxpooling(ob_k_p2)

print("\nУже maxpool список")
for k in range(len(posle_maxpooling2)):
    print(posle_maxpooling2[k])