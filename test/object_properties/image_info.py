# -*- coding: future_fstrings -*-

from PIL import Image, ImageColor
from statistics import mode

import numpy as np

import os
import webcolors

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


ROOT = './Images'

for img_name in os.listdir(ROOT):
    pass

#img_name = 'chair_154547347345997269585187269222787520331.jpg'     TEAL
#img_name = 'potted plant_148305230774084591629901217846242062572.jpg'      BLACK
#img_name = 'banana_54911710780589897412953802260948443365.jpg' GRAY
#img_name = 'bottle_225131438382852575702210306912782962173.jpg' TEAL


##### If BLACK --> check how close to black it actually is and manipulate the numbers


path = os.path.join(ROOT, img_name)
img = Image.open(path)

data = np.array(img)

print(data.shape)

#print(np.mean(data, axis=(0,1)))

new_data = data.reshape(-1, 3)

print(new_data.shape)

#indices = np.dstack(np.indices(data.shape[:2]))
#xycolors = np.concatenate((data, indices), axis=-1)
#newest_data = np.reshape(xycolors, [-1,5])

dbscan = DBSCAN(eps=5, min_samples = 50)
db = dbscan.fit(new_data)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters)

rows, cols, chs = data.shape

plt.figure(2)
plt.subplot(2, 1, 1)
plt.imshow(img)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.show()


indices = np.dstack(np.indices(data.shape[:2]))
xycolors = np.concatenate((data, indices), axis=-1)
all_dim = np.reshape(xycolors, [-1,5])
#print(all_dim)

result = zip(all_dim, labels)

middle_triplets = []

# need more complicated rules if the object is not in the middle

for res in result:
    dims = res[0]
    if int(0.2 * len(indices)) < dims[3] < int(0.8 * len(indices)) and (int(0.2 * len(indices)) < dims[4] < int(0.8 * len(indices))):
        middle_triplets.append(res)

mf = mode([res[1] for res in middle_triplets])

"""
dominant = list()

for label in labels:
    if len([l for l in ])
"""

rgb_triplets = [res[0] for res in result if res[1] == mf]

r_list = [triplet[0] for triplet in rgb_triplets]
g_list = [triplet[1] for triplet in rgb_triplets]
b_list = [triplet[2] for triplet in rgb_triplets]

avg_r = sum(r_list) / len(r_list)
avg_g = sum(g_list) / len(g_list)
avg_b = sum(b_list) / len(b_list)

avg_triplet = (int(avg_r), int(avg_g), int(avg_b))
print(avg_triplet)

def get_color_name(rgb_triplet): # snatched
    min_colors = {}
    for key, name in webcolors.css21_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

try:
    color = webcolors.rgb_to_name(avg_triplet)
except ValueError:
    color = get_color_name(avg_triplet)

print(color)




##### take hues and choose most represented + most centered
##### multi-color objects?
