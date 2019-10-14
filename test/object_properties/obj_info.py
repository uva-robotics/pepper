# -*- coding: future_fstrings -*-

from PIL import Image

from statistics import mode

import json

import numpy as np

import os
import webcolors

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt


def get_color_name(rgb_triplet): # snatched

    min_colors = {}
    for key, name in webcolors.css21_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colors[(rd + gd + bd)] = name

    return min_colors[min(min_colors.keys())]


def biggest_cluster(result):

    return mode([res[1] for res in result])


def closest_cluster(result, clusters):

    depth_dict = dict()

    for cluster in clusters:
        relevant = [res for res in result if res[1] == cluster]
        avg_depth = sum([res[0][3] for res in relevant]) / len(relevant)
        depth_dict[cluster] = avg_depth

    return min(depth_dict, key=depth_dict.get)


def middle_cluster(result, clusters):

    position_dict = dict()

    for cluster in clusters:

        relevant = [res for res in result if res[1] == cluster]

        avg_4 = sum([res[0][4] for res in relevant]) / len(relevant)
        avg_5 = sum([res[0][5] for res in relevant]) / len(relevant)

        ### TODO: Euclidian distance or something smoother...?
        diff_4 = abs(0.5 * max([res[0][4] for res in result]) - avg_4)
        diff_5 = abs(0.5 * max([res[0][5] for res in result]) - avg_5)

        total_diff = diff_4 + diff_5

        position_dict[cluster] = total_diff

    return min(position_dict, key=position_dict.get)



def main(obj_path, name):

    depth = os.path.join(obj_path, 'depth.npy')
    meta = os.path.join(obj_path, 'meta.json')
    rgb = os.path.join(obj_path, 'rgb.png')

    with open(meta) as json_file:
        meta_info = json.load(json_file)
        object_type = meta_info['name']
        confidence = meta_info['confidence']

    obj_depth = np.load(depth)

    # TODO: add white balancing
    img = Image.open(rgb)
    data = np.array(img)

    ### included depth!!!

    combined = np.dstack((data, obj_depth))

    new_data = combined.reshape(-1, 4)

    dbscan = DBSCAN(eps=5, min_samples=50)
    db = dbscan.fit(new_data)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    rows, cols, chs = combined.shape

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(np.reshape(labels, [rows, cols]))
    plt.axis('off')
    #plt.show()

    indices = np.dstack(np.indices(combined.shape[:2]))
    xycolors = np.concatenate((combined, indices), axis=-1)
    all_dim = np.reshape(xycolors, [-1, 6])

    result = zip(all_dim, labels)
    clusters = set([res[1] for res in result])

    biggest = biggest_cluster(result)
    closest = closest_cluster(result, clusters)
    middle = middle_cluster(result, clusters)

    dom_dict = dict()
    
    for cluster in clusters:
        dom_dict[cluster] = [biggest, closest, middle].count(cluster)

    dom_cluster = max(dom_dict, key=dom_dict.get)

    rgb_triplets = [res[0] for res in result if res[1] == dom_cluster]

    r_list = [triplet[0] for triplet in rgb_triplets]
    g_list = [triplet[1] for triplet in rgb_triplets]
    b_list = [triplet[2] for triplet in rgb_triplets]

    avg_r = sum(r_list) / len(r_list)
    avg_g = sum(g_list) / len(g_list)
    avg_b = sum(b_list) / len(b_list)

    avg_triplet = (int(avg_r), int(avg_g), int(avg_b))

    try:
        color = webcolors.rgb_to_name(avg_triplet)
    except ValueError:
        color = get_color_name(avg_triplet)

    print(f'I see a {color} {object_type} with {confidence} confidence.')

    #plt.savefig(f'./results/{color}_{object_type}_{name}')


if __name__ == '__main__':

    ROOT = './data/20190930_133826'

    for p in os.listdir(ROOT):
        pathname = os.path.join(ROOT, p)

        if os.path.isdir(pathname):

            for name in os.listdir(pathname):
                obj_path = os.path.join(pathname, name)
                if os.path.isdir(obj_path):
                    main(obj_path, name)
