
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import pairwise_distances, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def calculate_sigma(d):
    return d / np.sqrt(-np.log(0.5))

def gaussian_membership(x_k, x, d):
    """

    :param x_k: 
    :param x: 
    :param d: 
    :return: 
    """
    sigma = calculate_sigma(d)
    distance = euclidean_distance(x, x_k)
    membership = np.exp(-((distance / sigma) ** 2))
    return membership



def find_neighborhood_radius(X, y):
    n_samples = X.shape[0]
    eta_list = []


    distances = pairwise_distances(X, metric='euclidean')

    for i in range(n_samples):
        label_x = y[i]
        same_label_indices = np.where(y == label_x)[0]
        different_label_indices = np.where(y != label_x)[0]

        same_label_indices = same_label_indices[same_label_indices != i]

        if len(same_label_indices) > 0:
            nho_distance = np.min(distances[i, same_label_indices])
        else:
            nho_distance = np.inf

        nhe_distance = np.min(distances[i, different_label_indices])
        eta = nhe_distance - 0.01 * nho_distance
        eta_list.append(eta)
    return eta_list



# def find_optimal_alpha(membership, lam):
#     # Sort the membership values
#     membership_sorted = np.sort(membership)
#
#
#     def compute_loss(alpha):
#         loss1 = np.sum(membership[membership <= alpha])  # 对 membership <= alpha 的部分求和
#         loss2 = np.sum(0.5 - membership[membership > alpha])  # 对 membership > alpha 的部分求和
#         return lam * loss1 + loss2
#
#     # Search for optimal alpha among unique membership values
#     alpha_candidates = np.unique(membership_sorted)
#     losses = np.array([compute_loss(alpha) for alpha in alpha_candidates])
#
#     # Find the alpha corresponding to the minimum loss
#     min_loss_idx = np.argmin(losses)
#     optimal_alpha = alpha_candidates[min_loss_idx]
#
#     return optimal_alpha


def find_optimal_alpha(membership, lam):
    # Adjust membership values: for values greater than 0.5, transform them to 1 - membership
    membership_adjusted = np.where(membership > 0.5, 1 - membership, membership)

    # Sort the adjusted membership values
    membership_sorted = np.sort(membership_adjusted)

    # Vectorized loss function
    def compute_loss(alpha):
        loss1 = np.sum(membership_adjusted[membership_adjusted <= alpha])  # 对 membership <= alpha 的部分求和
        loss2 = np.sum(0.5 - membership_adjusted[membership_adjusted > alpha])  # 对 membership > alpha 的部分求和
        return lam * loss1 + loss2

    # Search for optimal alpha among unique membership values
    alpha_candidates = np.unique(membership_sorted)
    losses = np.array([compute_loss(alpha) for alpha in alpha_candidates])

    # Find the alpha corresponding to the minimum loss
    min_loss_idx = np.argmin(losses)
    optimal_alpha = alpha_candidates[min_loss_idx]

    return optimal_alpha



def construct_granules_by_class(X, y, eta_list):

    optimal_alpha_list = []
    unique_classes = np.unique(y)
    granules_by_class = {label: [] for label in unique_classes}  

    for i in range(len(X)):
        x_k = X[i]
        label = y[i]  
        d = eta_list[i]

               memberships = np.array([gaussian_membership(x_k, X[j], d) for j in range(len(X))])
        # print("memberships", np.round(memberships, 3))



        # optimal_alpha = find_optimal_alpha(memberships, lam=1)
        # optimal_alpha_list.append(optimal_alpha)



        inside_granule_indices = np.where(memberships >= 0.75)[0]


        points = [X[j] for j in inside_granule_indices]


        if len(points) == 0:
            continue  


        new_distances = pairwise_distances(points, [x_k], metric='euclidean').flatten()
        new_radius = np.max(new_distances)


        granule = {
            "center": x_k,
            "radius": new_radius,  
            "points": points,  
            "label": label,
            "optimal_alpha": 0.75          }

        granules_by_class[label].append(granule)

    return granules_by_class



def merge_homogeneous_granules(granules_by_class):
       for label, granules in granules_by_class.items():
        i = 0
        while i < len(granules):
            granule_i = granules[i]
            center_i = granule_i['center']
            radius_i = granule_i['radius']
            points_i = granule_i['points']

            j = i + 1

            while j < len(granules):
                granule_j = granules[j]
                center_j = granule_j['center']
                radius_j = granule_j['radius']
                points_j = granule_j['points']

                                distance = np.linalg.norm(center_i - center_j)

    
                if distance < min(radius_i, radius_j):
   
                    new_center = (center_i + center_j) / 2
                    new_radius = min(radius_i, radius_j)

                  
                    merged_points = points_i + points_j
                    merged_points = np.vstack(merged_points)

       
                    distances = np.linalg.norm(merged_points - new_center, axis=1)

                    filtered_points = merged_points[distances <= new_radius]

                                        if filtered_points.size == 0:


                        granules.pop(j)
                        granules.pop(i)
                        i -= 1  
                        break  
                    else:
     
                        new_granule = {
                            "center": new_center,
                            "radius": new_radius,
                            "points": filtered_points,
                            "label": label
                        }

    
                        granules[i] = new_granule
                        granules.pop(j)  
                else:
                    j += 1

            i += 1

    return granules_by_class


def process_heterogeneous_granules(granules_by_class):
    labels = list(granules_by_class.keys())
    num_labels = len(labels)

    for idx_i in range(num_labels):
        label_i = labels[idx_i]
        granules_i = granules_by_class[label_i]

        for idx_j in range(idx_i + 1, num_labels):
            label_j = labels[idx_j]
            granules_j = granules_by_class[label_j]

            to_remove_i = set()
            to_remove_j = set()

            for granule_idx_i, granule_i in enumerate(granules_i):
                center_i = np.array(granule_i['center'])
                radius_i = granule_i['radius']
                points_i = np.array(granule_i['points'])

                for granule_idx_j, granule_j in enumerate(granules_j):
                    center_j = np.array(granule_j['center'])
                    radius_j = granule_j['radius']
                    points_j = np.array(granule_j['points'])


                    distance = np.linalg.norm(center_i - center_j)

                    # if abs(radius_i - radius_j) < distance < radius_i + radius_j:
                    if  distance < radius_i + radius_j:

                        if radius_i > radius_j:

                            new_radius_i = distance - radius_j
                            radius_i = new_radius_i
                        else:

                            new_radius_j = distance - radius_i
                            radius_j = new_radius_j


                        distances_i = np.linalg.norm(points_i - center_i, axis=1)
                        distances_j = np.linalg.norm(points_j - center_j, axis=1)

                        granule_i['points'] = points_i[distances_i <= radius_i].tolist()
                        granule_j['points'] = points_j[distances_j <= radius_j].tolist()


                        granule_i['radius'] = radius_i
                        granule_j['radius'] = radius_j

                    elif distance < abs(radius_i - radius_j):
                                    
                                    if radius_i > radius_j:
                                        to_remove_j.add(granule_idx_j)
                                    else:
                                        to_remove_i.add(granule_idx_i)


                    granules_i = [g for idx, g in enumerate(granules_i) if idx not in to_remove_i]
                    granules_j = [g for idx, g in enumerate(granules_j) if idx not in to_remove_j]
                    granules_by_class[label_i] = granules_i
                    granules_by_class[label_j] = granules_j


    return granules_by_class



def update_granules(granules_by_class):
        final_granules_by_class = {}


    updated_granules_by_class = merge_homogeneous_granules(granules_by_class)
    updated_granules_by_class = process_heterogeneous_granules(updated_granules_by_class)

    for label, granules in updated_granules_by_class.items():
        updated_granules = []
        radii = [g['radius'] for g in granules]
        mean_radius = np.mean(radii)
        median_radius = np.median(radii)
        threshold_radius = 2 * max(mean_radius, median_radius)

        for granule in granules:
            if granule['radius'] == 0:
                continue
            if len(granule['points']) <= 1:
                continue
            updated_granules.append(granule)

        final_granules_by_class[label] = updated_granules


    total_granules = sum(len(granules) for granules in final_granules_by_class.values())
    for label, granules in final_granules_by_class.items():
        print(f" {label} : {len(granules)}")



    return final_granules_by_class






def knn_classify(test_samples, granules_by_class):
    predictions = []

    for test_sample in test_samples:
        nearest_distance = float('inf')
        nearest_label = None
        nearest_granules = []  


        for label, granules in granules_by_class.items():
            for granule in granules:
                center = granule['center']
                radius = granule['radius']
                points = granule['points']
                distance = np.linalg.norm(test_sample - center) - radius

                                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_label = label
                    nearest_granules = [(label, granule)]

                elif distance == nearest_distance:
                    nearest_granules.append((label, granule))

                if len(nearest_granules) > 1:
            max_density = -1
            for label, granule in nearest_granules:

                density = len(granule['points']) / granule['radius'] if granule['radius'] > 0 else 0
                if density > max_density:
                    max_density = density
                    nearest_label = label

        predictions.append(nearest_label)

    return predictions










def check_coverage(granules_by_class, original_data):


    original_samples_set = set(tuple(sample) for sample in original_data)


    covered_samples_set = set()


    for label, granules in granules_by_class.items():
        for granule in granules:
            for point in granule['points']:
                covered_samples_set.add(tuple(point))  

    is_covered = original_samples_set.issubset(covered_samples_set)



    return len(covered_samples_set), is_covered


def new_granule_knn_algorithm(train, test):
    

    eta_list = find_neighborhood_radius(train[:, 1:], train[:, 0])


    granules_by_class = construct_granules_by_class(train[:, 1:], train[:, 0], eta_list)


    granules_by_class_updated = update_granules(granules_by_class)


    is_covered = check_coverage(granules_by_class_updated, train)



    y_pred = knn_classify(test[:, 1:], granules_by_class_updated)


    accuracy = accuracy_score(test[:, 0], y_pred)
    precision = precision_score(test[:, 0], y_pred, average='macro')
    recall = recall_score(test[:, 0], y_pred, average='macro')
    f1 = f1_score(test[:, 0], y_pred, average='macro')

   
    return accuracy,precision,recall,f1


