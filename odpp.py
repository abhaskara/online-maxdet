# Command line example
# python3.6 odpp.py {number_columns} {epsilon} {input columns}
import pandas as pd
import numpy as np
import sys
import math
from numpy import linalg as LA

# Function that computes the volume of a matrix
def volume(matrix):
    transpose = matrix.transpose()
    determinant = np.linalg.det(np.matmul(matrix, transpose))
    # Sometimes determinant computation with numpy are faulty if the value is too small.
    # Note that this not effect the reesult almost never(sometimes just in the first column insertion)
    if determinant < 0 and determinant > -0.000001:
        determinant = 0
    return math.sqrt(determinant);

def swap_row(matrix, index, vector):
    tmp_matrix = np.copy(matrix)
    tmp_matrix = np.delete(tmp_matrix, index, 0)
    tmp_matrix = np.vstack((tmp_matrix, vector))
    return tmp_matrix;

def find_candidate_swap(matrix, candidate):
    best_candidate_matrix = np.array([])
    best_candidate_value = 0
    ind = 0
    for j in range(np.shape(matrix)[0]):
        candidate_matrix = swap_row(matrix, j, candidate)
        candidate_value = volume(candidate_matrix)
        if candidate_value >= best_candidate_value:
            best_candidate_value = candidate_value
            best_candidate_matrix = candidate_matrix
            ind = j
    return best_candidate_matrix, best_candidate_value, ind;

def append_row(matrix, row):
    if matrix.size == 0:
        matrix = np.append(matrix, row)
    else:
        matrix = np.vstack((matrix, row))
    return matrix;

data = pd.read_csv((sys.argv[3]), header=None, delimiter=r"\s+")

# Compute max norm
max_norm = 0
for i in range(len(data.index)):
    candidate = np.array(data.iloc[i].values.ravel())
    norm = LA.norm(candidate)
    if norm > max_norm:
        max_norm = norm

# Normalize the data so the all vectors have max norm 1
data = data.div(max_norm)

# Initialize solution for our algorithm and other versions
sol_our = np.array([])
stash = np.array([])
value_our = 0
comp_our = 0
swap_our = 0
sol_th = np.array([])
value_th = 0
comp_th = 0
swap_th = 0
sol_greedy = np.array([])
value_greedy = 0
comp_greedy = 0
swap_greedy = 0
sol_sq_greedy = np.array([])
value_sq_greedy = 0
comp_sq_greedy = 0

k = int(sys.argv[1])
th = float(sys.argv[2])

# We start by computing the solution of sequential Greedy
for j in range(k):
    comp_sq_greedy = comp_sq_greedy + len(data.index)
    best_greedy_candidate_matrix = np.array([])
    best_greedy_candidate_value = 0
    for i in range(len(data.index)):
        candidate = np.array(data.iloc[i].values.ravel())
        tmp_greedy_matrix = np.copy(sol_sq_greedy)
        tmp_greedy_matrix = append_row(tmp_greedy_matrix, candidate)
        tmp_greedy_value = 0
        if j == 0:
            tmp_greedy_value = LA.norm(tmp_greedy_matrix)
        else:
            tmp_greedy_value = volume(tmp_greedy_matrix)
        if tmp_greedy_value >= best_greedy_candidate_value:
           best_greedy_candidate_value = tmp_greedy_value
           best_greedy_candidate_matrix = tmp_greedy_matrix
    sol_sq_greedy = best_greedy_candidate_matrix
    value_sq_greedy = best_greedy_candidate_value

# Now we compute our online algorithm, the greedy algorirthm and the greedy algo with threshold
for i in range(len(data.index)):
    candidate = np.array(data.iloc[i].values.ravel())
    if i < k:
        sol_our = append_row(sol_our, candidate)
        sol_greedy = append_row(sol_greedy, candidate)
        sol_th = append_row(sol_th, candidate)
        if i == k-1:
            value_our = volume(sol_our)
            value_greedy = value_our
            value_th = value_our
    else:
        # We start to do some work, we try to remove the column
        # and improve with colum i
        comp_th = comp_th + k
        best_candidate_matrix_th, best_candidate_value_th, index = find_candidate_swap(sol_th, candidate)
        comp_greedy = comp_greedy + k
        best_candidate_matrix_greedy, best_candidate_value_greedy, index = find_candidate_swap(sol_greedy, candidate)
        comp_our = comp_our + k
        best_candidate_matrix_our, best_candidate_value_our, index = find_candidate_swap(sol_our, candidate)
        if best_candidate_value_greedy > value_greedy:
            value_greedy = best_candidate_value_greedy
            sol_greedy = best_candidate_matrix_greedy
            swap_greedy = swap_greedy + 1
        if best_candidate_value_th > (th * value_th):
            value_th = best_candidate_value_th
            sol_th = best_candidate_matrix_th
            swap_th = swap_th + 1
        if best_candidate_value_our > (th * value_our):
            value_our = best_candidate_value_our
            discarded = sol_our[index]
            if stash.size == 0:
                recent_swap = 0
            else:
                recent_swap = 1
            stash = append_row(stash, discarded)
            sol_our = best_candidate_matrix_our
            swap_our = swap_our + 1
            while(recent_swap):
                recent_swap = 0
                best_candidate_matrix = np.array([])
                best_candidate_value = 0
                stash_index = 0
                best_index = 0
                for l in range(np.shape(stash)[0]):
                    comp_our = comp_our + k
                    best_candidate_matrix_stash, best_candidate_value_stash, index = find_candidate_swap(sol_our, candidate)
                    if best_candidate_value_stash >= best_candidate_value:
                        best_candidate_matrix = best_candidate_matrix_stash
                        best_candidate_value = best_candidate_value_stash
                        stash_index = l
                        best_index = index
                if best_candidate_value > (th * value_our):
                    recent_swap = 1
                    value_our = best_candidate_value
                    discarded = np.copy(sol_our[best_index])
                    sol_our = best_candidate_matrix
                    stash = swap_row(stash, stash_index, discarded)
                    swap_our = swap_our + 1
    print(str(value_our) + " " + str(comp_our) + " " + str(swap_our) + " " + str(value_th) + " " + str(comp_th) + " " + str(swap_th) + " " + str(value_greedy) + " " + str(comp_greedy) + " " + str(swap_greedy) + " " + str(value_sq_greedy) + " " + str(comp_sq_greedy))
