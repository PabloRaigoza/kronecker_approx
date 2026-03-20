import numpy as np
import matplotlib.pyplot as plt

m1, n1, m2, n2 = 1, 8, 5, 5
DO_PRINT = True

def print_partition(state, num_ranks): print(f"Ranks: {num_ranks} {state}")
def sort_partition(state):
    new_state = [[] for _ in range(n1)]
    for i in range(n1):
        new_state[i] = sorted(state[i])
    return new_state

def split_each_rank(m1, n1, m2, n2, state, num_ranks):
    # state is a list of lists, where len(state) == n1
    # state[i] is a list of rank id's that share the partition for column i
    new_state = [[] for _ in range(n1)]
    for i in range(n1):
        states_to_add = max(n2 - len(state[i]), 0)
        for j in range(len(state[i])):
            new_state[i].append(state[i][j])
            if states_to_add > 0:
                new_state[i].append(state[i][j] + num_ranks)
                states_to_add -= 1

    new_state = sort_partition(new_state)
    if DO_PRINT: print_partition(new_state, num_ranks * 2)
    return new_state, num_ranks * 2

def split_columns(m1, n1, m2, n2, state, num_ranks):
    # if rank i is split across columns 1, 2, 3, 4, then split so i is split across columns 1, 2 and i + num_ranks is split across columns 3, 4
    # if num_ranks >= n1: return state, num_ranks
    is_last = num_ranks * 2 > n1
    
    new_state = [[] for _ in range(n1)]
    if not is_last:
        for rank in range(num_ranks):
            columns = [i for i in range(n1) if rank in state[i]]
            split_point = len(columns) // 2
            for i in range(split_point):
                new_state[columns[i]].append(rank)
            for i in range(split_point, len(columns)):
                new_state[columns[i]].append(rank + num_ranks)
    else:
        for rank in range(num_ranks):
            columns = [i for i in range(n1) if rank in state[i]]
            if len(columns) == 1:
                for i in range(len(columns)):
                    new_state[columns[i]].append(rank)
            else:
                split_point = len(columns) // 2
                for i in range(split_point):
                    new_state[columns[i]].append(rank)
                for i in range(split_point, len(columns)):
                    new_state[columns[i]].append(rank + num_ranks)
                    
    new_state = sort_partition(new_state)
    if DO_PRINT: print_partition(new_state, num_ranks * 2)
    return new_state, num_ranks * 2

def plot_partition(state, num_ranks, filename='partition.png'):
    A = np.full((m1 * m2, n1 * n2), -1, dtype=int)

    for i in range(n1):
        ranks = state[i]
        k = len(ranks)

        if k == 0: continue

        base = n2 // k
        remainder = n2 % k
        col_start = i * n2

        for idx, rank in enumerate(ranks):
            extra = 1 if idx < remainder else 0
            col_end = col_start + base + extra
            A[:, col_start:col_end] = rank
            col_start = col_end
    print(A[0, :n2])
    plt.imshow(A, cmap='turbo', vmin=-1, vmax=num_ranks-1)
    for i in range(1, n1): plt.axvline(x=i*n2-0.5, color='black', linestyle='--')
    for i in range(1, m1): plt.axhline(y=i*m2-0.5, color='black', linestyle='--')
    plt.yticks([])
    plt.xticks([i*n2 + n2/2 - 0.5 for i in range(n1)], [f"Col {i}" for i in range(n1)])
    plt.gca().xaxis.set_ticks_position('top')
    plt.tight_layout()

    # plt.savefig(filename, dpi=300)
# Intialize state
state, num_ranks = [[0] for _ in range(n1)], 1
print_partition(state, num_ranks)

# Split
state, num_ranks = split_columns(m1, n1, m2, n2, state, num_ranks)
plot_partition(state, num_ranks, filename='partition_1.png')
state, num_ranks = split_each_rank(m1, n1, m2, n2, state, num_ranks)
plot_partition(state, num_ranks, filename='partition_2.png')
state, num_ranks = split_columns(m1, n1, m2, n2, state, num_ranks)
plot_partition(state, num_ranks, filename='partition_3.png')
state, num_ranks = split_each_rank(m1, n1, m2, n2, state, num_ranks)
plot_partition(state, num_ranks, filename='partition_4.png')
state, num_ranks = split_columns(m1, n1, m2, n2, state, num_ranks)
plot_partition(state, num_ranks, filename='partition_5.png')
state, num_ranks = split_each_rank(m1, n1, m2, n2, state, num_ranks)
plot_partition(state, num_ranks, filename='partition_6.png')
state, num_ranks = split_columns(m1, n1, m2, n2, state, num_ranks)
plot_partition(state, num_ranks, filename='partition_7.png')
