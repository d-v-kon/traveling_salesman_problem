import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

beta_start = 1
q = 1
N = 50
L = int(N**2)
sigma = 0.5
accepted_changes = 0


def initialize_system(N=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    positions = np.random.rand(N, 2)
    np.random.seed()

    return positions


def calculate_energy(positions):
    N = len(positions)
    diffs = []
    for i in range(N):
        iPlus1 = (i + 1) % N
        diffs.append(positions[iPlus1] - positions[i])
    segment_lengths = np.linalg.norm(diffs, axis=1)
    energy = np.sum(segment_lengths)

    return energy


def exchange_positions(positions, beta):
    N = len(positions)
    i = np.random.randint(0, N - 1)
    j = np.random.randint(i + 1, N)

    if i == 0 and j == N - 1:
        return None

    i_prev = (i - 1) % N
    j_next = (j + 1) % N

    E_diff = np.linalg.norm(positions[i_prev] - positions[j]) + np.linalg.norm(positions[i] - positions[j_next]) \
             - np.linalg.norm(positions[i_prev] - positions[i]) - np.linalg.norm(positions[j] - positions[j_next])

    acceptance_ratio = np.exp(-beta * E_diff)
    p_acc = min(1.0, acceptance_ratio)
    # p_acc = 1 / (1 + np.exp(beta * E_diff))

    if np.random.rand() < p_acc:
        positions[i:j + 1] = positions[i:j + 1][::-1]
        global accepted_changes
        accepted_changes += 1


def optimize_path_specific_heat(positions, beta_start):
    N = len(positions)
    best_E = 999
    E_mean_list = []
    E_squared_mean_list = []
    scaled_variance_list = []
    stop = False
    beta = beta_start
    global_while_counter = 0  #change letter for real criterion
    last_improvement_counter = 0

    while not stop:
        Ek_list = []

        for _ in tqdm(range(L)):
            Ek = calculate_energy(positions)
            Ek_list.append(Ek)
            if Ek < best_E:
                last_improvement_counter = -1
                best_E = Ek
                print(best_E)
                best_positions = positions.copy()
            exchange_positions(positions, beta)

        E_mean_list.append(np.mean(Ek_list))
        E_squared_mean_list.append(np.mean(np.square(Ek_list)))
        scaled_variance_list.append(beta ** 2 * (E_squared_mean_list[-1] - E_mean_list[-1] ** 2))

        global_while_counter += 1

        beta = beta_start*global_while_counter**q

        # r = 1 - sigma / (np.sqrt(scaled_variance_list[-1])) if scaled_variance_list[-1] > 1e-8 else 0.99
        # beta = beta / r

        last_improvement_counter += 1
        if global_while_counter == 120:
            stop = True

    k_index = np.arange(global_while_counter)
    print(best_positions, best_E)
    best_positions = np.vstack([best_positions, best_positions[0]])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_index, E_mean_list)
    plt.xlabel('k_index')
    plt.ylabel('Average Energy')
    plt.title('Average Energy vs k_index')

    plt.subplot(1, 2, 2)
    plt.plot(k_index, scaled_variance_list)
    plt.xlabel('k_index')
    plt.ylabel('scaled_variance')
    plt.title('scaled_variance vs k_index')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(best_positions[:, 0], best_positions[:, 1], marker='o', linestyle='-')
    plt.title("Path through particles (closed loop)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis('equal')  # so x and y have the same scale
    plt.show()


positions = initialize_system(N, 12345)

plt.figure(figsize=(6, 6))
plt.plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-')
plt.title("Path through particles (closed loop)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')  # so x and y have the same scale
plt.show()

optimize_path_specific_heat(positions, beta_start)
print(accepted_changes)
