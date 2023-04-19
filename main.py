#!/usr/bin/env python
# coding: utf-8

import os
import random
import subprocess
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from IPython.display import clear_output
import copy
import time
from tqdm import trange

# Seed everything
np.random.seed(2023)
random.seed(2023)

# #### The next cell contains the hyperparameters of the genetic algorithm.
N = 9  # The board size (n by n)!

MUTATION = .04  # The fraction of the population that suffer a mutation
TARGET_MUTATION = 0.0125  # The fraction of the mutant with "clever" mutation.
N_GENERATION = 50_000  # Number of generations
P = 10_000  # population
N_ELITES = 128  # parents unchanged for next generation


# Individual structure: a numpy array of side (9,9). int type
# positions = [(i, j) for i in range(N) for j in range(N)] # This list contains all the board positions.

def load_from_file(fname: str = 'Soduku4.csv') -> np.ndarray:
    ind = np.array([np.nan for _ in range(9 * 9)]).reshape((N, N))
    with open(fname, 'r') as fp:
        lines = [line.strip() for line in fp.readlines()]
    for i in range(N):
        for j, e in enumerate(lines[i].split(',')):
            if 'nan' in e.lower():
                continue
            ind[i, j] = int(e.strip())

    return ind


base = load_from_file()


def print_sudoku(sudoku_board):
    """
    Print a Sudoku board to the command line interface.

    Args:
    - sudoku_board: a numpy array of shape (9, 9) representing a Sudoku board
    """
    for i in range(9):
        # Print a horizontal line every three rows
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - -")

        for j in range(9):
            # Print a vertical line every three columns
            if j % 3 == 0 and j != 0:
                print("|", end=" ")

            # Print the value at this cell
            value = int(sudoku_board[i][j])
            if value == 0:
                print(".", end=" ")
            else:
                print(value, end=" ")

        print()


def random_parents(n_parents: int, base: np.ndarray) -> list:
    """
    It generate a list of n_parents elements, each element is a individual following definition of cell #3.
    """
    parents = []
    remaining_values = list(range(1, N + 1)) * 9
    for val in base[~np.isnan(base)].reshape((-1,)).tolist():
        remaining_values.remove(val)

    for _ in range(n_parents):
        ind = np.random.randint(1, N, size=(N, N), dtype=int)
        ind[~np.isnan(base)] = base[~np.isnan(base)]
        remaining_values_bis = copy.deepcopy(remaining_values)
        random.shuffle(remaining_values_bis)
        ind[np.isnan(base)] = remaining_values_bis
        parents.append(ind)

    return parents


def loss(ind: np.ndarray) -> int:
    """
    This is the fitness function of the GA algorithm. It count the number of errors in all
    the board.
    """
    errors = 0
    if isinstance(ind, list):
        ind = np.array(ind).astype(int)
    # Row direction check
    for row in range(N):
        errors += N - len(set(ind[row, :].tolist()))

    # Row direction check
    for col in range(N):
        errors += N - len(set(ind[:, col].tolist()))

    # Small boxes check
    for i in range(0, N, 3):
        for j in range(0, N, 3):
            errors += N - len(set(ind[i:i + 3, j:j + 3].reshape((-1,)).tolist()))
    return errors


# #### A helper function to score all the population
def select_best(parents: list):
    """
    It does the fitness of all the population and sort it by it score in a ascending way.
    Returns the sorted population, the score of the best, and best individual.
    """
    scores = []
    for parent in parents:  # loop all individual
        scores += [loss(parent)]  # score the individual and append to scores list.So, parents, and score are
        # parallel list.

    sorted_idx = np.argsort(
        scores)  # Next 3 line of code sort both parents, and scores. It uses numpy to get a better performance.
    new_parents = np.array(parents)[sorted_idx].tolist()
    best_values = np.array(scores)[sorted_idx].tolist()

    return new_parents, best_values, best_values[0], new_parents[0]


def parallel_loss(parents):
    scores = []
    for parent in parents:
        scores += [loss(parent)]
    return scores


def select_best____(parents: list):
    """
    It does the fitness of all the population and sort it by it score in a ascending way.
    Returns the sorted population, the score of the best, and best individual.

    The difference with the baseline is the use of all the processor with joblib parallel library.
    """

    step = 2500

    # Break the population (parents) in subset of step size, then score them using all the processor.

    #     parents = [list(e) for e in set([tuple(e) for e in parents])] # Simplify parents.
    parents = list(set((tuple([tuple(e) for e in ind])) for ind in parents))
    n_jobs = multiprocessing.cpu_count()
    scores_raw = Parallel(n_jobs=10)(delayed(parallel_loss)(parents[i: i + step]) for i in range(0, len(parents), step))

    # Next lines collect the output of the last step in scores variable.
    scores = []
    for score_raw in scores_raw:
        scores += score_raw

    sorted_idx = np.argsort(
        scores)  # Next 3 line of code sort both parents, and scores. It uses numpy to get a better performance.
    new_parents = np.array(parents)[sorted_idx].tolist()
    best_values = np.array(scores)[sorted_idx].tolist()
    ct = 1
    while best_values[0] == best_values[ct]:
        ct += 1

    return new_parents, best_values[0], new_parents[0]


def random_combine(parents, n_offsprings: int, scores: list) -> list:
    """
    This function takes care of the reproduce stage of the GA algorithm.
    n_offsprings: is the number of indivuals in the next generation.
    parents is the list with the current individuals of the population.

    It returns a list of size n_offsprings with the new individuals.

    The childs are made breaking mom and dad in the same random index: child "a" is the concatenation of
    mom's head and dad's tail, while child "b" is the concatenation of
    dad's head and mom's tail.

    """
    freq_selection = [1. / (score + 1e-6) for score in scores]
    freq_selection_tot = sum(freq_selection)
    freq_selection_norm = [freq / freq_selection_tot for freq in freq_selection]
    dads = random.choices(parents, k=int(n_offsprings // 2), weights=freq_selection_norm)
    moms = random.choices(parents, k=int(n_offsprings // 2), weights=freq_selection_norm)

    childs = []
    break_points = [(3, 3), (3, 6), (3, 9),
                    (6, 0), (6, 3), (6, 6), (6, 9),
                    (9, 0), (9, 3), (9, 6)]
    for dad, mom in zip(dads, moms):
        if isinstance(dad, list):
            dad = np.array(dad).astype(int)
        if isinstance(mom, list):
            mom = np.array(mom).astype(int)
        i, j = random.choice(break_points)
        mask = np.ones((N, N))
        mask[:i, :j] = 0
        mask = mask.astype(bool)

        child0, child1 = np.zeros((N, N)), np.zeros((N, N))
        child0[mask] = dad[mask]
        child0[~mask] = mom[~mask]
        child1[mask] = mom[mask]
        child1[~mask] = dad[~mask]
        childs += [child0, child1]
    return childs


def mutate(parents, mutation=MUTATION, targeted_mutation_trh=TARGET_MUTATION, base=None):
    """
    mutate walks all the population, and acts on an individual with a mutation frequency (probability).
    The mutation updates 1 position of the board.

    Later I develop a target_mutation function that is smaller, and slower the mutate: This new function
        mutate the individual by taken 1 position and doing force brute on it.
    """
    proc_parents = []
    for i, parent in enumerate(parents):
        if isinstance(parent, list):
            parent = np.array(parent).astype(int)
        if random.random() < mutation:
            if random.random() < 0.5:
                col, row = random.randint(0, N - 1), random.randint(0, N - 1)
                if np.isnan(base[row, col]):
                    res = set(list(range(1, N + 1))) - set(parent[:, col].tolist())
                    if len(res):
                        n = random.choice(list(res))
                        parent[row, col] = n
                proc_parents += [parent]
            else:
                col, row = random.randint(0, N - 1), random.randint(0, N - 1)
                if np.isnan(base[row, col]):
                    res = set(list(range(1, N + 1))) - set(parent[row, :].tolist())
                    if len(res):
                        n = random.choice(list(res))
                        parent[row, col] = n
                proc_parents += [parent]

            if random.random() < 0.5:
                col, row = random.randint(0, N - 1), random.randint(0, N - 1)
                if np.isnan(base[row, col]):
                    curr_best = parent.copy()
                    best_score = loss(parent)
                    for val in range(1, N + 1):
                        parent[row, col] = val
                        curr_loss = loss(parent)
                        if curr_loss < best_score:
                            best_score = curr_loss
                            curr_best = parent.copy()
                    proc_parents[-1] = parent
        else:
            proc_parents.append(parent)
    return proc_parents


start_mark = time.time()  # A time mark to monitor the execution time.

base = load_from_file()
parents = random_parents(P, base)  # It generates the initial population.

parents, scores, best_score, best = select_best(parents)  # It does fitness of all populations, and sorted it.
elites = copy.deepcopy(parents[:N_ELITES])  # Initializate the elites list.
best_score_bis = best_score

i = 0
with trange(N_GENERATION) as tr:
    for i in tr:  # Loop until the target generation.
        if len(parents) < P:
            new_parents = random_parents(P - len(parents), base=base)
            new_parents, new_scores, _, _ = select_best(
                new_parents)  # It does fitness of all populations, and sorted it.
            parents += new_parents
            scores += new_scores
        len(parents), P
        parents = random_combine(parents, P, scores)  # It produce the next generetion of individuals.
        len(parents), P
        parents = mutate(parents, base=base)  # It generate random mutations in the population.
        len(parents), P

        parents = elites + parents  # The elites pass trough unchange to the next generation.
        assert loss(parents[0]) == best_score_bis  # It confirms the presence of the best individual.

        parents, scores, best_score, best = select_best(parents)  # It does fitness of all populations, and sorted it.
        # Discard worst 10%
        len(parents), P

        Pd = int(0.1 * P)
        parents, scores = parents[:-Pd], scores[:-Pd]

        assert best_score <= best_score_bis
        # Some of the individuals in the current population that have lower fitness are chosen as elite.
        # These elite individ
        elites = copy.deepcopy(parents[:N_ELITES])  # It segregates a copy of the elite.
        best_score_bis = loss(elites[0])  # It mems the score of the best individual.

        # if not (i) % 5 or best_score == 0:
        #     # Just a log to make confortable the waiting.
        #     # print(f'n_iter: {i}, best_score: {best_score}, best: {", ".join([f"({y}, {x})" for i, (y, x) in enumerate(best)])}')
        #     print(f'n_iter: {i}, best_score: {best_score}, score mean: {np.mean(scores)}, score std: {np.std(scores)}') # , best: {", ".join([f"({y}, {x})" for i, (y, x) in enumerate(best)])}')
        # if not i % (50):
        #     print_sudoku(best)

        # Description will be displayed on the left
        tr.set_description(f'best_score: {best_score}')

        if best_score == 0:  # early stopping !!!
            print(f'Early stopping!  enlp time: {round(time.time() - start_mark, 3)} s')
            break

        if round(time.time() - start_mark, 3) > 3000:
            print(f'I was not able to find a solution in less than 3000 s.')
            break

        if np.std(scores) == 0:  # A way to reset!
            if random.random() < 0.25:
                parents = random_parents(P, base)  # It generates the initial population.
                parents, scores, best_score, best = select_best(
                    parents)  # It does fitness of all populations, and sorted it.
                elites = copy.deepcopy(parents[:N_ELITES])  # It segregates a copy of the elite.
                best_score_bis = loss(elites[0])  # It mems the score of the best individual.
                tr.set_description("    - FULL Reset HERE!")
            else:
                parents = elites
                parents, scores, best_score, best = select_best(
                    parents)  # It does fitness of all populations, and sorted it.
                tr.set_description("    - Reset HERE!")
#### Finally print the solution, if it is found.

if best_score == 0:
    print(f'Solution found for Soduku!')
    print_sudoku(best)

assert loss(best) == 0