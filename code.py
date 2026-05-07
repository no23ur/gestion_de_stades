#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def probleme_graphique():
    print("\n--- Résolution du Problème 1: Méthode Graphique ---")

    x = np.linspace(0, 150, 400)

    y1 = 200 - 5 * x
    y3 = 120 * np.ones_like(x)
    y4 = (300 - 5 * x) / 2

    plt.figure(figsize=(10, 8))
    plt.title("Problème 1: Optimisation par Méthode Graphique")
    plt.xlabel("X (Nombre de portiques)")
    plt.ylabel("Y (Nombre d'agents de sécurité)")
    plt.grid(True)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.axhline(0, color='k', linewidth=0.5)

    plt.plot(x, y1, label='5X + Y <= 200')
    plt.fill_between(x, 0, y1, where=y1 > 0, color='skyblue', alpha=0.1)

    plt.axvline(40, color='orange', linestyle='--', label='X <= 40')

    plt.plot(x, y3, color='green', linestyle=':', label='Y <= 120')
    plt.fill_between(x, 0, y3, where=y3 > 0, color='lightgreen', alpha=0.1)

    plt.plot(x, y4, color='purple', label='5X + 2Y <= 300')
    plt.fill_between(x, 0, y4, where=y4 > 0, color='lavender', alpha=0.1)

    y_feasible_upper_bound = np.minimum(y1, np.minimum(y3, y4))

    plt.fill_between(x, 0, y_feasible_upper_bound,
                     where=(x >= 0) & (x <= 40) & (y_feasible_upper_bound >= 0),
                     color='gray', alpha=0.5, label='Région réalisable')

    potential_vertices = [
        (0, 0),
        (0, 120),
        (12, 120),
        (16, 120),
        (20, 100),
        (40, 50),
        (40, 0)
    ]

    feasible_vertices = []
    for vx, vy in potential_vertices:
        if (5 * vx + vy <= 200 + 1e-9) and \
           (vx <= 40 + 1e-9) and \
           (vy <= 120 + 1e-9) and \
           (5 * vx + 2 * vy <= 300 + 1e-9) and \
           (vx >= -1e-9) and \
           (vy >= -1e-9):
            feasible_vertices.append((vx, vy))

    feasible_vertices = sorted(list(set(feasible_vertices)))

    print("Sommets de la région réalisable:")
    for v in feasible_vertices:
        z_val = 500 * v[0] + 200 * v[1]
        print(f"  ({v[0]}, {v[1]}) - Valeur de Z: {z_val}")
        plt.plot(v[0], v[1], 'ro')
        plt.text(v[0] + 1, v[1] + 5, f'({v[0]}, {v[1]})', fontsize=9)

    obj_values = [500 * v[0] + 200 * v[1] for v in feasible_vertices]

    if obj_values:
        max_z_index = np.argmax(obj_values)
        optimal_vertex = feasible_vertices[max_z_index]
        optimal_z = obj_values[max_z_index]
        print(f"\nSolution optimale: X = {optimal_vertex[0]}, Y = {optimal_vertex[1]}, Z = {optimal_z}")
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'go', markersize=10, label='Solution optimale')
        plt.text(optimal_vertex[0] + 1, optimal_vertex[1] - 10, f'Optimal ({optimal_vertex[0]}, {optimal_vertex[1]})', fontsize=10, color='green')

    plt.xlim(-5, 50)
    plt.ylim(-5, 150)
    plt.legend()
    plt.savefig('probleme_graphique.png')


def probleme_simplexe():
    print("\n--- Résolution du Problème 2: Méthode du Simplexe ---")

    c = np.array([-5000, -1000, -2000])

    A = np.array([
        [5000, 1000, 2000],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [400, 200, 300]
    ])

    b = np.array([800000, 40, 120, 60, 60000])

    num_vars = len(c)
    num_constraints = len(b)
    num_slack_vars = num_constraints

    tableau = np.zeros((num_constraints + 1, num_vars + num_slack_vars + 1))

    tableau[:num_constraints, :num_vars] = A
    tableau[:num_constraints, -1] = b

    for i in range(num_constraints):
        tableau[i, num_vars + i] = 1

    tableau[-1, :num_vars] = c
    tableau[-1, -1] = 0

    print("Tableau Simplex initial:")
    print(tableau)
    print("\n")

    iteration = 0
    while np.any(tableau[-1, :-1] < -1e-9):
        iteration += 1
        print(f"--- Itération {iteration} ---")

        pivot_col_idx = np.argmin(tableau[-1, :-1])
        print(f"  Colonne pivot (variable entrante): {pivot_col_idx}")

        ratios = []
        for i in range(num_constraints):
            if tableau[i, pivot_col_idx] > 1e-9:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col_idx])
            else:
                ratios.append(np.inf)

        if all(r == np.inf for r in ratios):
            print("  Problème non borné.")
            break

        pivot_row_idx = np.argmin(ratios)
        print(f"  Ligne pivot (variable sortante): {pivot_row_idx}")

        pivot_element = tableau[pivot_row_idx, pivot_col_idx]
        print(f"  Élément pivot: {pivot_element}")

        tableau[pivot_row_idx, :] = tableau[pivot_row_idx, :] / pivot_element

        for i in range(num_constraints + 1):
            if i != pivot_row_idx:
                factor = tableau[i, pivot_col_idx]
                tableau[i, :] = tableau[i, :] - factor * tableau[pivot_row_idx, :]

        print("Tableau après itération:")
        print(tableau)
        print("\n")

    print("--- Solution du Simplexe Manuelle ---")
    solution_vars = np.zeros(num_vars)
    for j in range(num_vars):
        one_indices = np.where(np.isclose(tableau[:num_constraints, j], 1))[0]

        if len(one_indices) == 1:
            row_idx = one_indices[0]
            other_elements_in_col = np.delete(tableau[:num_constraints, j], row_idx)
            if np.all(np.isclose(other_elements_in_col, 0)):
                solution_vars[j] = tableau[row_idx, -1]

    optimal_f_manual = -tableau[-1, -1]
    print(f"  X (Portiques) = {solution_vars[0]:.2f}")
    print(f"  Y (Agents de sécurité) = {solution_vars[1]:.2f}")
    print(f"  Z (Caméras intelligentes) = {solution_vars[2]:.2f}")
    print(f"  Valeur maximale de f = {-optimal_f_manual:.2f}")

    print("\n--- Validation avec scipy.optimize.linprog ---")

    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))

    if res.success:
        print(f"  Statut: {res.message}")
        print(f"  X (Portiques) = {res.x[0]:.2f}")
        print(f"  Y (Agents de sécurité) = {res.x[1]:.2f}")
        print(f"  Z (Caméras intelligentes) = {res.x[2]:.2f}")
        print(f"  Valeur maximale de f = {-res.fun:.2f}")
    else:
        print(f"  Erreur: {res.message}")


if __name__ == "__main__":
    probleme_graphique()
    probleme_simplexe()
