from typing import List, Tuple

import pyomo.environ as pyo
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

N = 5


CLUES = [
    (1, 1, 10),
    (3, 2, 15),
    (5, 2, 12),
    (1, 3, 8),
    (3, 4, 10),
    (5, 4, 7),
    (1, 5, 0),
]

def one_number_per_hook(model, n):
    return sum(model.hook_number[n, i] for i in model.N) == 1


def all_hook_numbers_used(model, n):
    return sum(model.hook_number[i, n] for i in model.N) == 1


def add_hook_cell_constraints(model):
    model.hook_cell_constraints = pyo.ConstraintList()
    for n in model.N:  # n == N is outermost
        origin = N - n + 1

        for n2 in model.N:
            # origin constraint
            model.hook_cell_constraints.add(
                model.cell_number[origin, origin, n2] <= model.hook_number[n, n2]
            )

            for i in range(origin, N):
                model.hook_cell_constraints.add(  # Row
                    model.cell_number[i + 1, origin, n2] <= model.hook_number[n, n2]
                )
                model.hook_cell_constraints.add(  # Column
                    model.cell_number[origin, i + 1, n2] <= model.hook_number[n, n2]
                )


def cell_number_count_constraint(model, n):
    return sum(model.cell_number[i, j, n] for i in model.N for j in model.N) == n


def add_unfilled_constraint(model):
    """
    Satisfy "every 2-by-2 region must contain at least one unfilled square"
    """
    model.unfilled_constraints = pyo.ConstraintList()
    for i in range(1, N):
        for j in range(1, N):
            model.unfilled_constraints.add(
                sum(
                    model.cell_number[i, j, n]
                    + model.cell_number[i + 1, j, n]
                    + model.cell_number[i, j + 1, n]
                    + model.cell_number[i + 1, j + 1, n]
                    for n in model.N
                )
                <= 3
            )

def add_clues(model, clues: List[Tuple[int, int, int]]):
    model.clue_constraints = pyo.ConstraintList()
    for i, j, total in clues:
        # Clues are in cells that are not filled
        model.clue_constraints.add(sum(model.cell_number[i, j, n] for n in model.N) == 0)
        # if i > 1:
        left_cell = sum(model.cell_number[i - 1, j, n] * n for n in model.N) if i > 1 else 0
        right_cell = sum(model.cell_number[i + 1, j, n] * n for n in model.N) if i < N else 0
        below_cell = sum(model.cell_number[i, j - 1, n] * n for n in model.N) if j > 1 else 0
        above_cell = sum(model.cell_number[i, j + 1, n] * n for n in model.N) if j < N else 0
        model.clue_constraints.add(left_cell + right_cell + below_cell + above_cell == total)


def main(clues=CLUES):
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(N)

    model.hook_number = pyo.Var(model.N, model.N, domain=pyo.Binary)
    model.cell_number = pyo.Var(model.N, model.N, model.N, domain=pyo.Binary)

    model.one_number_per_hook = pyo.Constraint(model.N, rule=one_number_per_hook)
    model.all_hook_numbers_used = pyo.Constraint(model.N, rule=all_hook_numbers_used)
    model.cell_number_count_constraint = pyo.Constraint(model.N, rule=cell_number_count_constraint)
    add_hook_cell_constraints(model)
    add_unfilled_constraint(model)
    add_clues(model, clues)

    model.dummy_objective = pyo.Objective(rule=1)

    solver = pyo.SolverFactory("cbc")
    solution = solver.solve(model)

    print(solution)

    cell_numbers = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for n in range(N):
                cell_numbers[i][j][n] = model.cell_number[i + 1, j + 1, n + 1].value

    cell_values = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if not any(cell_numbers[i, j, :]):
                continue
            cell_values[i, j] = np.argmax(cell_numbers[i, j, :]) + 1

    colour_dict = {
        0: "white",
        1: "orange",
        2: "purple",
        3: "blue",
        4: "green",
        5: "grey",
        6: "pink",
        7: "pink",
        8: "pink",
        9: "pink",
    }
    fig, ax = plt.subplots()

    for i in range(N):
        for j in range(N):
            value = int(cell_values[i, j])
            # ax.add_patch(Rectangle((i+1, j+1), 1, 1, facecolor=colour_dict[value]))
            ax.add_patch(Rectangle((i, j), 1, 1, color=colour_dict[value], fill=True, alpha=0.5))
            if value:
                plt.text(i + 0.5, j + 0.5, value, va="center", ha="center", fontsize=20)

    for i, j, total in clues:
        plt.text(i - 0.5, j - 0.5, total, va="center", ha="center", fontsize=20, alpha=0.5)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.grid(True, lw=3, c="k")
    plt.show()


if __name__ == "__main__":
    main()
