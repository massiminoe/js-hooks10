from typing import List, Tuple

import pyomo.environ as pyo
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

N = 5


NO_ROTATION_5X5 = [
    (1, 1, 10),
    (3, 2, 15),
    (5, 2, 12),
    (1, 3, 8),
    (3, 4, 10),
    (5, 4, 7),
    (1, 5, 0),
]
ROTATION_5X5 = [
    (1, 1, 10),
    (3, 2, 15),
    (5, 2, 12),
    (1, 3, 8),
    (3, 4, 9),
    (5, 4, 7),
    (1, 5, 0),
]
PATHOLOGICAL_CLUES = [
    (1, 2, 8),
    (5, 2, 8),
    (1, 3, 4),
    (4, 3, 9),
    (5, 3, 0),
    (1, 4, 4),
    (5, 4, 2),
    (1, 5, 4),
    (3, 5, 7),
    (5, 5, 1),
]
FULL_CLUES = [
    (3, 1, 22),
    (8, 1, 15),
    (5, 2, 14),
    (2, 3, 9),
    (7, 3, 19),
    (2, 5, 5),
    (4, 5, 11),
    (6, 5, 22),
    (8, 5, 22),
    (3, 7, 9),
    (8, 7, 31),
    (5, 8, 12),
    (2, 9, 18),
    (7, 9, 7),
]


def one_number_per_hook(model, n):
    return sum(model.hook_number[n, i] for i in model.N) == 1


def all_hook_numbers_used(model, n):
    return sum(model.hook_number[i, n] for i in model.N) == 1


def add_hook_cell_constraints(model):
    model.hook_cell_constraints = pyo.ConstraintList()

    # for n in model.N:  # n == N is outermost
    #     for i in model.N:
    #         for j in model.N:
    #             for n2 in model.N:
    #                 model.hook_cell_constraints.add(
    #                     model.cell_number[i, j, n2]
    #                     <= (1 - model.cell_hook_number[i, j, n]) + model.hook_number[n, n2]
    #                 )

    # cell_number[i, j, n2] <= cell_hook_number[i, j, n] && hook_number[n, n2]

    for n in model.N:  # n == N is outermost
        for i in range(1, model.grid_size + 2 - n):
            for j in range(1, model.grid_size + 2 - n):
                origin = model.hook_origin[i, j, n]
                is_right = model.hook_right[n]
                is_top = model.hook_top[n]

                for n2 in model.N:
                    for ii in range(i, i + n):
                        model.hook_cell_constraints.add(
                            model.cell_number[ii, j, n2]
                            <= (1 - origin) + model.hook_number[n, n2] + is_top
                        )  # T = 0
                        model.hook_cell_constraints.add(
                            model.cell_number[ii, j + n - 1, n2]
                            <= (1 - origin) + model.hook_number[n, n2] + (1 - is_top)
                        )  # T = 1
                    for jj in range(j, j + n):
                        model.hook_cell_constraints.add(
                            model.cell_number[i, jj, n2]
                            <= (1 - origin) + model.hook_number[n, n2] + is_right
                        )  # R = 0
                        model.hook_cell_constraints.add(
                            model.cell_number[i + n - 1, jj, n2]
                            <= (1 - origin) + model.hook_number[n, n2] + (1 - is_right)
                        )  # R = 1

    for i in model.N:
        for j in model.N:
            model.hook_cell_constraints.add(sum(model.cell_number[i, j, n] for n in model.N) <= 1)


def cell_number_count_constraint(model, n):
    return sum(model.cell_number[i, j, n] for i in model.N for j in model.N) == n


def add_unfilled_constraint(model):
    """
    Satisfy "every 2-by-2 region must contain at least one unfilled square"
    """
    model.unfilled_constraints = pyo.ConstraintList()
    for i in range(1, model.grid_size):
        for j in range(1, model.grid_size):
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
        right_cell = (
            sum(model.cell_number[i + 1, j, n] * n for n in model.N) if i < model.grid_size else 0
        )
        below_cell = sum(model.cell_number[i, j - 1, n] * n for n in model.N) if j > 1 else 0
        above_cell = (
            sum(model.cell_number[i, j + 1, n] * n for n in model.N) if j < model.grid_size else 0
        )
        model.clue_constraints.add(left_cell + right_cell + below_cell + above_cell == total)


def add_hook_origin_constraints(model):
    model.hook_origin_constraints = pyo.ConstraintList()
    # Each of 1 type of hook
    for n in model.N:
        model.hook_origin_constraints.add(
            sum(model.hook_origin[i, j, n] for i in model.N for j in model.N) == 1
        )

    # Make sure hook origins aren't in spots that would see hooks extend outside of grid
    for n in model.N:
        for i in model.N:
            for j in model.N:
                if (i <= (model.grid_size - n + 1)) and (j <= (model.grid_size - n + 1)):
                    continue
                model.hook_origin_constraints.add(model.hook_origin[i, j, n] == 0)

    # # Make sure hook origins aren't inside other hooks
    # for n in model.N:  # n == N is outermost
    #     for i in range(1, model.grid_size + 2 - n):
    #         for j in range(1, model.grid_size + 2 - n):
    #             origin = model.hook_origin[i, j, n]
    #             is_right = model.hook_right[n]
    #             is_top = model.hook_top[n]

    #             for ii in range(i, i + n):
    #                 model.hook_origin_constraints.add(
    #                     sum(model.hook_origin[ii, j, n2] for n2 in model.N if n2 != n)
    #                     <= ((1 - origin) + is_top) * model.grid_size
    #                 )  # T = 0
    #                 model.hook_origin_constraints.add(
    #                     sum(model.hook_origin[ii, j + n - 1, n2] for n2 in model.N if n2 != n)
    #                     <= ((1 - origin) + (1 - is_top)) * model.grid_size
    #                 )  # T = 1
    #             for jj in range(j, j + n):
    #                 model.hook_origin_constraints.add(
    #                     sum(model.hook_origin[i, jj, n2] for n2 in model.N if n2 != n)
    #                     <= ((1 - origin) + is_right) * model.grid_size
    #                 )  # R = 0
    #                 model.hook_origin_constraints.add(
    #                     sum(model.hook_origin[i + n - 1, jj, n2] for n2 in model.N if n2 != n)
    #                     <= ((1 - origin) + (1 - is_right)) * model.grid_size
    #                 )  # R = 1


def add_cell_hook_number_constraints(model):
    model.cell_hook_number_constraints = pyo.ConstraintList()

    # Each cell must be assigned exactly one hook
    for i in model.N:
        for j in model.N:
            model.cell_hook_number_constraints.add(
                sum(model.cell_hook_number[i, j, n] for n in model.N) == 1
            )

    for n in model.N:  # n == N is outermost
        for i in range(1, model.grid_size + 2 - n):
            for j in range(1, model.grid_size + 2 - n):
                origin = model.hook_origin[i, j, n]
                is_right = model.hook_right[n]
                is_top = model.hook_top[n]

                for n2 in model.N:
                    for ii in range(i, i + n):
                        model.cell_hook_number_constraints.add(
                            model.cell_hook_number[ii, j, n2]
                            <= (1 - origin) + model.hook_number[n, n2] + is_top
                        )  # T = 0
                        model.cell_hook_number_constraints.add(
                            model.cell_hook_number[ii, j + n - 1, n2]
                            <= (1 - origin) + model.hook_number[n, n2] + (1 - is_top)
                        )  # T = 1
                    for jj in range(j, j + n):
                        model.cell_hook_number_constraints.add(
                            model.cell_hook_number[i, jj, n2]
                            <= (1 - origin) + model.hook_number[n, n2] + is_right
                        )  # R = 0
                        model.cell_hook_number_constraints.add(
                            model.cell_hook_number[i + n - 1, jj, n2]
                            <= (1 - origin) + model.hook_number[n, n2] + (1 - is_right)
                        )  # R = 1


def main(grid_size=N, clues=NO_ROTATION_5X5):
    model = pyo.ConcreteModel()
    model.grid_size = grid_size
    model.N = pyo.RangeSet(grid_size)

    model.hook_right = pyo.Var(model.N, domain=pyo.Binary)
    model.hook_top = pyo.Var(model.N, domain=pyo.Binary)
    model.hook_origin = pyo.Var(model.N, model.N, model.N, domain=pyo.Binary)
    model.hook_number = pyo.Var(model.N, model.N, domain=pyo.Binary)
    model.cell_hook_number = pyo.Var(model.N, model.N, model.N, domain=pyo.Binary)
    model.cell_number = pyo.Var(model.N, model.N, model.N, domain=pyo.Binary)

    model.one_number_per_hook = pyo.Constraint(model.N, rule=one_number_per_hook)
    model.all_hook_numbers_used = pyo.Constraint(model.N, rule=all_hook_numbers_used)
    model.cell_number_count_constraint = pyo.Constraint(model.N, rule=cell_number_count_constraint)
    add_hook_cell_constraints(model)
    add_unfilled_constraint(model)
    add_clues(model, clues)
    add_hook_origin_constraints(model)
    add_cell_hook_number_constraints(model)

    # Fix
    # model.hook_right[1].fix(1)
    # model.hook_top[1].fix(1)

    model.dummy_objective = pyo.Objective(rule=1)

    # model.pprint()
    model.write('9x9.lp')
    exit()

    solver = pyo.SolverFactory("cbc", tee=True)


    solution = solver.solve(model)

    cell_numbers = np.zeros((grid_size, grid_size, grid_size))
    hook_origins = np.zeros((grid_size, grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            for n in range(grid_size):
                cell_numbers[i][j][n] = model.cell_number[i + 1, j + 1, n + 1].value
                hook_origins[i][j][n] = model.hook_origin[i + 1, j + 1, n + 1].value

    cell_values = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            if not any(cell_numbers[i, j, :]):
                continue
            cell_values[i, j] = np.argmax(cell_numbers[i, j, :]) + 1

    # print(cell_numbers[:, :, 0])
    # print(cell_numbers[:, :, 1])
    # print(cell_numbers[:, :, 2])
    # print(cell_numbers[:, :, 3])
    # print(cell_numbers[:, :, 4])

    print(hook_origins[:, :, 0])
    print(hook_origins[:, :, 1])
    print(hook_origins[:, :, 2])
    print(hook_origins[:, :, 3])
    print(hook_origins[:, :, 4])

    a = hook_origins[:, :, 0]
    print(np.unravel_index(np.argmax(a, axis=None), a.shape))

    print()

    print()
    print(solution)
    # model.pprint()
    # model.hook_cell_constraints.pprint()
    model.hook_origin_constraints.pprint()

    model.hook_right.pprint()
    model.hook_top.pprint()
    model.hook_number.pprint()
    print(hook_origins[0, 0])

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

    for i in range(grid_size):
        for j in range(grid_size):
            value = int(cell_values[i, j])
            # ax.add_patch(Rectangle((i+1, j+1), 1, 1, facecolor=colour_dict[value]))
            ax.add_patch(Rectangle((i, j), 1, 1, color=colour_dict[value], fill=True, alpha=0.5))
            if value:
                plt.text(i + 0.5, j + 0.5, value, va="center", ha="center", fontsize=20)

    for i, j, total in clues:
        plt.text(i - 0.5, j - 0.5, total, va="center", ha="center", fontsize=20, alpha=0.5)

    for n in range(grid_size):
        a = hook_origins[:, :, n]
        i, j = np.unravel_index(np.argmax(a, axis=None), a.shape)
        plt.text(
            i + 0.25,
            j + np.random.uniform(0.5, 1),
            n + 1,
            va="center",
            ha="center",
            fontsize=10,
            alpha=0.5,
            color="red",
        )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.grid(True, lw=3, c="k")
    plt.show()


if __name__ == "__main__":
    # main()
    # main(grid_size=5, clues=ROTATION_5X5)
    main(grid_size=9, clues=FULL_CLUES)
    # main(clues=PATHOLOGICAL_CLUES)
