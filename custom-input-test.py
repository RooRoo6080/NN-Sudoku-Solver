import numpy as np
import pandas as pd

from keras.utils import to_categorical
from keras.models import load_model

empty = ['000001006007060001000090400010009300500006070003000908041020000080000000090047200']
answer = ['954731826837264591162598437618479352529386174473152968741925683285613749396847215']

def print_sudoku(string):
    board = []
    n = 0
    for _ in range(9):
        row = []
        for _ in range(9):
            row.append(int(string[n]))
            n += 1
        board.append(row)
    print("+" + "---+"*9)
    for i, row in enumerate(board):
        print(("|" + " {}   {}   {} |"*3).format(*[x if x != 0 else " " for x in row]))
        if i % 3 == 2:
            print("+" + "---+"*9)
        else:
            print("+" + "   +"*9)

def print_sudoku_arr(board):
    print("+" + "---+"*9)
    for i, row in enumerate(board):
        print(("|" + " {}   {}   {} |"*3).format(*[x if x != 0 else " " for x in row]))
        if i % 3 == 2:
            print("+" + "---+"*9)
        else:
            print("+" + "   +"*9)

def load_data(train = 0):

    quizzes = empty
    solutions = answer

    flatX = np.array(
        [np.reshape([int(d) for d in flatten_grid], (9, 9)) for flatten_grid in quizzes]
    )
    flaty = np.array(
        [
            np.reshape([int(d) for d in flatten_grid], (9, 9))
            for flatten_grid in solutions
        ]
    )
    return (flatX[:train], flaty[:train]), (flatX[train:], flaty[train:])

def batch_smart_solve(grids, solver):
    grids = grids.copy()
    for _ in range((grids == 0).sum((1, 2)).max()):
        preds = np.array(solver.predict(to_categorical(grids)))
        probs = preds.max(2).T
        values = preds.argmax(2).T + 1
        zeros = (grids == 0).reshape((grids.shape[0], 81))

        for grid, prob, value, zero in zip(grids, probs, values, zeros):
            if any(zero):
                where = np.where(zero)[0]
                confidence_position = where[
                    prob[zero].argmax()
                ]
                confidence_value = value[confidence_position]
                grid.flat[confidence_position] = confidence_value
    return grids

def diff(grids_true, grids_pred):
    return (grids_true != grids_pred).sum((1, 2))

input_shape = (9, 9, 10)
(_, ytrain), (xtest, ytest) = (
    load_data()
)

xtest = to_categorical(xtest).astype("float32")
ytest = to_categorical(ytest - 1).astype("float32")

solver = load_model('model.keras')

quizzes = xtest.argmax(3)
true_grids = ytest.argmax(3) + 1
smart_guesses = batch_smart_solve(quizzes, solver)

deltas = diff(true_grids, smart_guesses)
accuracy = (deltas == 0).mean()

f_smart_guesses = []
for i in range(9):
    f_smart_guesses.append(smart_guesses[0][i].tolist())

print('Input:')
print_sudoku(empty[0])
print('Attempt:')
print_sudoku_arr(f_smart_guesses)

print(
"""
Attempted:\t {}
Correct:\t {}
Accuracy:\t {}
""".format(
deltas.shape[0], (deltas==0).sum(), accuracy
)
)