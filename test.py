import numpy as np
import pandas as pd

from keras import Model, Sequential
from keras.utils import to_categorical
from keras.models import load_model

def load_data(train=0, test=1000, full=False):

    if full:
        sudokus = pd.read_csv("sudoku.csv").values
    else:
        sudokus = next(pd.read_csv("sudoku.csv", chunksize=(train + test))).values

    quizzes, solutions = sudokus.T
    flatx = np.array(
        [np.reshape([int(d) for d in flatten_grid], (9, 9)) for flatten_grid in quizzes]
    )
    flaty = np.array(
        [
            np.reshape([int(d) for d in flatten_grid], (9, 9))
            for flatten_grid in solutions
        ]
    )
    return (flatx[:train], flaty[:train]), (flatx[train:], flaty[train:])

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

print(
"""
Attempted:\t {}
Correct:\t {}
Accuracy:\t {}
""".format(
deltas.shape[0], (deltas==0).sum(), accuracy
)
)