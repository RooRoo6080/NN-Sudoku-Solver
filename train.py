import numpy as np
import pandas as pd

from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import to_categorical

def load_data(train=40000, test=10000, full=False):

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


def diff(grids_true, grids_pred):
    return (grids_true != grids_pred).sum((1, 2))


def delete_digits(x, delete=1):
    grids = x.argmax(3)
    for grid in grids:
        grid.flat[np.random.randint(0, 81, delete)] = (0)

    return to_categorical(grids)


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


input_shape = (9, 9, 10)
(_, ytrain), (xtest, ytest) = (
    load_data()
)

Xtrain = to_categorical(ytrain).astype("float32")
xtest = to_categorical(xtest).astype("float32")

ytrain = to_categorical(ytrain - 1).astype("float32")
ytest = to_categorical(ytest - 1).astype("float32")

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=input_shape))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())

grid = Input(shape=input_shape)
features = model(grid)

digit_placeholders = [
    Dense(9, activation='softmax')(features)
    for _ in range(81)
]


solver = Model(grid, digit_placeholders)
solver.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'] * 81,
)

solver.fit(
    delete_digits(Xtrain, 0),
    [ytrain[:, i, j, :] for i in range(9) for j in range(9)],
    batch_size=128,
    epochs=1,
    verbose=2
)

early_stop = EarlyStopping(patience=2, verbose=1)

i = 1
for epochs, nb_delete in zip(
        [1, 2, 3, 4, 6, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55]
):
    print('Pass {}'.format(i))
    i += 1
    
    solver.fit(
        delete_digits(Xtrain, nb_delete),
        [ytrain[:, i, j, :] for i in range(9) for j in range(9)],
        validation_data=(
            delete_digits(Xtrain, nb_delete),
            [ytrain[:, i, j, :] for i in range(9) for j in range(9)]),
        batch_size=128,
        epochs=epochs,
        verbose=2,
        callbacks=[early_stop]
    )

solver.save('model.keras')