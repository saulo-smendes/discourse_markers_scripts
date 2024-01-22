import argparse
import platform
from datetime import datetime
from pathlib import Path

import h5py
import keras_tuner as kt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import losses, models, optimizers, utils
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout


if platform.system() == "Darwin":
    DATA = Path("~/Documents/LISN/DATA/Saulo/data").expanduser()
else:
    DATA = Path("/people/evrard/DATA/Saulo/data")

MODEL = Path("models")
RES = Path("results")

DATA_FP = DATA / "merged_CSTR-KEELE_file_20230227.hdf5"

# Erros with b_size <= 128 with GPUs:
# TypeError: `generator` yielded an element of shape (0,)
#   where an element of shape (None, None, None, None) was expected.

ALL_PARAMS = {
    "w_size": (7, 38, 1),
    "b_size": 512,
    "l_rate": 1e-4,
    "dropouts": (0.2, 0.4),
    "epochs": 30,
    "smoothing": 0.1,
    "fts": "all",
    "seed": 42,
}

MFCC_PARAMS = {
    "w_size": (7, 20, 1),
    "b_size": 512,
    "l_rate": 1e-3,
    "dropouts": (0.6, 0.0),
    "epochs": 30,
    "smoothing": 0.2,
    "fts": "mfcc",
    "seed": 42,
}

F0_PARAMS = {
    "w_size": (7, 18, 1),
    "b_size": 512,
    "l_rate": 1e-3,
    "dropouts": (0.2, 0.4),
    "epochs": 30,
    "smoothing": 0.1,
    "fts": "f0",
    "seed": 42,
}

TOY_PARAMS = {
    "w_size": (7, 19, 1),
    "b_size": 1024,
    "l_rate": 1e-3,
    "dropouts": (0.2, 0.4),
    "epochs": 5,
    "smoothing": 0,
    "fts": "f0",
    "seed": 42,
}

PARAMS = F0_PARAMS  # ALL_PARAMS, MFCC_PARAMS, F0_PARAMS, TOY_PARAMS
FOLD_MAX = 1  # 2, 10


def prepare_data(groups, f, pad_width):  # (indices,
    arrays = []
    # for idx in indices:
    #     group_name = groups[idx]
    for idx, group_name in enumerate(groups):
        # Get the dataset in group
        data = np.array(f[group_name])
        # But we need to pad the edges of the 0-th axis with the same values
        #   of the first and last lines
        padded_data = np.pad(data, pad_width=pad_width)
        # Add the id of the group as a column of the data
        padded_data = np.hstack((padded_data, np.ones((padded_data.shape[0], 1)) * idx))
        # Append the padded data to the list
        arrays.append(padded_data)

    return np.vstack(arrays)


def split_features_labels(data, fts="mel"):
    ys = data[:, 1]
    if fts == "f0":
        xs = data[:, 2:20]
    elif fts == "mfcc":
        xs = data[:, 20:40]
    elif fts == "mel":
        xs = data[:, 2:34]
    else:  # all == "f0":
        xs = data[:, 2:40]
    return xs, ys


def stack_fold_data(f, pad_width, fts):
    """
    Stack and fold the data from the h5py file.
    f: h5py file
    """
    # groups = list(f.keys())
    # groups = [g for g in groups if "_egg" not in g and "_" in g]

    fname_df = pd.read_json(DATA / "fnames.json")

    cv_speakers = fname_df.speaker[fname_df.subset == "Train"].unique()
    # all_speakers = fname_df.speaker.unique()

    # TODO: use StratifiedKFold to spearate the data so that
    #       entire files separate for different training-evaluating folds

    kf = KFold(n_splits=7, shuffle=True, random_state=PARAMS["seed"])
    train_folds, valid_folds = [], []

    for idx, (train_speaker_ids, valid_speaker_ids) in enumerate(kf.split(cv_speakers)):
        train_mask = fname_df.speaker.isin(cv_speakers[train_speaker_ids])
        train_groups = fname_df.fname[train_mask].to_list()
        train_data = prepare_data(train_groups, f, pad_width)
        train_xs, train_ys = split_features_labels(train_data, fts=fts)
        train_folds.append((train_xs, train_ys))

        valid_mask = fname_df.speaker.isin(cv_speakers[valid_speaker_ids])
        valid_groups = fname_df.fname[valid_mask].to_list()
        valid_data = prepare_data(valid_groups, f, pad_width)
        valid_xs, valid_ys = split_features_labels(valid_data, fts=fts)
        valid_folds.append((valid_xs, valid_ys))

        if idx + 1 >= FOLD_MAX:
            break  # ***MEV*** Limit the number of folds

    # for idx, (train_ids, valid_ids) in enumerate(kf.split(groups)):
    #     # Train ids are the indices of the groups that will be used for training
    #     train_data = prepare_data(train_ids, groups, f, pad_width)
    #     train_folds.append(train_data)

    #     # Now we do the same but for the validation data
    #     valid_data = prepare_data(valid_ids, groups, f, pad_width)
    #     valid_folds.append(valid_data)

    return train_folds, valid_folds


def load_main_train_test_data(f, pad_width, fts):
    fname_df = pd.read_json(DATA / "fnames.json")
    train_speakers = fname_df.speaker[fname_df.subset == "Train"].unique()
    test_speakers = fname_df.speaker[fname_df.subset == "Test"].unique()

    train_mask = fname_df.speaker.isin(train_speakers)
    train_groups = fname_df.fname[train_mask].to_list()
    train_data = prepare_data(train_groups, f, pad_width)

    test_mask = fname_df.speaker.isin(test_speakers)
    test_groups = fname_df.fname[test_mask].to_list()
    test_data = prepare_data(test_groups, f, pad_width)

    train_xs, train_ys = split_features_labels(train_data, fts=fts)
    test_xs, test_ys = split_features_labels(test_data, fts=fts)

    return train_xs, test_xs, train_ys, test_ys


def create_model(hp):
    # create model
    # tf.compat.v1.keras.backend.clear_session()

    cnn = models.Sequential()
    # convolutional layer with rectified linear unit activation
    cnn.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=PARAMS["w_size"])
    )
    # 32 convolution filters used each of size 3x3
    # again
    cnn.add(Conv2D(64, (3, 3), activation="relu"))
    # 64 convolution filters used each of size 3x3
    # choose the best features via pooling
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly turn neurons on and off to improve convergence

    hp_dropout_1 = hp.Choice("dropout_rate_1", values=[0.0, 0.2, 0.4, 0.6])
    cnn.add(Dropout(hp_dropout_1))  # PARAMS["dropouts"][0]

    # flatten since too many dimensions, we only want a classification output
    cnn.add(Flatten())
    # fully connected to get all relevant data
    cnn.add(Dense(128, activation="relu"))
    # one more dropout for convergence' sake :)

    hp_dropout_2 = hp.Choice("dropout_rate_2", values=[0.0, 0.2, 0.4, 0.6])
    cnn.add(Dropout(hp_dropout_2))  # PARAMS["dropouts"][1]

    # output a softmax to squash the matrix into output probabilities
    cnn.add(Dense(1, activation="sigmoid"))

    hp_l_rate = hp.Choice("learning_rate", values=[1e-3, 1e-4])
    opt = optimizers.Adam(learning_rate=hp_l_rate)  # PARAMS["l_rate"]

    hp_lab_smooth = hp.Choice("label_smoothing", values=[0.0, 0.1, 0.2])
    loss = losses.BinaryCrossentropy(label_smoothing=hp_lab_smooth)
    # PARAMS["smoothing"]

    cnn.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    return cnn


# ===========
# TRAIN MODEL
# ===========
# Define the Sliding window generator
class SlidingWindowDataGenerator(utils.Sequence):
    def __init__(self, x, y, window_size, batch_size, shuffle=True):
        self.x, self.y = x, y
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Problem: the last window
        # self.indices = np.arange(len(self.x))
        self.indices = np.arange(len(self.x) - (self.window_size[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        # O problema está aqui
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_x = []
        batch_y = []
        for i in batch_indices:
            window = self.x[i : i + self.window_size[0], :].reshape(*self.window_size)
            label = self.y[i + (self.window_size[0] // 2)]
            batch_x.append(window)
            batch_y.append(label)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y, dtype=int)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            # Random state is set to SEED to ensure reproducibility
            rng = np.random.default_rng(PARAMS["seed"])
            rng.shuffle(self.indices)


def plot_losses(history, t_stamp, param):
    plt.plot(history.history["loss"], label=f"{param} - Train loss")
    plt.plot(history.history["val_loss"], label=f"{param} - Valid loss")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    # Save picture
    plt.savefig(RES / f"loss_{param}_{t_stamp}.pdf")

    # Plot the accuracy
    plt.plot(history.history["accuracy"], label=f"{param} - Train accuracy")
    plt.plot(history.history["val_accuracy"], label=f"{param} - Valid accuracy")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    # Save picture
    plt.savefig(RES / f"accuracy_{param}_{t_stamp}.pdf")


def main(t_stamp):
    params = pd.Series(PARAMS)
    print("   ***  PARAMS  ***   ")
    print(params.to_string())
    params.to_csv(RES / f"params_{t_stamp}.csv")

    with h5py.File(DATA_FP) as f:
        train_folds, valid_folds = stack_fold_data(
            f, pad_width=((3, 3), (0, 0)), fts=PARAMS["fts"]
        )

    train_xs, train_ys = train_folds[0]
    valid_xs, valid_ys = valid_folds[0]

    # Print the shapes of the train and valid sets
    print("Train set size:", len(train_xs))
    print("Valid set size:", len(valid_xs))

    # Create the DataGenerators for train and valid sets
    train_generator = SlidingWindowDataGenerator(
        train_xs,
        train_ys,
        window_size=PARAMS["w_size"],
        batch_size=PARAMS["b_size"],
        shuffle=True,
    )
    valid_generator = SlidingWindowDataGenerator(
        valid_xs,
        valid_ys,
        window_size=PARAMS["w_size"],
        batch_size=PARAMS["b_size"],
        shuffle=True,
    )

    # Instantiate new model with backend session cleared.
    # This is done inside the create_model function
    # Otherwise, the model will continue training from the previous fold

    # train_ds = tf.keras.utils.timeseries_dataset_from_array(
    #     x_train,
    #     y_train,
    #     sequence_length=WINDOW_SIZE[0],
    #     batch_size=BATCH_SIZE,
    # )

    # cnn = create_model()

    tuner = kt.Hyperband(
        create_model,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        directory="kt_dir",
        project_name="saulo",
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    tuner.search(
        train_generator,
        epochs=30,
        # validation_split=0.2,
        validation_data=valid_generator,
        callbacks=[stop_early],
    )

    print(tuner.results_summary())

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)

    # -----------------
    # Train the model
    # -----------------
    # Find the optimal number of epochs to train the model
    # with the hyperparameters obtained from the search.

    # Build the model with the optimal hyperparameters and train it on the data
    # for 50 epochs

    with h5py.File(DATA_FP) as f:
        train_all_xs, test_xs, train_all_ys, test_ys = load_main_train_test_data(
            f, pad_width=((3, 3), (0, 0)), fts=PARAMS["fts"]
        )

    # Create the DataGenerators for train and valid sets
    train_all_generator = SlidingWindowDataGenerator(
        train_all_xs,
        train_all_ys,
        window_size=PARAMS["w_size"],
        batch_size=PARAMS["b_size"],
        shuffle=True,
    )
    test_generator = SlidingWindowDataGenerator(
        test_xs,
        test_ys,
        window_size=PARAMS["w_size"],
        batch_size=PARAMS["b_size"],
        shuffle=True,
    )

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        train_all_generator,
        epochs=30,
        # validation_split=0.2,
        validation_data=test_generator,
    )

    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print("Best epoch:", best_epoch)

    # Re-instantiate the hypermodel and train it
    # with the optimal number of epochs from above.

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(
        train_all_generator,
        epochs=best_epoch,
        # validation_split=0.2,
        validation_data=test_generator,
    )

    # eval_result = hypermodel.evaluate(img_test, label_test)
    # print("[test loss, test accuracy]:", eval_result)

    # # Fit model on training data
    # history = cnn.fit(
    #     train_generator,
    #     validation_data=valid_generator,
    #     epochs=PARAMS["epochs"],
    #     verbose=1,
    # )

    # Save the model with the fold number
    hypermodel.save(MODEL / f"cnn_{PARAMS['fts']}_{t_stamp}.h5")

    # Save the history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_json(RES / f"history_{PARAMS['fts']}_{t_stamp}.json")

    # ========================
    # VISUALIZE TRAINING DATA
    # ========================
    # We plot the accuracy and loss for each fold stocked in the histories list

    plot_losses(history, t_stamp, param=PARAMS["fts"])


if __name__ == "__main__":
    start_time = datetime.now()
    t_stamp = start_time.strftime("%Y-%m-%dT%H-%M")  # -%S

    try:
        main(t_stamp)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    duration = datetime.now() - start_time
    print(f"Duration: {duration}")
