import platform
from datetime import datetime
from pathlib import Path

import h5py
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

BEST_PARAMS = {
    "w_size": (7, 38, 1),
    "b_size": 512,  # Erros with b_size = 64 with GPUs
    "l_rate": 1e-3,
    "dropouts": (0.2, 0.4),
    "epochs": 20,
    "smoothing": 0.1,
    "seed": 42,
}

TOY_PARAMS = {
    "w_size": (7, 38, 1),
    "b_size": 1024,
    "l_rate": 1e-3,
    "dropouts": (0.2, 0.4),
    "epochs": 5,
    "smoothing": 0,
    "seed": 42,
}

PARAMS = BEST_PARAMS  # TOY_PARAMS, BEST_PARAMS
FOLD_MAX = 2  # 2, 5


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


def stack_fold_data(f, pad_width):
    """
    Stack and fold the data from the h5py file.
    f: h5py file
    """
    # groups = list(f.keys())
    # groups = [g for g in groups if "_egg" not in g and "_" in g]

    fname_df = pd.read_json(MODEL / "fnames.json")

    speakers = fname_df.speaker[fname_df.subset == "Train"].unique()
    # speakers = fname_df.speaker.unique()

    # TODO: use StratifiedKFold to spearate the data so that
    #       entire files separate for different training-evaluating folds

    kf = KFold(n_splits=5, shuffle=True, random_state=PARAMS["seed"])
    train_folds, valid_folds = [], []

    for idx, (train_speaker_ids, valid_speaker_ids) in enumerate(
        kf.split(speakers)
    ):
        train_mask = fname_df.speaker.isin(speakers[train_speaker_ids])
        train_groups = fname_df.fname[train_mask].to_list()
        train_data = prepare_data(train_groups, f, pad_width)
        train_folds.append(train_data)

        valid_mask = fname_df.speaker.isin(speakers[valid_speaker_ids])
        valid_groups = fname_df.fname[valid_mask].to_list()
        valid_data = prepare_data(valid_groups, f, pad_width)
        valid_folds.append(valid_data)

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


def create_model():
    # create model
    # tf.compat.v1.keras.backend.clear_session()

    cnn = models.Sequential()
    # convolutional layer with rectified linear unit activation
    cnn.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(7, 38, 1)))
    # 32 convolution filters used each of size 3x3
    # again
    cnn.add(Conv2D(64, (3, 3), activation="relu"))
    # 64 convolution filters used each of size 3x3
    # choose the best features via pooling
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly turn neurons on and off to improve convergence
    cnn.add(Dropout(PARAMS["dropouts"][0]))
    # flatten since too many dimensions, we only want a classification output
    cnn.add(Flatten())
    # fully connected to get all relevant data
    cnn.add(Dense(128, activation="relu"))
    # one more dropout for convergence' sake :)
    cnn.add(Dropout(PARAMS["dropouts"][1]))
    # output a softmax to squash the matrix into output probabilities
    cnn.add(Dense(1, activation="sigmoid"))

    opt = optimizers.Adam(learning_rate=PARAMS["l_rate"])  # , epsilon=0.1)
    loss = losses.BinaryCrossentropy(label_smoothing=PARAMS["smoothing"])

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


def plot_losses(history, fold_i, t_stamp):
    plt.plot(history.history["loss"], label=f"Fold {fold_i} - Train loss")
    plt.plot(history.history["val_loss"], label=f"Fold {fold_i} - Valid loss")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    # Save picture
    plt.savefig(RES / f"loss_fold_{fold_i}_{t_stamp}.pdf")

    # Plot the accuracy
    plt.plot(history.history["accuracy"], label=f"Fold {fold_i} - Train accuracy")
    plt.plot(history.history["val_accuracy"], label=f"Fold {fold_i} - Valid accuracy")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    # Save picture
    plt.savefig(RES / f"accuracy_fold_{fold_i}_{t_stamp}.pdf")


def main(t_stamp):
    params = pd.Series(PARAMS)
    print("    *** PARAMS ***    ")
    print(params.to_string())
    params.to_csv(RES / f"params_{t_stamp}.csv")

    with h5py.File(DATA_FP) as f:
        train_folds_list, valid_folds_list = stack_fold_data(
            f, pad_width=((3, 3), (0, 0))
        )

    # =============
    # DEFINE MODEL
    # =============
    # Build model layers
    # Define and wrap the model
    # import KerasClassifier from keras wrappers
    # Define the function that creates the model

    # Iterate over the folds and train/valid sets
    histories = []
    accuracy_df = pd.DataFrame()

    for fold_i, (train_fold, valid_fold) in enumerate(
        zip(train_folds_list, valid_folds_list)
    ):
        fold_i += 1

        # Split the data into train and valid sets
        # X whole feature set: 2:40
        # Y: 1
        x_train, y_train = train_fold[:, 2:40], train_fold[:, 1]
        x_valid, y_valid = valid_fold[:, 2:40], valid_fold[:, 1]

        # Print the shapes of the train and valid sets
        print(f"Fold {fold_i}:")
        print(f"Train set size: {len(x_train)}")
        print(f"Valid set size: {len(x_valid)}")

        # Create the DataGenerators for train and valid sets
        train_generator = SlidingWindowDataGenerator(
            x_train,
            y_train,
            window_size=PARAMS["w_size"],
            batch_size=PARAMS["b_size"],
            shuffle=True,
        )
        valid_generator = SlidingWindowDataGenerator(
            x_valid,
            y_valid,
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

        cnn = create_model()

        # Fit model on training data
        history = cnn.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=PARAMS["epochs"],
            verbose=1,
        )
        # history = cnn.fit(
        #     train_ds,
        #     validation_data=(x_valid, y_valid),
        #     epochs=EPOCHS,
        #     verbose=1,
        #     batch_size=BATCH_SIZE,
        # )

        # Save the model with the fold number
        cnn.save(MODEL / f"cnn_fold_{fold_i}_{t_stamp}.h5")

        # Append the history to the list
        histories.append(history)

        # Save the history
        hist_df = pd.DataFrame(history.history)
        hist_df.to_json(RES / f"history_fold_{fold_i}_{t_stamp}.json")

        accuracy_df[f"fold_{fold_i}"] = hist_df.val_accuracy

    # ========================
    # VISUALIZE TRAINING DATA
    # ========================
    # We plot the accuracy and loss for each fold stocked in the histories list

    accuracy_df["mean"] = accuracy_df.mean(axis="columns")
    accuracy_df["std"] = accuracy_df.std(axis="columns")
    accuracy_df.to_json(RES / f"accuracies_{t_stamp}.json")

    for fold_i, history in enumerate(histories):
        plot_losses(history, fold_i + 1, t_stamp)


if __name__ == "__main__":
    start_time = datetime.now()
    t_stamp = start_time.strftime("%Y-%m-%dT%H-%M")  # -%S

    try:
        main(t_stamp)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    duration = datetime.now() - start_time
    print(f"Duration: {duration}")
