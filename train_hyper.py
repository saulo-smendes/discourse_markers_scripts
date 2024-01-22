import platform
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV, KFold
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence, to_categorical


if platform.system() == "Darwin":
    DATA = Path("~/Documents/LISN/DATA/Saulo/data").expanduser()
else:
    DATA = Path("/people/evrard/DATA/Saulo/data")

MODEL = Path("models")
RES = Path("results")

TRAIN_VALID_FP = DATA / "merged_CSTR-KEELE_file_20230227.hdf5"
MODEL_FP = MODEL / "20230131_BRNN_2D_WIN7_BATCH250_KEELE-CSTR.h5"
RES_FP = RES / "20230131_BRNN_model_valid_KEELE-CSTR_after-mod.txt"

# Define the GridSearch
RANDOM = True

# Seed
SEED = 42

# Window size
WINDOW_SIZE = (7, 38, 1)

# Number of samples to input in the hyperparameter tuning network
# The ideal would be to use all the train samples,
# but it is not possible due to memory limitations
SAMPLES = 100_000

#### Set seed ####
np.random.seed(SEED)

#### Set GPU ####
"""
# Check if GPU is available and print the name of the GPU device
if tf.test.is_gpu_available():
    print("GPU is available")
    print(tf.config.list_physical_devices("GPU"))

# Create a TensorFlow session with GPU memory growth enabled
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""


# Define function to stack data
# We change the function to customize the number of folds
# In this specific case, we want to have just one fold for training and evaluation
# We also use the shuffle parameter to shuffle the data before splitting it into folds
def stack_split_data(f, validation_split=0.25):
    """
    Stack and fold the data from the h5py file.
    f: h5py file

    """
    # we want to pad the last 7 rows with zeros
    pad_width = ((3, 4), (0, 0))
    # constant mode use edge values to fill padded regions with zeros as default
    mode = "constant"

    # Get the list of groups in the h5py file
    groups = list(f.keys())

    # Filter out the groups containing "_egg" not "_" from the groups list
    groups = [group for group in groups if "_egg" not in group and "_" in group]

    # Create a dictionary to associate the index of the group as a key
    # to the group name as the value
    group_dict = {i: group for i, group in enumerate(groups)}

    # Attention: instead of using KFold, we will build an array with the indices
    # of the groups and randomly pick items for training and validation sets
    # This is necessary because KFold does not allow to have a group with only one fold:
    # n_splits must be at least 2
    # We will use the same random state to make sure that the same groups are picked
    # for training and validation
    ids_array = np.arange(len(groups))
    np.random.seed(SEED)
    np.random.shuffle(ids_array)
    training_ratio = 1 - validation_split
    train_ids = ids_array[: int(len(ids_array) * training_ratio)]
    valid_ids = ids_array[int(len(ids_array) * training_ratio) :]
    train_arrays, valid_arrays = [], []

    # Now we iterate over the groups and stack the data
    for train_i in train_ids:
        group_name = group_dict[train_i]
        # Get the dataset in group
        data = np.array(f[group_name])
        # But we need to pad the edges of the 0-th axis with the same values
        # of the first and last lines
        padded_data = np.pad(data, pad_width=pad_width, mode=mode)
        # Add the id of the group as a column of the data
        padded_data = np.hstack(
            (padded_data, np.ones((padded_data.shape[0], 1)) * train_i)
        )
        # Append the padded data to the list
        train_arrays.append(padded_data)
    train_data = np.vstack(train_arrays)
    # Now we do the same but for the validation data
    for valid_i in valid_ids:
        group_name = group_dict[valid_i]
        data = np.array(f[group_name])
        padded_data = np.pad(data, pad_width=pad_width, mode=mode)
        padded_data = np.hstack(
            (padded_data, np.ones((padded_data.shape[0], 1)) * valid_i)
        )
        valid_arrays.append(padded_data)
    valid_data = np.vstack(valid_arrays)

    return train_data, valid_data, group_dict


# Define the Sliding window generator
class SlidingWindowDataGenerator(Sequence):
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
            label = self.y[i + (self.window_size[0] // 2)]  # We need to the label
            batch_x.append(window)
            batch_y.append(label)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y, dtype=int)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            # Random state is set to SEED to ensure reproducibility
            np.random.seed(SEED)
            np.random.shuffle(self.indices)


# Define the function to create the model with the given hyperparameters
def create_model(dropout_01=0.5, dropout_02=0.5, activation="sigmoid"):
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(7, 38, 1)))
    cnn.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(dropout_01))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(dropout_02))
    cnn.add(Dense(1, activation=activation))
    # cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return cnn


if __name__ == "__main__":

    # Import the data
    with h5py.File(TRAIN_VALID_FP) as f:
        train_data, test_data, group_dict = stack_split_data(f)

    # We subset the train_data array
    x_hyper, y_hyper = train_data[:, 2:40], train_data[:, 1]

    # We create the SlidingWindowDataGenerator
    hyper_generator = SlidingWindowDataGenerator(
        x_hyper, y_hyper, WINDOW_SIZE, batch_size=train_data.shape[0], shuffle=True
    )   # batch_size=SAMPLES

    # We create the validation subset
    x_hyper, y_hyper = hyper_generator[0]

    # Define and wrap the model
    tf.random.set_seed(SEED)

    # Clear the session
    tf.keras.backend.clear_session()

    # Wrap the Keras model into a scikit-learn estimator
    clf = KerasClassifier(
        model=create_model,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
        metrics="accuracy",
        loss="binary_crossentropy",
        optimizer="adam",
        optimizer__learning_rate=0.001,
        model__dropout_01=0.5,
        model__dropout_02=0.5,
        verbose=1,
    )

    # Define the parameters to be tuned
    params = {
        "batch_size": [16, 32, 64, 128, 256],
        "epochs": [5, 10, 20, 30],
        "optimizer": ["Adam", "RMSprop", "Adadelta"],
        "optimizer__learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
        "model__dropout_01": np.arange(0.1, 0.6, 0.1),
        "model__dropout_02": np.arange(0.1, 0.6, 0.1),
    }

    if RANDOM:
        # Run HalvingRandomSearchCV with the defined parameters and estimator
        search = HalvingRandomSearchCV(
            estimator=clf,
            param_distributions=params,
            factor=2,
            n_candidates=6,
            min_resources="exhaust",
            scoring="accuracy",
            n_jobs=1,
            cv=3,
            verbose=2,
        )
    else:
        # Run HalvingGridSearchCV with the defined parameters and estimator
        search = HalvingGridSearchCV(
            estimator=clf,
            param_grid=params,
            factor=2,
            min_resources="exhaust",
            scoring="accuracy",
            n_jobs=1,
            cv=3,
            verbose=2,
        )

    # Fit the search object to the data
    rsh = search.fit(x_hyper, y_hyper)

    # Print the best parameters
    print(rsh.best_params_)

    # Plot the results
    results = pd.DataFrame(rsh.cv_results_)
    results["params_str"] = results.params.apply(str)
    results.drop_duplicates(subset=("params_str", "iter"), inplace=True)
    mean_scores = results.pivot(
        index="iter", columns="params_str", values="mean_test_score"
    )
    ax = mean_scores.plot(legend=False, alpha=0.6)

    labels = [
        f"iter={i}\nn_samples={rsh.n_resources_[i]}\nn_candidates={rsh.n_candidates_[i]}"
        for i in range(rsh.n_iterations_)
    ]

    ax.set_xticks(range(rsh.n_iterations_))
    ax.set_xticklabels(labels, rotation=45, multialignment="left")
    ax.set_title("Scores of candidates over iterations")
    ax.set_ylabel("mean test score", fontsize=15)
    ax.set_xlabel("iterations", fontsize=15)
    # plt.tight_layout()
    # plt.show()

    # Save the plot
    plt.savefig(RES / "hyperparameter_tuning.pdf", bbox_inches="tight")

    # Save the results
    results.to_csv(RES / "hyperparameter_tuning.csv")
