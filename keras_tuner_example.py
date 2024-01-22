import keras_tuner
import tensorflow as tf
from tensorflow import keras


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    # hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    # model.add(keras.layers.Dense(units=hp_units, activation="relu"))
    model.add(keras.layers.Dense(units=128, activation="relu"))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":

    (img_train, label_train), (img_test, label_test) = (
        keras.datasets.fashion_mnist.load_data()
    )

    # Normalize pixel values between 0 and 1
    img_train = img_train.astype("float32") / 255.0
    img_test = img_test.astype("float32") / 255.0

    tuner = keras_tuner.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        directory="kt_test",
        project_name="intro_to_kt",
    )

    # The Hyperband tuning algorithm uses adaptive resource allocation
    # and early-stopping to quickly converge on a high-performing model.
    # This is done using a sports championship style bracket.
    # The algorithm trains a large number of models for a few epochs
    # and carries forward only the top-performing half of models to the next round.
    # Hyperband determines the number of models to train in a bracket by computing
    # 1 + log<sub>`factor`</sub>(`max_epochs`)
    # and rounding it up to the nearest integer.

    # Create a callback to stop training early after reaching a certain value
    # for the validation loss.

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    # Run the hyperparameter search.
    # The arguments for the search method are the same as those used for
    # `tf.keras.model.fit` in addition to the callback above.

    tuner.search(
        img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early]
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # print(
    #     f"""
    # Hyperparameter search complete.
    # Optimal number of units in the 1st densely-connected layer: {best_hps.get('units')}
    # and optimal learning rate for the optimizer: {best_hps.get('learning_rate')}
    # """
    # )

    # Train the model

    # Find the optimal number of epochs to train the model
    # with the hyperparameters obtained from the search.

    # Build the model with the optimal hyperparameters and train it on the data
    # for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print("Best epoch:", best_epoch)

    # Re-instantiate the hypermodel and train it
    # with the optimal number of epochs from above.

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

    # To finish this tutorial, evaluate the hypermodel on the test data.

    eval_result = hypermodel.evaluate(img_test, label_test)
    print("[test loss, test accuracy]:", eval_result)


    # The `my_dir/intro_to_kt` directory contains detailed logs
    # and checkpoints for every trial (model configuration) run
    # during the hyperparameter search.
    # If you re-run the hyperparameter search,
    # the Keras Tuner uses the existing state from these logs to resume the search.
    # To disable this behavior, pass an additional `overwrite=True` argument
    # while instantiating the tuner.
