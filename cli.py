import logging
from datetime import datetime

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


def f(x):
    """
    This is the first function to be used before the concept drift.
    """
    return 0


def g(x):
    """
    This is the function used after the concept drift happened.
    """
    return -x


def f_then_g(row, last_index_f):
    # return f until last_index, then g
    if row.name <= last_index_f:
        return row["f"]
    else:
        return row["g"]


def cli(sample_count=10_000, step_size=100, concept_drift_after=0.1):
    # generate the dataset containing the concept drift
    df = get_training_df(sample_count, concept_drift_after)
    logging.info("created the training set")
    logging.info(
        "first part will be f(x), "
        f"concept drift to g(x) after {100 * concept_drift_after}% of samples"
    )
    print(df)

    start_time = datetime.utcnow()

    regressor = MLPRegressor(hidden_layer_sizes=(10,), max_iter=10_000)
    for i in range(step_size, sample_count + 1, step_size):
        # copying here is essential to really remove other samples
        df_train = pd.DataFrame(df[["x", "f_then_g"]].iloc[:i])
        logging.info("created the training set for the current step")

        # fit the regressor
        regressor.partial_fit(df_train[["x"]], df_train["f_then_g"])
        logging.info("performed partial fit")

        # mse on full set (should be test-set, I know)
        mse = mean_squared_error(df["f_then_g"], regressor.predict(df[["x"]]))
        logging.info(f"current fit ({i=}): {mse}")

        # test prediction on select samples
        df_sample = pd.DataFrame({"x": [1, 10, 100, 1000, 10000]})
        df_sample["f_then_g_pred"] = pd.Series(regressor.predict(df_sample))
        print(df_sample)

        print()
    logging.info("training set has now been fitted completely once")

    logging.info("performing partial fits on the full dataset, just because")
    for i in range(100):
        regressor.partial_fit(df[["x"]], df["f_then_g"])

    mse = mean_squared_error(df["f_then_g"], regressor.predict(df[["x"]]))
    print(f"final mse: {mse}")

    end_time = datetime.utcnow()
    took = end_time - start_time
    print(took)


def get_training_df(sample_count, concept_drift_after=0.5):
    # create a dataframe that starts with function f
    # but switches to function g after a specific index
    # to simulate a (hard) concept drift

    # create df with x, f(x), g(x)
    df = pd.DataFrame(index=range(sample_count))
    df["x"] = df.index
    df["f"] = df["x"].apply(f)
    df["g"] = df["x"].apply(g)

    # sample to shuffle data around
    df = df.sample(len(df))
    df.index = range(len(df))

    # switch halfway in
    last_index_f = int(sample_count * concept_drift_after)

    # use f below index, and g after
    df["f_then_g"] = df.apply(lambda r: f_then_g(r, last_index_f), axis=1)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
