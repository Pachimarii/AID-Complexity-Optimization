import pickle
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    PredictionErrorDisplay,
    r2_score,
)


# Split function
def __prepare_splits(
    dataset,
    time_interval,
    sat_value,
    train_split_percent: float = 0.7,  # 70% for train and val; 30% for test
    val_split_percent: float = 0.2,  # 80% for train; 20% for val
):
    # transform confusion matrix features into rates
    for prefix in ["curr", "prev", "next2prev"]:
        dataset[f"{prefix}-TPR"] = dataset[f"{prefix}_tp"] / (
            dataset[f"{prefix}_tp"] + dataset[f"{prefix}_fn"]
        )
        dataset[f"{prefix}-TNR"] = dataset[f"{prefix}_tn"] / (
            dataset[f"{prefix}_tn"] + dataset[f"{prefix}_fp"]
        )
        dataset[f"{prefix}-FNR"] = dataset[f"{prefix}_fn"] / (
            dataset[f"{prefix}_tp"] + dataset[f"{prefix}_fn"]
        )
        dataset[f"{prefix}-FPR"] = dataset[f"{prefix}_fp"] / (
            dataset[f"{prefix}_tn"] + dataset[f"{prefix}_fp"]
        )

    # add delta column
    dataset["delta-TPR"] = dataset["avg-TPR-retrain"] - dataset["curr-TPR"]
    dataset["delta-TNR"] = dataset["avg-TNR-retrain"] - dataset["curr-TNR"]
    dataset["delta-TPR-nop"] = dataset["avg-TPR-no_retrain"] - dataset["curr-TPR"]
    dataset["delta-TNR-nop"] = dataset["avg-TNR-no_retrain"] - dataset["curr-TNR"]

    # split dataset into train, val, test
    duration = dataset["period_end_time"].max() - dataset["period_end_time"].min()

    # use train_split_percent % for training
    train_test_split_time = int(duration.days * train_split_percent) * 24  # in hours
    # ensure it is a multiple of the time_interval
    if train_test_split_time % time_interval != 0:
        train_test_split_time = train_test_split_time - (
            train_test_split_time % time_interval
        )
    test_start_time = dataset["prev_retrain_hour"].min() + train_test_split_time

    # split data set into train + val and test
    train_val_split = dataset.loc[dataset["curr_retrain_hour"] < test_start_time]
    test_split = dataset.loc[dataset["prev_retrain_hour"] >= test_start_time].copy()

    # split train into train (1-val_split_percent) and val (val_split_percent)
    val_split_time = train_val_split["curr_retrain_hour"].min() + int(
        train_test_split_time * (1 - val_split_percent)
    )

    train_split = train_val_split.loc[
        train_val_split["curr_retrain_hour"] < val_split_time
    ]
    val_split = train_val_split.loc[
        (train_val_split["prev_retrain_hour"] >= val_split_time)
        # & (train_val_split["curr_retrain_hour"] >= val_split_time)
    ].copy()

    # saturate features in val and test
    max_val = train_split["total_data"].max()
    print(max_val)
    val_split["total_data"] = val_split["total_data"].apply(
        lambda x: x if x < sat_value * max_val else sat_value * max_val
    )
    max_val = train_split["amount_old_data"].max()
    print(max_val)

    val_split["amount_old_data"] = val_split["amount_old_data"].apply(
        lambda x: x if x < sat_value * max_val else sat_value * max_val
    )
    max_val = train_split["ratio_new_old_data"].max()
    print(max_val)
    val_split["ratio_new_old_data"] = val_split["ratio_new_old_data"].apply(
        lambda x: x if x < sat_value * max_val else sat_value * max_val
    )
    print("prepare splits completed")
    return [train_split, val_split, test_split], test_start_time


def __init_benefits_models(
    model_type, nop_models, seed, retrain_models, train_split, val_split, test_split
):
    # specify model features
    benefits_model_features = [
        "amount_new_data",
        "amount_old_data",
        "total_data",
        "ratio_new_old_data",
        "retrain_delta",
        "curr_fraud_rate",
        "scores-JSD",
        "curr-TPR",
        "curr-TNR",
    ]

    if "delta" in model_type:
        target_TPR = "delta-TPR"
        target_TNR = "delta-TNR"
        target_TPR_nop = "delta-TPR-nop"
        target_TNR_nop = "delta-TNR-nop"
    else:
        target_TPR = "avg-TPR-retrain"
        target_TNR = "avg-TNR-retrain"
        target_TPR_nop = "avg-TPR-no_retrain"
        target_TNR_nop = "avg-TNR-no_retrain"
    table_dict = {}
    print(f"[D] Training TPR and TNR {model_type} models")
    # train models to predict delta-TP and delta-TN
    if "random_forest" in retrain_models:
        retrain_TPR_model = train_model_random_forest(
            seed,
            train_split[benefits_model_features],
            train_split[target_TPR],
        )
        retrain_TNR_model = train_model_random_forest(
            seed,
            train_split[benefits_model_features],
            train_split[target_TNR],
        )
        print("========= Evaluating TPR RB model validation performance ===========")
        table_dict["TPR Val"] = evaluate_performance(
            val_split[target_TPR],
            retrain_TPR_model.predict(val_split[benefits_model_features]),
        )
        print("========= Evaluating TPR RB model test performance ===========")
        table_dict["TPR Test"] = evaluate_performance(
            test_split[target_TPR],
            retrain_TPR_model.predict(test_split[benefits_model_features]),
        )
        print("========= Evaluating TNR RB model validation performance ===========")
        table_dict["TNR Val"] = evaluate_performance(
            val_split[target_TNR],
            retrain_TNR_model.predict(val_split[benefits_model_features]),
        )
        print("========= Evaluating TNR RB model test performance ===========")
        table_dict["TNR Test"] = evaluate_performance(
            test_split[target_TNR],
            retrain_TNR_model.predict(test_split[benefits_model_features]),
        )

    if "random_forest" in nop_models:
        print(f"[D] Training NOP TPR and TNR {model_type} models")
        nop_TPR_model = train_model_random_forest(
            seed,
            train_split[benefits_model_features],
            train_split[target_TPR_nop],
        )
        nop_TNR_model = train_model_random_forest(
            seed,
            train_split[benefits_model_features],
            train_split[target_TNR_nop],
        )
        print("========= Evaluating NOP TPR model validation performance ===========")
        table_dict["NOP TPR Val"] = evaluate_performance(
            val_split[target_TPR_nop],
            nop_TPR_model.predict(val_split[benefits_model_features]),
        )
        print("========= Evaluating NOP TPR model test performance ===========")
        table_dict["NOP TPR Test"] = evaluate_performance(
            test_split[target_TPR_nop],
            nop_TPR_model.predict(test_split[benefits_model_features]),
        )
        print("========= Evaluating NOP TNR model validation performance ===========")
        table_dict["NOP TNR Val"] = evaluate_performance(
            val_split[target_TNR_nop],
            nop_TNR_model.predict(val_split[benefits_model_features]),
        )
        print("========= Evaluating NOP TNR model test performance ===========")
        table_dict["NOP TNR Test"] = evaluate_performance(
            test_split[target_TNR_nop],
            nop_TNR_model.predict(test_split[benefits_model_features]),
        )
        # export recorded performance evaluation data
        df = pd.DataFrame.from_dict(
            table_dict,
            orient="index",
            columns=["MAE", "MAPE", "CORR COEF", "R^2 SCORE"],
        )
        df.to_excel("performance_evaluation.xlsx")
        return retrain_TPR_model, retrain_TNR_model, nop_TPR_model, nop_TNR_model

    # export recorded performance evaluation data
    df = pd.DataFrame.from_dict(
        table_dict, orient="index", columns=["MAE", "MAPE", "CORR COEF", "R^2 SCORE"]
    )
    df.to_excel("performance_evaluation.xlsx")
    return retrain_TPR_model, retrain_TNR_model


def evaluate_performance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[D] MAE = {mae}")
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"[D] MAPE = {mape}")
    corr_coef = stats.pearsonr(y_true, y_pred)
    print(f"[D] CORR_COEF = {corr_coef}")
    accuracy = r2_score(y_true, y_pred)
    print(f"[D] R^2 score = {accuracy}")
    # plot real vs predicted
    display = PredictionErrorDisplay(y_true=y_true, y_pred=y_pred)
    display.plot(kind="actual_vs_predicted")
    row = [mae, mape, corr_coef, accuracy]
    print("-------------------------------------------------------------")
    return row


def train_model_random_forest(seed, feature, target):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=seed,
    )
    model.fit(feature, target)
    return model


# Load dataset
with open("data/timeInterval_10-rand_sample.pkl", "rb") as f:
    dataset = pickle.load(f)
# print(dataset.head())
# print(list(dataset.columns))
# print(dataset["period_end_time"])

train_split_percent = 0.7  # 70% for train and val; 30% for test
val_split_percent = 0.2  # 80% for train; 20% for val

splits, test_start_time = __prepare_splits(
    dataset, 10, 0.9, train_split_percent, val_split_percent
)

models = __init_benefits_models(
    "delta",
    "random_forest",
    1,
    "random_forest",
    train_split=splits[0],
    val_split=splits[1],
    test_split=splits[2],
)

retrain_tpr_model = models[0]
retrain_tnr_model = models[1]

# if "random_forest" in nop_models:
nop_tpr_model = models[2]
nop_tnr_model = models[3]
