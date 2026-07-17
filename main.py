import json
import logging
import math
import os
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LOGGER = logging.getLogger("btc_volatility")


# Модуль подготовки данных

NETWORK_FEATURES = ["unique_addresses", "transfer_volume_btc", "avg_fee_usd"]
MARKET_FREQUENCY = "5min"
EXPECTED_INTRADAY_POINTS = 288
SUMMARY_STATISTICS = ("mean", "std", "median")
LOG_SCALE_DISTRIBUTION_METRICS = {"MAE", "MSE", "RMSE", "MAPE", "QLIKE"}


def load_config(config_path):
    path = Path(config_path).expanduser()

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return config, path.resolve().parent


def resolve_path(path_value, project_root):
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return project_root / path


def to_utc_datetime(values):
    converted = pd.to_datetime(values, errors="coerce", utc=True)

    if converted.isna().any():
        count = int(converted.isna().sum())
        raise ValueError(f"Некорректных временных меток: {count}")

    return converted


def load_market_data(config, project_root):
    path = resolve_path(config["paths"]["market_5m_csv"], project_root)

    if not path.exists():
        raise FileNotFoundError(f"Нет файла рыночных данных: {path}")

    required = ["timestamp", "close", "volume"]
    df = pd.read_csv(path)
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"В файле {path} отсутствуют обязательные колонки: {missing}")

    result = pd.DataFrame()
    result["timestamp"] = to_utc_datetime(df["timestamp"])

    for column in ["close", "volume"]:
        result[column] = pd.to_numeric(df[column], errors="coerce")

    result = result.dropna(subset=["close"])
    result = result.drop_duplicates(subset=["timestamp"], keep="last")
    result = result.sort_values("timestamp").reset_index(drop=True)

    if result.empty:
        raise ValueError(f"Нет корректных рыночных данных: {path}")

    if (result["close"] <= 0).any():
        raise ValueError("Цена close должна быть положительной")

    return result


def load_network_data(config, project_root):
    path = resolve_path(config["paths"]["network_daily_csv"], project_root)

    if not path.exists():
        raise FileNotFoundError(f"Нет файла сетевых данных: {path}")

    features = list(NETWORK_FEATURES)
    df = pd.read_csv(path)
    required = ["date", *features]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"В файле {path} отсутствуют обязательные колонки: {missing}")

    result = pd.DataFrame()
    result["date_utc"] = to_utc_datetime(df["date"]).dt.floor("D")

    for feature in features:
        result[feature] = pd.to_numeric(df[feature], errors="coerce")

    if result[features].isna().any().any():
        missing = result[features].isna().sum().to_dict()
        raise ValueError(f"В сетевых признаках есть пропуски или нечисловые значения: {missing}")

    result = result.groupby("date_utc", as_index=False)[features].mean()
    result = result.sort_values("date_utc").reset_index(drop=True)

    return result


def calculate_daily_realized_volatility(market_df):
    market = market_df.set_index("timestamp")
    grid = pd.date_range(
        start=market.index.min().floor(MARKET_FREQUENCY),
        end=market.index.max().floor(MARKET_FREQUENCY),
        freq=MARKET_FREQUENCY,
        tz="UTC",
    )
    aligned = market.reindex(grid)
    aligned["date_utc"] = aligned.index.floor("D")
    original_points = aligned["close"].notna().groupby(aligned["date_utc"]).sum()
    daily_volume = aligned["volume"].fillna(0.0).groupby(aligned["date_utc"]).sum()
    unique_close_prices = aligned["close"].groupby(aligned["date_utc"]).nunique()

    aligned["previous_close"] = aligned["close"].shift(1)
    aligned["log_return"] = np.log(aligned["close"] / aligned["previous_close"])
    returns = aligned[["date_utc", "log_return"]].replace([np.inf, -np.inf], np.nan).dropna()
    daily = returns.groupby("date_utc", as_index=False).agg(
        realized_variance=("log_return", lambda values: float(np.square(values).sum())),
        intraday_returns=("log_return", "count"),
    )
    daily["original_intraday_points"] = daily["date_utc"].map(original_points).fillna(0).astype(int)
    daily["daily_volume"] = daily["date_utc"].map(daily_volume).fillna(0.0).astype(float)
    daily["unique_close_prices"] = daily["date_utc"].map(unique_close_prices).fillna(0).astype(int)
    before = len(daily)
    daily = daily[
        (daily["original_intraday_points"] == EXPECTED_INTRADAY_POINTS)
        & (daily["intraday_returns"] == EXPECTED_INTRADAY_POINTS)
    ].copy()
    dropped = before - len(daily)

    if dropped:
        LOGGER.debug("Исключено неполных UTC-суток: %s.", dropped)

    if daily.empty:
        raise ValueError("Нет полных UTC-суток с 288 интервалами")

    invalid_market_days = (
        (daily["realized_variance"] <= 0.0)
        | (
            (daily["unique_close_prices"] <= 1)
            & (daily["daily_volume"] <= 0.0)
        )
    )
    invalid_count = int(invalid_market_days.sum())

    if invalid_count:
        LOGGER.debug("Исключено неинформативных UTC-суток: %s.", invalid_count)
        daily = daily[~invalid_market_days].copy()

    if daily.empty:
        raise ValueError("Нет пригодных UTC-суток")

    daily["realized_volatility"] = np.sqrt(daily["realized_variance"])
    daily = daily[["date_utc", "realized_variance", "realized_volatility"]]

    return daily


def align_daily_calendar(
    df,
    value_columns,
    start_date,
    end_date,
    interpolation_flag_column,
):
    calendar = pd.DataFrame(
        {
            "date_utc": pd.date_range(
                start=start_date,
                end=end_date,
                freq="D",
                tz="UTC",
            )
        }
    )
    aligned = calendar.merge(df[["date_utc", *value_columns]], on="date_utc", how="left")
    aligned[interpolation_flag_column] = aligned[value_columns].isna().any(axis=1)

    for column in value_columns:
        aligned[column] = aligned[column].interpolate(
            method="linear",
            limit_area="inside",
        )

    return aligned


def build_daily_dataset(config, project_root):
    market = load_market_data(config, project_root)
    daily = calculate_daily_realized_volatility(market)
    features = list(NETWORK_FEATURES)
    calendar_start = daily["date_utc"].min()
    calendar_end = daily["date_utc"].max()
    use_network_features = "extended" in config["experiment"]["feature_sets"]

    if use_network_features:
        network = load_network_data(config, project_root)
        calendar_start = max(calendar_start, network["date_utc"].min())
        calendar_end = min(calendar_end, network["date_utc"].max())

        if calendar_start > calendar_end:
            raise ValueError("Нет общего диапазона дат")

    daily = align_daily_calendar(
        df=daily,
        value_columns=["realized_volatility"],
        start_date=calendar_start,
        end_date=calendar_end,
        interpolation_flag_column="is_realized_volatility_interpolated",
    )

    if use_network_features:
        network = align_daily_calendar(
            df=network,
            value_columns=features,
            start_date=calendar_start,
            end_date=calendar_end,
            interpolation_flag_column="is_network_interpolated",
        )
        merged = daily.merge(network, on="date_utc", how="left")
        validation_features = features
    else:
        merged = daily
        validation_features = []

    before_drop = len(merged)
    merged = merged.dropna(subset=["realized_volatility", *validation_features]).copy()
    dropped = before_drop - len(merged)
    restored_volatility = int(merged["is_realized_volatility_interpolated"].sum())
    restored_network = int(merged["is_network_interpolated"].sum()) if use_network_features else 0

    if restored_volatility or restored_network:
        LOGGER.debug(
            "Интерполированы пропуски: волатильность=%s, сетевые признаки=%s.",
            restored_volatility,
            restored_network,
        )

    if dropped:
        LOGGER.debug("Исключено граничных суток с пропусками: %s.", dropped)

    output_columns = ["date_utc", "realized_volatility", *validation_features]
    merged = merged[output_columns].sort_values("date_utc").reset_index(drop=True)
    validate_daily_dataset(merged, validation_features)
    output_path = resolve_path(config["paths"]["daily_dataset_csv"], project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    LOGGER.info("Подготовлен суточный набор: %s строк, %s.", len(merged), output_path)

    return merged


def validate_daily_dataset(df, network_features):
    required = ["date_utc", "realized_volatility", *network_features]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"В суточном наборе данных отсутствуют колонки: {missing}")

    numeric_columns = ["realized_volatility", *network_features]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("В суточных данных есть NaN или inf")

    if (numeric < 0).any():
        raise ValueError("В суточных данных есть отрицательные значения")


# Модуль предобработки данных

def add_log_features(df, config):
    epsilon = float(config["experiment"]["epsilon"])
    network_features = list(NETWORK_FEATURES) if "extended" in config["experiment"]["feature_sets"] else []
    result = df.copy().sort_values("date_utc").reset_index(drop=True)
    result["realized_volatility"] = pd.to_numeric(result["realized_volatility"], errors="coerce")
    result["log_realized_volatility"] = np.log(result["realized_volatility"].clip(lower=0.0) + epsilon)

    for feature in network_features:
        values = pd.to_numeric(result[feature], errors="coerce")
        result[f"log_{feature}"] = np.log1p(values)

    return result


def scale_features_for_window(
    df,
    feature_columns,
    train_start,
    train_end,
    scaler_name,
):
    values = df[feature_columns].to_numpy(dtype=np.float64)
    scaler_types = {"standard": StandardScaler, "minmax": MinMaxScaler}
    scaler = scaler_types[scaler_name]()
    scaler.fit(values[train_start:train_end])
    scaled = scaler.transform(values)

    return scaled.astype(np.float32)


def make_sequences(
    features,
    y_log,
    target_start,
    target_end,
    input_window,
    forecast_horizon,
    model_name,
    min_input_index,
):
    windows = []
    n_features = features.shape[1]
    first_target = max(target_start, min_input_index + input_window + forecast_horizon - 1)
    target_indices = np.arange(first_target, target_end, dtype=np.int64)

    for target_index in target_indices:
        input_end = target_index - forecast_horizon
        input_start = input_end - input_window + 1
        windows.append(features[input_start : input_end + 1])

    x = np.asarray(windows, dtype=np.float32)
    y = y_log[target_indices].astype(np.float32)

    if model_name == "mlp":
        x = x.reshape((x.shape[0], input_window * n_features))

    return x, y, target_indices


def make_train_test_sequences(
    data,
    scaled_features,
    model_name,
    train_start,
    train_end,
    test_start,
    test_end,
    config,
):
    experiment_config = config["experiment"]
    input_window = int(experiment_config["input_window"])
    forecast_horizon = int(experiment_config["forecast_horizon"])
    y_log = data["log_realized_volatility"].to_numpy(dtype=np.float64)
    x_train, y_train, _ = make_sequences(
        features=scaled_features,
        y_log=y_log,
        target_start=train_start,
        target_end=train_end,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        model_name=model_name,
        min_input_index=train_start,
    )
    x_test, y_test, target_indices = make_sequences(
        features=scaled_features,
        y_log=y_log,
        target_start=test_start,
        target_end=test_end,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        model_name=model_name,
        min_input_index=train_start,
    )

    return x_train, y_train, x_test, y_test, target_indices


# Модуль моделей и оценки качества

def calculate_metrics(
    y_true_sigma,
    y_pred_sigma,
    epsilon,
    y_reference_sigma,
    metric_names,
):
    true_sigma = np.asarray(y_true_sigma, dtype=np.float64)
    pred_sigma = np.asarray(y_pred_sigma, dtype=np.float64)
    pred_sigma = np.maximum(pred_sigma, epsilon)
    reference_sigma = None

    if "DA" in metric_names and y_reference_sigma is not None:
        reference_sigma = np.asarray(y_reference_sigma, dtype=np.float64)

    metric_values = {}
    errors = pred_sigma - true_sigma

    if "MAE" in metric_names:
        metric_values["MAE"] = float(np.mean(np.abs(errors)))

    if "MSE" in metric_names or "RMSE" in metric_names or "R2" in metric_names:
        squared_errors = np.square(errors)

        if "MSE" in metric_names or "RMSE" in metric_names:
            mse = float(np.mean(squared_errors))

            if "MSE" in metric_names:
                metric_values["MSE"] = mse

            if "RMSE" in metric_names:
                metric_values["RMSE"] = float(math.sqrt(mse))

        if "R2" in metric_names:
            residual_sum_of_squares = float(np.sum(squared_errors))
            total_sum_of_squares = float(np.sum(np.square(true_sigma - np.mean(true_sigma))))

            if math.isclose(total_sum_of_squares, 0.0, abs_tol=epsilon):
                r2 = 1.0 if math.isclose(residual_sum_of_squares, 0.0, abs_tol=epsilon) else 0.0
            else:
                r2 = 1.0 - residual_sum_of_squares / total_sum_of_squares

            metric_values["R2"] = float(r2)

    if "MAPE" in metric_names:
        denominator = true_sigma + epsilon
        metric_values["MAPE"] = float(np.mean(np.abs(errors) / denominator) * 100.0)

    if "QLIKE" in metric_names:
        variance_floor = epsilon**2
        true_variance = np.maximum(np.square(true_sigma), variance_floor)
        pred_variance = np.maximum(np.square(pred_sigma), variance_floor)
        variance_ratio = true_variance / pred_variance
        metric_values["QLIKE"] = float(np.mean(variance_ratio - np.log(variance_ratio) - 1.0))

    if "DA" in metric_names:
        metric_values["DA"] = calculate_directional_accuracy(true_sigma, pred_sigma, reference_sigma)

    return {metric_name: metric_values[metric_name] for metric_name in metric_names}


def calculate_directional_accuracy(
    true_sigma,
    pred_sigma,
    reference_sigma,
):
    if reference_sigma is None:
        if len(true_sigma) < 2:
            return 0.0

        actual_direction = np.sign(np.diff(true_sigma))
        predicted_direction = np.sign(np.diff(pred_sigma))
    else:
        actual_direction = np.sign(true_sigma - reference_sigma)
        predicted_direction = np.sign(pred_sigma - reference_sigma)

    valid_direction = actual_direction != 0

    if not valid_direction.any():
        return 0.0

    return float(np.mean(actual_direction[valid_direction] == predicted_direction[valid_direction]) * 100.0)


def qlike_loss(y_true_log, y_pred_log):
    y_true_log = tf.reshape(tf.cast(y_true_log, y_pred_log.dtype), tf.shape(y_pred_log))
    log_variance_ratio = 2.0 * (y_true_log - y_pred_log)
    variance_ratio = tf.exp(log_variance_ratio)
    loss = variance_ratio - log_variance_ratio - 1.0

    return tf.reduce_mean(loss, axis=-1)


def get_training_loss(config):
    loss_name = config["training"]["loss"]

    if loss_name == "qlike":
        return qlike_loss

    if loss_name == "huber":
        return keras.losses.Huber()

    return loss_name


def build_mlp(input_shape, config):
    model_config = config["mlp"]
    training_config = config["training"]
    model = keras.Sequential(name="MLP")
    model.add(keras.Input(shape=input_shape))

    for units in model_config["hidden_units"]:
        model.add(layers.Dense(int(units), activation=model_config["activation"]))

        if float(model_config["dropout"]) > 0:
            model.add(layers.Dropout(float(model_config["dropout"])))

    model.add(layers.Dense(1, activation="linear"))
    optimizer = keras.optimizers.Adam(
        learning_rate=float(training_config["learning_rate"]),
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss=get_training_loss(config), metrics=["mae"])

    return model


def build_lstm(input_shape, config):
    model_config = config["lstm"]
    training_config = config["training"]
    units_list = [int(units) for units in model_config["units"]]
    model = keras.Sequential(name="LSTM")
    model.add(keras.Input(shape=input_shape))

    for index, units in enumerate(units_list):
        model.add(
            layers.LSTM(
                units,
                return_sequences=index < len(units_list) - 1,
                dropout=float(model_config["dropout"]),
                recurrent_dropout=float(model_config["recurrent_dropout"]),
            )
        )

    for units in model_config["dense_units"]:
        model.add(layers.Dense(int(units), activation=model_config["activation"]))

    model.add(layers.Dense(1, activation="linear"))
    optimizer = keras.optimizers.Adam(
        learning_rate=float(training_config["learning_rate"]),
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss=get_training_loss(config), metrics=["mae"])

    return model


def save_plots(
    predictions,
    metrics_by_fold,
    summary,
    metric_names,
    plots_dir,
):
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_data = predictions.copy()
    plot_data["target_date"] = pd.to_datetime(plot_data["target_date"])

    for (model_name, feature_set), group in plot_data.groupby(["model_name", "feature_set"]):
        model_slug = "".join(char if char.isalnum() else "_" for char in str(model_name).lower()).strip("_")
        feature_slug = "".join(char if char.isalnum() else "_" for char in str(feature_set).lower()).strip("_")
        by_date = (
            group.groupby("target_date", as_index=False)
            .agg(y_true_sigma=("y_true_sigma", "first"), y_pred_sigma=("y_pred_sigma", "mean"))
            .sort_values("target_date")
        )
        save_volatility_plot(
            plot_data=by_date,
            title=f"{model_name} + {feature_set}: фактическая и прогнозная волатильность",
            path=plots_dir / f"actual_vs_predicted_{model_slug}_{feature_slug}.png",
        )

    labels = summary["model_name"].astype(str) + " + " + summary["feature_set"].astype(str)
    mean_metric_columns = [column for column in summary.columns if column.startswith("mean_")]
    median_metric_columns = [column for column in summary.columns if column.startswith("median_")]

    save_metrics_comparison_plot(summary, labels, mean_metric_columns, plots_dir / "metrics_comparison.png")
    save_metrics_comparison_plot(summary, labels, median_metric_columns, plots_dir / "metrics_comparison_median.png")

    for metric_name in metric_names:
        metric_slug = "".join(char if char.isalnum() else "_" for char in metric_name.lower()).strip("_")
        save_metric_distribution_plot(
            metrics_by_fold=metrics_by_fold,
            metric_name=metric_name,
            path=plots_dir / f"metrics_distribution_{metric_slug}.png",
        )


def save_volatility_plot(plot_data, title, path):
    plt.figure(figsize=(12, 5))
    plt.plot(plot_data["target_date"], plot_data["y_true_sigma"], label="Фактическая волатильность")
    plt.plot(plot_data["target_date"], plot_data["y_pred_sigma"], label="Прогнозная волатильность")
    plt.title(title)
    plt.xlabel("Дата")
    plt.ylabel("Реализованная волатильность")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_metrics_comparison_plot(
    summary,
    labels,
    metric_columns,
    path,
):
    n_columns = min(3, len(metric_columns))
    n_rows = math.ceil(len(metric_columns) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(5.5 * n_columns, 4.5 * n_rows), squeeze=False)
    flat_axes = axes.ravel()

    for axis, metric in zip(flat_axes, metric_columns, strict=False):
        axis.bar(labels, summary[metric].astype(float))
        axis.set_title(metric)
        axis.set_xlabel("Конфигурация")
        axis.set_ylabel("Значение")
        axis.tick_params(axis="x", rotation=45)

    for axis in flat_axes[len(metric_columns) :]:
        axis.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metric_distribution_plot(metrics_by_fold, metric_name, path):
    distribution_values = []
    labels = []

    for (model_name, feature_set), group in metrics_by_fold.groupby(["model_name", "feature_set"], sort=True):
        values = group[metric_name].to_numpy(dtype=float)
        distribution_values.append(values)
        labels.append(f"{model_name} + {feature_set}\n(n={values.size})")

    fig, axis = plt.subplots(figsize=(10, 6))
    boxplot = axis.boxplot(
        distribution_values,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        showfliers=True,
        medianprops={"color": "#1A1A1A", "linewidth": 2},
        meanprops={
            "marker": "D",
            "markerfacecolor": "#2CA02C",
            "markeredgecolor": "#1A1A1A",
            "markersize": 6,
        },
        whiskerprops={"color": "#4A4A4A", "linewidth": 1.2},
        capprops={"color": "#4A4A4A", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#D62728",
            "markeredgecolor": "#D62728",
            "alpha": 0.7,
            "markersize": 5,
        },
    )

    color_map = plt.get_cmap("tab10")

    for index, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(color_map(index % color_map.N))
        box.set_alpha(0.8)

    use_log_scale = metric_name in LOG_SCALE_DISTRIBUTION_METRICS and all(
        np.all(values > 0) for values in distribution_values
    )

    if use_log_scale:
        axis.set_yscale("log")
        axis.set_ylabel(f"{metric_name} (логарифмическая шкала)")
    else:
        axis.set_ylabel(metric_name)

    axis.set_title(f"Распределение {metric_name} по тестовым интервалам")
    axis.set_xlabel("Конфигурация модели")
    axis.grid(axis="y", which="both", linestyle="--", linewidth=0.7, alpha=0.45)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Line2D([0], [0], color="#1A1A1A", linewidth=2, label="Медиана"),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#2CA02C",
                markeredgecolor="#1A1A1A",
                markersize=6,
                label="Среднее",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D62728",
                markeredgecolor="#D62728",
                markersize=5,
                label="Выброс",
            ),
        ],
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_experiments(
    daily_df,
    config,
    project_root,
):
    set_random_seed(int(config["experiment"]["random_seed"]))
    result_paths = create_result_dirs(config, project_root)
    data = add_log_features(daily_df, config)
    folds = list(
        iter_sliding_windows(
            n_rows=len(data),
            train_window=int(config["experiment"]["train_window"]),
            test_window=int(config["experiment"]["test_window"]),
            step_size=int(config["experiment"]["step_size"]),
        )
    )

    if not folds:
        train_window = int(config["experiment"]["train_window"])
        test_window = int(config["experiment"]["test_window"])
        required_rows = train_window + test_window
        raise ValueError(f"Недостаточно данных: {len(data)} из {required_rows}")

    predictions_rows = []
    metrics_rows = []
    models = config["experiment"]["models"]
    feature_sets = config["experiment"]["feature_sets"]
    for feature_set in feature_sets:
        feature_columns = ["log_realized_volatility"]

        if feature_set == "extended":
            feature_columns.extend(f"log_{feature}" for feature in NETWORK_FEATURES)

        for model_name in models:
            for fold, train_start, train_end, test_start, test_end in tqdm(
                folds,
                desc=f"{model_name.upper()} + {feature_set}",
                total=len(folds),
            ):
                fold_predictions, fold_metrics = run_single_fold(
                    data=data,
                    feature_columns=feature_columns,
                    model_name=model_name,
                    feature_set=feature_set,
                    fold=fold,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    config=config,
                )
                predictions_rows.extend(fold_predictions)
                metrics_rows.append(fold_metrics)

    predictions = pd.DataFrame(predictions_rows)
    metrics_by_fold = pd.DataFrame(metrics_rows)
    enabled_metrics = config["experiment"]["metrics"]
    metrics_summary = summarize_metrics(metrics_by_fold, enabled_metrics)

    if config["saving"]["save_predictions"]:
        predictions.to_csv(result_paths["predictions"] / "all_predictions.csv", index=False)

    if config["saving"]["save_metrics"]:
        metrics_by_fold.to_csv(result_paths["metrics"] / "metrics_by_fold.csv", index=False)
        metrics_summary.to_csv(result_paths["metrics"] / "metrics_summary.csv", index=False)

    if config["saving"]["save_plots"]:
        save_plots(predictions, metrics_by_fold, metrics_summary, enabled_metrics, result_paths["plots"])

    LOGGER.info("Эксперимент завершен: %s запусков, результаты в %s.", len(metrics_rows), result_paths["base"])

    return predictions, metrics_by_fold, metrics_summary


def run_single_fold(
    data,
    feature_columns,
    model_name,
    feature_set,
    fold,
    train_start,
    train_end,
    test_start,
    test_end,
    config,
):
    experiment_config = config["experiment"]
    scaled_features = scale_features_for_window(
        df=data,
        feature_columns=feature_columns,
        train_start=train_start,
        train_end=train_end,
        scaler_name=experiment_config["scaler"],
    )
    x_train, y_train, x_test, y_test, target_indices = make_train_test_sequences(
        data=data,
        scaled_features=scaled_features,
        model_name=model_name,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        config=config,
    )

    y_pred_log = fit_and_predict_model(
        model_name=model_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        config=config,
        seed=int(experiment_config["random_seed"]) + fold,
    )
    y_true_sigma, y_pred_sigma, metrics = evaluate_predictions(
        data=data,
        target_indices=target_indices,
        y_pred_log=y_pred_log,
        config=config,
    )
    metrics_row = {
        "fold": fold,
        "model_name": model_name.upper(),
        "feature_set": feature_set,
        **metrics,
        "n_test": int(len(y_test)),
    }
    prediction_rows = build_prediction_rows(
        data=data,
        fold=fold,
        model_name=model_name,
        feature_set=feature_set,
        target_indices=target_indices,
        y_true_log=y_test,
        y_pred_log=y_pred_log,
        y_true_sigma=y_true_sigma,
        y_pred_sigma=y_pred_sigma,
    )

    return prediction_rows, metrics_row


def fit_and_predict_model(
    model_name,
    x_train,
    y_train,
    x_test,
    config,
    seed,
):
    model = fit_model(model_name, x_train, y_train, config, seed)
    training_config = config["training"]
    predictions = model.predict(
        x_test,
        batch_size=int(training_config["batch_size"]),
        verbose=int(training_config["verbose"]),
    ).reshape(-1)

    return predictions.astype(np.float64)


def evaluate_predictions(
    data,
    target_indices,
    y_pred_log,
    config,
):
    experiment_config = config["experiment"]
    epsilon = float(experiment_config["epsilon"])
    forecast_horizon = int(experiment_config["forecast_horizon"])
    y_pred_sigma = np.maximum(np.exp(np.asarray(y_pred_log, dtype=np.float64)) - epsilon, epsilon)
    realized_volatility = data["realized_volatility"].to_numpy(dtype=np.float64)
    y_true_sigma = realized_volatility[target_indices]
    y_reference_sigma = realized_volatility[target_indices - forecast_horizon]
    metrics = calculate_metrics(
        y_true_sigma,
        y_pred_sigma,
        epsilon,
        y_reference_sigma,
        config["experiment"]["metrics"],
    )

    return y_true_sigma, y_pred_sigma, metrics


def fit_model(
    model_name,
    x_train,
    y_train,
    config,
    seed,
):
    set_random_seed(seed)
    keras.backend.clear_session()
    builders = {
        "mlp": build_mlp,
        "lstm": build_lstm,
    }
    model = builders[model_name](tuple(x_train.shape[1:]), config)
    training_config = config["training"]
    validation_split = float(training_config["validation_split"])
    callbacks = build_callbacks(config, validation_split)
    model.fit(
        x_train,
        y_train,
        epochs=int(training_config["epochs"]),
        batch_size=int(training_config["batch_size"]),
        validation_split=validation_split,
        shuffle=False,
        verbose=int(training_config["verbose"]),
        callbacks=callbacks,
    )

    return model


def build_callbacks(config, validation_split):
    callbacks = []
    training_config = config["training"]
    monitor = "val_loss" if validation_split > 0 else "loss"

    if training_config["early_stopping"]:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=int(training_config["early_stopping_patience"]),
                restore_best_weights=True,
            )
        )

    return callbacks


def build_prediction_rows(
    data,
    model_name,
    feature_set,
    target_indices,
    y_true_log,
    y_pred_log,
    y_true_sigma,
    y_pred_sigma,
    fold=None,
):
    rows = []

    for position, target_index in enumerate(target_indices):
        row = {}

        if fold is not None:
            row["fold"] = fold

        row.update(
            {
                "model_name": model_name.upper(),
                "feature_set": feature_set,
                "target_date": pd.Timestamp(data.loc[int(target_index), "date_utc"]).date().isoformat(),
                "y_true_log": float(y_true_log[position]),
                "y_pred_log": float(y_pred_log[position]),
                "y_true_sigma": float(y_true_sigma[position]),
                "y_pred_sigma": float(y_pred_sigma[position]),
            }
        )
        rows.append(row)

    return rows


def iter_sliding_windows(
    n_rows,
    train_window,
    test_window,
    step_size,
):# -> Generator[tuple[int, Any | Literal[0], Any, Any, Any], An...:
    fold = 1
    start = 0

    while start + train_window + test_window <= n_rows:
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = train_end + test_window
        yield fold, train_start, train_end, test_start, test_end
        fold += 1
        start += step_size


def summarize_metrics(metrics_by_fold, metric_names):
    aggregations = {}

    for metric_name in metric_names:
        for statistic in SUMMARY_STATISTICS:
            aggregations[f"{statistic}_{metric_name}"] = (metric_name, statistic)

    aggregations["n_folds"] = ("fold", "nunique")

    return metrics_by_fold.groupby(["model_name", "feature_set"], as_index=False).agg(**aggregations).fillna(0.0)


def create_result_dirs(config, project_root):
    base = resolve_path(config["paths"]["results_dir"], project_root)
    result_paths = {
        "base": base,
        "predictions": base / "predictions",
        "metrics": base / "metrics",
        "plots": base / "plots",
    }

    for path in result_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return result_paths


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if hasattr(keras.utils, "set_random_seed"):
        keras.utils.set_random_seed(seed)


# Точка входа

CONFIG_FILE = globals().get("CONFIG_FILE", "config.json")


def main(config_file=CONFIG_FILE):
    project_root = Path.cwd().resolve()
    config_path = project_root / config_file
    config, project_root = load_config(config_path)
    LOGGER.info(
        "Запуск эксперимента: модели=%s, признаки=%s, loss=%s.",
        ", ".join(config["experiment"]["models"]),
        ", ".join(config["experiment"]["feature_sets"]),
        config["training"]["loss"],
    )

    daily_dataset = build_daily_dataset(config, project_root)
    predictions, metrics_by_fold, metrics_summary = run_experiments(daily_dataset, config, project_root)

    return predictions, metrics_by_fold, metrics_summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()