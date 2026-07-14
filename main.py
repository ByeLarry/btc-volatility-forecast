import json
import logging
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Iterator

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
LOGGER = logging.getLogger("btc_volatility")


# Модуль подготовки данных

NETWORK_FEATURES = ["unique_addresses", "transfer_volume_btc", "avg_fee_usd"]
MARKET_FREQUENCY = "5min"
EXPECTED_INTRADAY_POINTS = 288
SUPPORTED_METRICS = ("MAE", "MSE", "RMSE", "MAPE", "QLIKE", "R2", "DA")
SUMMARY_STATISTICS = ("mean", "std", "median")
MAXIMIZED_METRICS = {"R2", "DA"}
LOG_SCALE_DISTRIBUTION_METRICS = {"MAE", "MSE", "RMSE", "MAPE", "QLIKE"}
SUPPORTED_LOSSES = ("mse", "mae", "huber", "qlike")
QLIKE_LOG_RATIO_LIMIT = 10.0


def normalize_metric_names(metric_names: Any) -> list[str]:
    """Проверяет и нормализует список метрик."""
    if not isinstance(metric_names, list) or not metric_names:
        raise ValueError("Параметр experiment.metrics должен быть непустым массивом названий метрик.")

    if any(not isinstance(name, str) or not name.strip() for name in metric_names):
        raise ValueError("Все элементы experiment.metrics должны быть непустыми строками.")

    normalized = [name.strip().upper() for name in metric_names]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    unknown = sorted(set(normalized) - set(SUPPORTED_METRICS))

    if duplicates:
        raise ValueError(f"В experiment.metrics указаны повторяющиеся метрики: {duplicates}")

    if unknown:
        raise ValueError(
            f"В experiment.metrics указаны неподдерживаемые метрики: {unknown}. "
            f"Допустимые значения: {list(SUPPORTED_METRICS)}"
        )

    return normalized


def get_enabled_metrics(config: dict[str, Any]) -> list[str]:
    """Возвращает показатели качества, заданные в настройках."""
    return normalize_metric_names(config["experiment"]["metrics"])


def resolve_selection_metric(config: dict[str, Any]) -> tuple[str, str, str]:
    """Проверяет метрику выбора модели и возвращает название столбца сводной таблицы."""
    enabled_metrics = get_enabled_metrics(config)
    raw_value = config["experiment"].get("selection_metric", enabled_metrics[0])

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("Параметр experiment.selection_metric должен быть непустой строкой.")

    selection_metric = raw_value.strip().lower()
    statistic = "mean"
    metric_name = selection_metric

    if "_" in selection_metric:
        statistic, metric_name = selection_metric.split("_", maxsplit=1)

        if statistic not in SUMMARY_STATISTICS:
            raise ValueError(
                "Префикс experiment.selection_metric должен быть mean, median или std. "
                "Также можно указать название метрики без префикса."
            )

    metric_name = metric_name.upper()

    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(
            f"В experiment.selection_metric указана неподдерживаемая метрика: {metric_name}. "
            f"Допустимые значения: {list(SUPPORTED_METRICS)}"
        )

    if metric_name not in enabled_metrics:
        raise ValueError(
            f"Метрика выбора {metric_name} запрещена текущей конфигурацией: "
            "добавьте ее в experiment.metrics или выберите другую selection_metric."
        )

    return f"{statistic}_{metric_name}", metric_name, statistic


def get_training_loss_name(config: dict[str, Any]) -> str:
    """Проверяет и возвращает название функции потерь."""
    raw_value = config["training"].get("loss", "mse")

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("Параметр training.loss должен быть непустой строкой.")

    loss_name = raw_value.strip().lower()

    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(
            f"В training.loss указана неподдерживаемая функция потерь: {loss_name}. "
            f"Допустимые значения: {list(SUPPORTED_LOSSES)}"
        )

    return loss_name


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Загружает файл настроек JSON и проверяет наличие обязательных разделов."""
    path = Path(config_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {path}. Поместите config.json рядом с main.py и укажите параметры эксперимента."
        )

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    validate_config(config)

    return config, path.resolve().parent


def validate_config(config: dict[str, Any]) -> None:
    """Проверяет наличие в настройках всех разделов, необходимых для запуска."""
    required_sections = ["paths", "experiment", "training", "mlp", "lstm", "saving"]
    missing = [section for section in required_sections if section not in config]

    if missing:
        raise ValueError(f"В config.json отсутствуют обязательные разделы: {missing}")

    required_paths = ["market_5m_csv", "network_daily_csv", "daily_dataset_csv", "results_dir"]
    missing_paths = [name for name in required_paths if name not in config["paths"]]

    if missing_paths:
        raise ValueError(f"В разделе paths отсутствуют параметры: {missing_paths}")


    required_experiment = [
        "input_window",
        "forecast_horizon",
        "train_window",
        "test_window",
        "step_size",
        "feature_sets",
        "models",
        "scaler",
        "epsilon",
        "random_seed",
        "metrics",
    ]
    missing_experiment = [name for name in required_experiment if name not in config["experiment"]]

    if missing_experiment:
        raise ValueError(f"В разделе experiment отсутствуют параметры: {missing_experiment}")

    validate_experiment_config(config)


def resolve_path(path_value: str, project_root: Path) -> Path:
    """Преобразует путь из настроек в абсолютный путь."""
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return project_root / path


def uses_extended_feature_set(config: dict[str, Any]) -> bool:
    """Проверяет, выбран ли расширенный набор признаков."""
    return "extended" in [name.lower() for name in config["experiment"]["feature_sets"]]


def should_train_final_model(config: dict[str, Any]) -> bool:
    """Проверяет, включено ли обучение итоговой модели."""
    return bool(config["experiment"].get("train_final_model", True))


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    """Проверяет наличие обязательных столбцов во входной таблице."""
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(f"В файле {source} отсутствуют обязательные колонки: {missing}")


def to_utc_datetime(values: pd.Series) -> pd.Series:
    """Приводит временные метки к формату UTC с указанием часового пояса."""
    converted = pd.to_datetime(values, errors="coerce", utc=True)

    if converted.isna().any():
        count = int(converted.isna().sum())
        raise ValueError(f"Не удалось распознать {count} временных меток.")

    return converted


def load_market_data(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Загружает локальные 5-минутные рыночные данные Биткойна из файла CSV."""
    path = resolve_path(config["paths"]["market_5m_csv"], project_root)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл рыночных данных не найден: {path}. "
            "Укажите существующий CSV с 5-минутными рыночными данными."
        )

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(path)
    require_columns(df, required, str(path))
    result = pd.DataFrame()
    result["timestamp"] = to_utc_datetime(df["timestamp"])

    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(df[column], errors="coerce")

    result = result.dropna(subset=["timestamp", "close"])
    result = result.drop_duplicates(subset=["timestamp"], keep="last")
    result = result.sort_values("timestamp").reset_index(drop=True)

    if result.empty:
        raise ValueError(f"Файл рыночных данных не содержит корректных строк: {path}")

    if (result["close"] <= 0).any():
        raise ValueError("Цена закрытия должна быть положительной для расчета логарифмических доходностей.")

    return result


def load_network_data(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Загружает суточные показатели сети Биткойн из файла CSV."""
    path = resolve_path(config["paths"]["network_daily_csv"], project_root)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл суточных сетевых признаков не найден: {path}. "
            "Этот файл нужен только для расширенного набора признаков. "
            "Чтобы запустить эксперимент без сетевых признаков, оставьте только базовый набор "
            "в параметре experiment.feature_sets."
        )

    features = list(NETWORK_FEATURES)
    df = pd.read_csv(path)
    require_columns(df, ["date", *features], str(path))
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


def calculate_daily_realized_volatility(market_df: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает дневную реализованную дисперсию и волатильность по 5-минутным ценам закрытия."""
    market = market_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    market = market.set_index("timestamp")
    grid = pd.date_range(
        start=market.index.min().floor(MARKET_FREQUENCY),
        end=market.index.max().floor(MARKET_FREQUENCY),
        freq=MARKET_FREQUENCY,
        tz="UTC",
    )
    aligned = market.reindex(grid)
    aligned["close"] = pd.to_numeric(aligned["close"], errors="coerce")
    aligned["volume"] = pd.to_numeric(aligned["volume"], errors="coerce")
    aligned["date_utc"] = aligned.index.floor("D")
    original_points = aligned["close"].notna().groupby(aligned["date_utc"]).sum()
    daily_volume = aligned["volume"].fillna(0.0).groupby(aligned["date_utc"]).sum()
    unique_close_prices = aligned["close"].groupby(aligned["date_utc"]).nunique()

    if aligned.empty:
        raise ValueError("После выравнивания 5-минутной сетки не осталось цен закрытия.")

    if (aligned["close"].dropna() <= 0).any():
        raise ValueError("Для расчета логарифмических доходностей все цены закрытия должны быть положительными.")

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
        LOGGER.info("Исключено UTC-суток без 288 корректных 5-минутных доходностей: %s.", dropped)

    if daily.empty:
        raise ValueError(
            "После исключения неполных суток не осталось наблюдений. "
            "Проверьте период данных или ожидаемое число 5-минутных интервалов в сутках."
        )

    invalid_market_days = (
        (daily["realized_variance"] <= 0.0)
        | (
            (daily["unique_close_prices"] <= 1)
            & (daily["daily_volume"] <= 0.0)
        )
    )
    invalid_count = int(invalid_market_days.sum())

    if invalid_count:
        LOGGER.info(
            "Исключено UTC-суток с нулевой реализованной дисперсией или плоской ценой при нулевом объеме: %s.",
            invalid_count,
        )
        daily = daily[~invalid_market_days].copy()

    if daily.empty:
        raise ValueError("После исключения нулевых или неинформативных UTC-суток не осталось наблюдений.")

    daily["realized_volatility"] = np.sqrt(daily["realized_variance"])
    daily = daily[["date_utc", "realized_variance", "realized_volatility"]]
    validate_daily_targets(daily)

    return daily


def align_daily_calendar(
    df: pd.DataFrame,
    value_columns: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    interpolation_flag_column: str,
) -> pd.DataFrame:
    """Выравнивает суточные данные по полному календарю UTC и интерполирует внутренние пропуски."""
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


def build_daily_dataset(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Формирует итоговый суточный набор данных и сохраняет его в каталоге data/processed."""
    LOGGER.info("Загрузка локальных 5-минутных рыночных данных BTC.")
    market = load_market_data(config, project_root)
    LOGGER.info("Расчет дневной реализованной волатильности.")
    daily = calculate_daily_realized_volatility(market)
    features = list(NETWORK_FEATURES)
    calendar_start = daily["date_utc"].min()
    calendar_end = daily["date_utc"].max()

    if uses_extended_feature_set(config):
        LOGGER.info("Загрузка суточных сетевых признаков для расширенного набора.")
        network = load_network_data(config, project_root)
        calendar_start = max(calendar_start, network["date_utc"].min())
        calendar_end = min(calendar_end, network["date_utc"].max())

        if calendar_start > calendar_end:
            raise ValueError("Рыночные и сетевые данные не имеют общего диапазона дат.")

        daily = align_daily_calendar(
            df=daily,
            value_columns=["realized_volatility"],
            start_date=calendar_start,
            end_date=calendar_end,
            interpolation_flag_column="is_realized_volatility_interpolated",
        )
        network = align_daily_calendar(
            df=network,
            value_columns=features,
            start_date=calendar_start,
            end_date=calendar_end,
            interpolation_flag_column="is_network_interpolated",
        )
        merged = daily.merge(network, on="date_utc", how="left")
        before_drop = len(merged)
        merged = merged.dropna(subset=["realized_volatility", *features]).copy()
        dropped = before_drop - len(merged)
        restored_volatility = int(merged["is_realized_volatility_interpolated"].sum())
        restored_network = int(merged["is_network_interpolated"].sum())

        if restored_volatility or restored_network:
            LOGGER.info(
                "Интерполированы пропуски в суточном календаре: волатильность=%s, сетевые признаки=%s.",
                restored_volatility,
                restored_network,
            )

        if dropped:
            LOGGER.info(
                "После интерполяции исключено граничных суток с невосстановленными пропусками: %s.",
                dropped,
            )

        output_columns = [
            "date_utc",
            "realized_volatility",
            *features,
        ]
        validation_features = features
    else:
        LOGGER.info("В конфигурации выбран только базовый набор признаков: сетевые признаки не загружаются.")
        merged = align_daily_calendar(
            df=daily,
            value_columns=["realized_volatility"],
            start_date=calendar_start,
            end_date=calendar_end,
            interpolation_flag_column="is_realized_volatility_interpolated",
        )
        before_drop = len(merged)
        merged = merged.dropna(subset=["realized_volatility"]).copy()
        dropped = before_drop - len(merged)
        restored_volatility = int(merged["is_realized_volatility_interpolated"].sum())

        if restored_volatility:
            LOGGER.info(
                "Интерполированы пропуски в суточном календаре: волатильность=%s.",
                restored_volatility,
            )

        if dropped:
            LOGGER.info(
                "После интерполяции исключено граничных суток с невосстановленными пропусками: %s.",
                dropped,
            )

        output_columns = ["date_utc", "realized_volatility"]
        validation_features = []

    merged = merged[output_columns].sort_values("date_utc").reset_index(drop=True)
    validate_daily_dataset(merged, validation_features)
    output_path = resolve_path(config["paths"]["daily_dataset_csv"], project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    LOGGER.info("Итоговый суточный набор сохранен в %s.", output_path)

    return merged

def validate_daily_targets(df: pd.DataFrame) -> None:
    """Проверяет корректность целевого столбца с реализованной волатильностью."""
    column = "realized_volatility"

    if column not in df.columns:
        raise ValueError(f"В таблице отсутствует обязательная колонка {column}.")

    values = pd.to_numeric(df[column], errors="coerce")

    if values.isna().any():
        raise ValueError(f"В колонке {column} есть NaN.")

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"В колонке {column} есть бесконечные значения.")

    if (values < 0).any():
        raise ValueError(f"В колонке {column} есть отрицательные значения.")

    zero_count = int((values == 0).sum())

    if zero_count:
        LOGGER.warning(
            "Найдено нулевых значений реализованной волатильности: %s. "
            "При логарифмировании будет использована малая добавка из конфигурации.",
            zero_count,
        )


def validate_daily_dataset(df: pd.DataFrame, network_features: list[str]) -> None:
    """Проверяет итоговый суточный набор данных перед обучением моделей."""
    required = ["date_utc", "realized_volatility", *network_features]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"В суточном наборе данных отсутствуют колонки: {missing}")

    validate_daily_targets(df)

    if df[network_features].isna().any().any():
        missing_counts = df[network_features].isna().sum().to_dict()
        raise ValueError(f"В сетевых признаках итогового набора остались пропуски: {missing_counts}")

    numeric = df[["realized_volatility", *network_features]].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Итоговый суточный набор содержит NaN или бесконечные значения.")


# Модуль предобработки данных

def add_log_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Добавляет логарифм целевой переменной и логарифмы сетевых признаков."""
    epsilon = float(config["experiment"]["epsilon"])
    network_features = list(NETWORK_FEATURES) if uses_extended_feature_set(config) else []
    result = df.copy().sort_values("date_utc").reset_index(drop=True)
    result["realized_volatility"] = pd.to_numeric(result["realized_volatility"], errors="coerce")
    result["log_realized_volatility"] = np.log(result["realized_volatility"].clip(lower=0.0) + epsilon)

    for feature in network_features:
        if feature not in result.columns:
            raise ValueError(f"Для расширенного набора признаков отсутствует колонка {feature}.")

        values = pd.to_numeric(result[feature], errors="coerce")

        if values.isna().any():
            raise ValueError(f"В сетевом признаке {feature} есть NaN или нечисловые значения.")

        if (values < 0).any():
            raise ValueError(f"Сетевой признак {feature} должен быть неотрицательным перед логарифмированием.")

        result[f"log_{feature}"] = np.log1p(values)

    if not np.isfinite(result["log_realized_volatility"].to_numpy()).all():
        raise ValueError("После логарифмирования целевой переменной получены некорректные значения.")

    return result


def get_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Возвращает список столбцов для базового или расширенного набора признаков."""
    if feature_set == "base":
        return ["log_realized_volatility"]

    if feature_set == "extended":
        network_features = [f"log_{feature}" for feature in NETWORK_FEATURES]
        missing = [feature for feature in network_features if feature not in df.columns]

        if missing:
            raise ValueError(f"Для расширенного набора признаков отсутствуют колонки: {missing}")

        return ["log_realized_volatility", *network_features]

    raise ValueError(f"Неизвестный набор признаков в config.json: {feature_set}")


def scale_features_for_window(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_start: int,
    train_end: int,
    scaler_name: str,
) -> tuple[np.ndarray, StandardScaler | MinMaxScaler]:
    """Масштабирует признаки и возвращает обученный преобразователь."""
    values = df[feature_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(f"В признаках есть NaN или бесконечные значения: {feature_columns}")

    scaler = create_scaler(scaler_name)
    scaler.fit(values[train_start:train_end])
    scaled = scaler.transform(values)

    if not np.isfinite(scaled).all():
        raise ValueError("После масштабирования признаков получены некорректные значения.")

    return scaled.astype(np.float32), scaler


def create_scaler(name: str) -> StandardScaler | MinMaxScaler:
    """Создает преобразователь масштаба по названию из файла настроек config.json."""
    if name == "standard":
        return StandardScaler()

    if name == "minmax":
        return MinMaxScaler()

    raise ValueError(f"Неизвестный способ масштабирования в config.json: {name}. Допустимы standard или minmax.")


def make_sequences(
    features: np.ndarray,
    y_log: np.ndarray,
    target_start: int,
    target_end: int,
    input_window: int,
    forecast_horizon: int,
    model_name: str,
    min_input_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Формирует обучающие или проверочные последовательности без использования будущих значений."""
    if forecast_horizon != 1:
        raise ValueError("В данной реализации поддерживается горизонт прогноза 1 день.")

    if input_window < 1:
        raise ValueError("Длина входного окна должна быть положительной.")

    if model_name not in {"mlp", "lstm"}:
        raise ValueError(f"Неизвестная модель в config.json: {model_name}")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    target_indices: list[int] = []
    n_features = features.shape[1]

    for target_index in range(target_start, target_end):
        input_end = target_index - forecast_horizon
        input_start = input_end - input_window + 1

        if input_start < min_input_index:
            continue

        window = features[input_start : input_end + 1]

        if len(window) != input_window:
            continue

        windows.append(window)
        targets.append(float(y_log[target_index]))
        target_indices.append(target_index)

    if not windows:
        return empty_sequences(input_window, n_features, model_name)

    x = np.asarray(windows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    indices = np.asarray(target_indices, dtype=np.int64)

    if model_name == "mlp":
        x = x.reshape((x.shape[0], input_window * n_features))

    return x, y, indices


def empty_sequences(
    input_window: int,
    n_features: int,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает пустой массив нужной формы, если для шага проверки нет допустимых окон."""
    if model_name == "mlp":
        x = np.empty((0, input_window * n_features), dtype=np.float32)
    else:
        x = np.empty((0, input_window, n_features), dtype=np.float32)

    return x, np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)


def make_train_test_sequences(
    data: pd.DataFrame,
    scaled_features: np.ndarray,
    model_name: str,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Формирует обучающие и проверочные последовательности для заданного временного разреза."""
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

def inverse_log_volatility(y_log: np.ndarray, epsilon: float) -> np.ndarray:
    """Переводит прогноз из логарифмической шкалы в исходную шкалу реализованной волатильности."""
    sigma = np.exp(np.asarray(y_log, dtype=np.float64)) - epsilon

    return np.maximum(sigma, epsilon)


def calculate_metrics(
    y_true_sigma: np.ndarray,
    y_pred_sigma: np.ndarray,
    epsilon: float,
    y_reference_sigma: np.ndarray | None = None,
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Считает метрики на исходной шкале реализованной волатильности."""
    selected_metrics = list(SUPPORTED_METRICS) if metric_names is None else normalize_metric_names(metric_names)
    true_sigma = np.asarray(y_true_sigma, dtype=np.float64)
    pred_sigma = np.asarray(y_pred_sigma, dtype=np.float64)

    if epsilon <= 0:
        raise ValueError("Малая добавка epsilon должна быть положительной.")

    if true_sigma.shape != pred_sigma.shape:
        raise ValueError("Фактические и прогнозные значения должны иметь одинаковую форму.")

    if not np.isfinite(true_sigma).all() or not np.isfinite(pred_sigma).all():
        raise ValueError("В данных для расчета метрик есть NaN или бесконечные значения.")

    if (true_sigma < 0).any():
        raise ValueError("Фактическая реализованная волатильность не может быть отрицательной.")

    pred_sigma = np.maximum(pred_sigma, epsilon)
    reference_sigma = None

    if "DA" in selected_metrics and y_reference_sigma is not None:
        reference_sigma = np.asarray(y_reference_sigma, dtype=np.float64)

        if true_sigma.shape != reference_sigma.shape:
            raise ValueError("Базовые значения для DA должны иметь ту же форму, что и фактические значения.")

        if not np.isfinite(reference_sigma).all():
            raise ValueError("В базовых значениях для DA есть NaN или бесконечные значения.")

        if (reference_sigma < 0).any():
            raise ValueError("Базовая реализованная волатильность для DA не может быть отрицательной.")

    metric_values: dict[str, float] = {}
    errors = pred_sigma - true_sigma

    if "MAE" in selected_metrics:
        metric_values["MAE"] = float(np.mean(np.abs(errors)))

    if "MSE" in selected_metrics or "RMSE" in selected_metrics or "R2" in selected_metrics:
        squared_errors = np.square(errors)

        if "MSE" in selected_metrics or "RMSE" in selected_metrics:
            mse = float(np.mean(squared_errors))

            if "MSE" in selected_metrics:
                metric_values["MSE"] = mse

            if "RMSE" in selected_metrics:
                metric_values["RMSE"] = float(math.sqrt(mse))

        if "R2" in selected_metrics:
            residual_sum_of_squares = float(np.sum(squared_errors))
            total_sum_of_squares = float(np.sum(np.square(true_sigma - np.mean(true_sigma))))

            if math.isclose(total_sum_of_squares, 0.0, abs_tol=epsilon):
                r2 = 1.0 if math.isclose(residual_sum_of_squares, 0.0, abs_tol=epsilon) else 0.0
            else:
                r2 = 1.0 - residual_sum_of_squares / total_sum_of_squares

            metric_values["R2"] = float(r2)

    if "MAPE" in selected_metrics:
        denominator = true_sigma + epsilon
        metric_values["MAPE"] = float(np.mean(np.abs(errors) / denominator) * 100.0)

    if "QLIKE" in selected_metrics:
        variance_floor = epsilon**2
        true_variance = np.maximum(np.square(true_sigma), variance_floor)
        pred_variance = np.maximum(np.square(pred_sigma), variance_floor)
        variance_ratio = true_variance / pred_variance
        metric_values["QLIKE"] = float(np.mean(variance_ratio - np.log(variance_ratio) - 1.0))

    if "DA" in selected_metrics:
        metric_values["DA"] = calculate_directional_accuracy(true_sigma, pred_sigma, reference_sigma)

    return {metric_name: metric_values[metric_name] for metric_name in selected_metrics}


def calculate_directional_accuracy(
    true_sigma: np.ndarray,
    pred_sigma: np.ndarray,
    reference_sigma: np.ndarray | None,
) -> float:
    """Считает долю совпадений направления изменения волатильности в процентах."""
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


@keras.utils.register_keras_serializable(package="btc_volatility")
def qlike_loss(y_true_log: tf.Tensor, y_pred_log: tf.Tensor) -> tf.Tensor:
    """Вычисляет дифференцируемую функцию потерь QLIKE по логарифмам регуляризованной дисперсии."""
    y_true_log = tf.reshape(tf.cast(y_true_log, y_pred_log.dtype), tf.shape(y_pred_log))
    log_variance_ratio = 2.0 * (y_true_log - y_pred_log)
    limit = tf.cast(QLIKE_LOG_RATIO_LIMIT, y_pred_log.dtype)
    capped_log_ratio = tf.minimum(log_variance_ratio, limit)
    capped_exponential = tf.exp(capped_log_ratio)
    regular_loss = capped_exponential - log_variance_ratio - 1.0
    limit_exponential = tf.exp(limit)
    loss_at_limit = limit_exponential - limit - 1.0
    linear_tail = loss_at_limit + (limit_exponential - 1.0) * (log_variance_ratio - limit)
    elementwise_loss = tf.where(log_variance_ratio <= limit, regular_loss, linear_tail)

    return tf.reduce_mean(elementwise_loss, axis=-1)


def get_training_loss(config: dict[str, Any]) -> Any:
    """Возвращает функцию потерь Keras, заданную в настройках обучения."""
    loss_name = get_training_loss_name(config)

    if loss_name == "qlike":
        return qlike_loss

    if loss_name == "huber":
        return keras.losses.Huber()

    return loss_name


def build_model(model_name: str, input_shape: tuple[int, ...], config: dict[str, Any]) -> keras.Model:
    """Создает модель по названию из файла настроек config.json."""
    if model_name == "mlp":
        return build_mlp(input_shape, config)

    if model_name == "lstm":
        return build_lstm(input_shape, config)

    raise ValueError(f"Неизвестная модель в config.json: {model_name}")


def build_mlp(input_shape: tuple[int, ...], config: dict[str, Any]) -> keras.Model:
    """Создает MLP для прогноза логарифма реализованной волатильности."""
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


def build_lstm(input_shape: tuple[int, ...], config: dict[str, Any]) -> keras.Model:
    """Создает LSTM для прогноза логарифма реализованной волатильности."""
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
    predictions: pd.DataFrame,
    metrics_by_fold: pd.DataFrame,
    summary: pd.DataFrame,
    metric_names: list[str],
    plots_dir: Path,
) -> None:
    """Сохраняет графики прогнозов, сводных метрик и их распределений по проверочным интервалам."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    if predictions.empty:
        raise ValueError("Нельзя построить графики: таблица прогнозов пуста.")

    plot_data = predictions.copy()
    plot_data["target_date"] = pd.to_datetime(plot_data["target_date"])

    for (model_name, feature_set), group in plot_data.groupby(["model_name", "feature_set"]):
        by_date = (
            group.groupby("target_date", as_index=False)
            .agg(y_true_sigma=("y_true_sigma", "first"), y_pred_sigma=("y_pred_sigma", "mean"))
            .sort_values("target_date")
        )
        save_volatility_plot(
            plot_data=by_date,
            title=f"{model_name} + {feature_set}: фактическая и прогнозная волатильность",
            path=plots_dir / f"actual_vs_predicted_{safe_filename(model_name)}_{safe_filename(feature_set)}.png",
        )

    if summary.empty:
        raise ValueError("Нельзя построить сравнение метрик: сводная таблица пуста.")

    labels = summary["model_name"].astype(str) + " + " + summary["feature_set"].astype(str)
    mean_metric_columns = [column for column in summary.columns if column.startswith("mean_")]
    median_metric_columns = [column for column in summary.columns if column.startswith("median_")]

    if not mean_metric_columns:
        raise ValueError("Нельзя построить сравнение метрик: в сводной таблице нет колонок mean_*.")

    if not median_metric_columns:
        raise ValueError("Нельзя построить сравнение медианных метрик: в сводной таблице нет колонок median_*.")

    save_metrics_comparison_plot(summary, labels, mean_metric_columns, plots_dir / "metrics_comparison.png")
    save_metrics_comparison_plot(summary, labels, median_metric_columns, plots_dir / "metrics_comparison_median.png")

    for metric_name in metric_names:
        save_metric_distribution_plot(
            metrics_by_fold=metrics_by_fold,
            metric_name=metric_name,
            path=plots_dir / f"metrics_distribution_{safe_filename(metric_name)}.png",
        )


def save_volatility_plot(plot_data: pd.DataFrame, title: str, path: Path) -> None:
    """Сохраняет график фактической и прогнозной волатильности."""
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
    summary: pd.DataFrame,
    labels: pd.Series,
    metric_columns: list[str],
    path: Path,
) -> None:
    """Сохраняет график сравнения выбранных метрик для вариантов моделей."""
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


def save_metric_distribution_plot(metrics_by_fold: pd.DataFrame, metric_name: str, path: Path) -> None:
    """Сохраняет распределение метрики по проверочным интервалам для каждого варианта модели."""
    required_columns = {"model_name", "feature_set", metric_name}
    missing_columns = sorted(required_columns.difference(metrics_by_fold.columns))

    if missing_columns:
        raise ValueError(f"Нельзя построить распределение {metric_name}: отсутствуют колонки {missing_columns}.")

    distribution_values = []
    labels = []

    for (model_name, feature_set), group in metrics_by_fold.groupby(["model_name", "feature_set"], sort=True):
        values = pd.to_numeric(group[metric_name], errors="coerce").to_numpy(dtype=float)

        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(
                f"Нельзя построить распределение {metric_name} для {model_name} + {feature_set}: "
                "обнаружены отсутствующие или некорректные значения."
            )

        distribution_values.append(values)
        labels.append(f"{model_name} + {feature_set}\n(n={values.size})")

    if not distribution_values:
        raise ValueError(f"Нельзя построить распределение {metric_name}: таблица метрик пуста.")

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


def safe_filename(value: str) -> str:
    """Преобразует название модели или набора признаков в безопасный фрагмент имени файла."""
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


VALID_MODELS = {"mlp", "lstm"}
VALID_FEATURE_SETS = {"base", "extended"}


def run_experiments(
    daily_df: pd.DataFrame,
    config: dict[str, Any],
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Запускает скользящую временную проверку для всех моделей и наборов признаков."""
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
        raise ValueError(
            "Недостаточно суточных наблюдений для скользящей проверки. "
            f"Доступно: {len(data)}; требуется минимум: {required_rows} "
            f"(обучающее окно {train_window} суток и проверочное окно {test_window} суток). "
            "Уменьшите окна в разделе experiment или загрузите более длинную историю."
        )

    predictions_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    models = [name.lower() for name in config["experiment"]["models"]]
    feature_sets = [name.lower() for name in config["experiment"]["feature_sets"]]
    for feature_set in feature_sets:
        feature_columns = get_feature_columns(data, feature_set)

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

    LOGGER.info("Выполнено запусков для шагов проверки, моделей и наборов признаков: %s.", len(folds) * len(models) * len(feature_sets))
    predictions = pd.DataFrame(predictions_rows)
    metrics_by_fold = pd.DataFrame(metrics_rows)
    enabled_metrics = get_enabled_metrics(config)
    metrics_summary = summarize_metrics(metrics_by_fold, enabled_metrics)
    save_results(predictions, metrics_by_fold, metrics_summary, config, result_paths)

    if config["saving"]["save_plots"]:
        save_plots(predictions, metrics_by_fold, metrics_summary, enabled_metrics, result_paths["plots"])

    return predictions, metrics_by_fold, metrics_summary


def run_single_fold(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    feature_set: str,
    fold: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Обучает одну модель на одном шаге проверки и возвращает прогнозы с метриками."""
    experiment_config = config["experiment"]
    scaled_features, _ = scale_features_for_window(
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

    if len(x_train) == 0:
        raise ValueError(
            f"Шаг проверки {fold}: обучающих последовательностей нет. "
            "Увеличьте параметр experiment.train_window или уменьшите experiment.input_window."
        )

    if len(x_test) == 0:
        raise ValueError(
            f"Шаг проверки {fold}: проверочных последовательностей нет. "
            "Проверьте параметры experiment.test_window, experiment.train_window и experiment.input_window."
        )

    LOGGER.info(
        "Обучение %s, набор признаков «%s», шаг проверки %s: обучающих примеров %s, проверочных примеров %s.",
        model_name.upper(),
        feature_set,
        fold,
        len(x_train),
        len(x_test),
    )
    _, y_pred_log = fit_and_predict_model(
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
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    config: dict[str, Any],
    seed: int,
    validation_split: float | None = None,
) -> tuple[keras.Model, np.ndarray]:
    """Обучает модель и возвращает ее вместе с прогнозом в логарифмической шкале."""
    model = fit_model(model_name, x_train, y_train, config, seed, validation_split)
    training_config = config["training"]
    predictions = model.predict(
        x_test,
        batch_size=int(training_config["batch_size"]),
        verbose=int(training_config["verbose"]),
    ).reshape(-1)

    return model, predictions.astype(np.float64)


def evaluate_predictions(
    data: pd.DataFrame,
    target_indices: np.ndarray,
    y_pred_log: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Восстанавливает волатильность и рассчитывает выбранные метрики прогноза."""
    experiment_config = config["experiment"]
    epsilon = float(experiment_config["epsilon"])
    forecast_horizon = int(experiment_config["forecast_horizon"])
    y_pred_sigma = inverse_log_volatility(y_pred_log, epsilon)
    realized_volatility = data["realized_volatility"].to_numpy(dtype=np.float64)
    y_true_sigma = realized_volatility[target_indices]
    y_reference_sigma = realized_volatility[target_indices - forecast_horizon]
    metrics = calculate_metrics(
        y_true_sigma,
        y_pred_sigma,
        epsilon,
        y_reference_sigma,
        get_enabled_metrics(config),
    )

    return y_true_sigma, y_pred_sigma, metrics


def fit_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    seed: int,
    validation_split: float | None = None,
) -> keras.Model:
    """Обучает модель с ранней остановкой и возвращает обученную модель Keras."""
    set_random_seed(seed)
    keras.backend.clear_session()
    model = build_model(model_name, tuple(x_train.shape[1:]), config)
    training_config = config["training"]
    effective_validation_split = (
        float(training_config["validation_split"]) if validation_split is None else float(validation_split)
    )
    callbacks = build_callbacks(config, effective_validation_split)
    model.fit(
        x_train,
        y_train,
        epochs=int(training_config["epochs"]),
        batch_size=int(training_config["batch_size"]),
        validation_split=effective_validation_split,
        shuffle=False,
        verbose=int(training_config["verbose"]),
        callbacks=callbacks,
    )

    return model


def build_callbacks(config: dict[str, Any], validation_split: float) -> list[Any]:
    """Создает обработчики Keras для ранней остановки."""
    callbacks: list[Any] = []
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
    data: pd.DataFrame,
    model_name: str,
    feature_set: str,
    target_indices: np.ndarray,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_sigma: np.ndarray,
    y_pred_sigma: np.ndarray,
    fold: int | None = None,
) -> list[dict[str, Any]]:
    """Формирует строки таблицы прогнозов для шага проверки или итоговой модели."""
    rows: list[dict[str, Any]] = []

    for position, target_index in enumerate(target_indices):
        row: dict[str, Any] = {}

        if fold is not None:
            row["fold"] = fold

        row.update(
            {
                "model_name": model_name.upper(),
                "feature_set": feature_set,
                "target_date": format_date(data.loc[int(target_index), "date_utc"]),
                "y_true_log": float(y_true_log[position]),
                "y_pred_log": float(y_pred_log[position]),
                "y_true_sigma": float(y_true_sigma[position]),
                "y_pred_sigma": float(y_pred_sigma[position]),
            }
        )
        rows.append(row)

    return rows


def iter_sliding_windows(
    n_rows: int,
    train_window: int,
    test_window: int,
    step_size: int,
) -> Iterator[tuple[int, int, int, int, int]]:
    """Формирует последовательность скользящих обучающих и проверочных окон."""
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


def summarize_metrics(metrics_by_fold: pd.DataFrame, metric_names: list[str] | None = None) -> pd.DataFrame:
    """Рассчитывает среднее значение, стандартное отклонение и медиану по шагам проверки для каждой пары модели и набора признаков."""
    if metrics_by_fold.empty:
        raise ValueError("Нет метрик для построения итоговой сводки.")

    selected_metrics = list(SUPPORTED_METRICS) if metric_names is None else normalize_metric_names(metric_names)
    missing_metrics = [metric_name for metric_name in selected_metrics if metric_name not in metrics_by_fold.columns]

    if missing_metrics:
        raise ValueError(f"В таблице метрик по фолдам отсутствуют выбранные метрики: {missing_metrics}")

    aggregations: dict[str, tuple[str, str]] = {}

    for metric_name in selected_metrics:
        for statistic in SUMMARY_STATISTICS:
            aggregations[f"{statistic}_{metric_name}"] = (metric_name, statistic)

    aggregations["n_folds"] = ("fold", "nunique")

    return metrics_by_fold.groupby(["model_name", "feature_set"], as_index=False).agg(**aggregations).fillna(0.0)


def save_results(
    predictions: pd.DataFrame,
    metrics_by_fold: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    config: dict[str, Any],
    result_paths: dict[str, Path],
) -> None:
    """Сохраняет прогнозы и метрики в файлы CSV."""
    if config["saving"]["save_predictions"]:
        path = result_paths["predictions"] / "all_predictions.csv"
        predictions.to_csv(path, index=False)
        LOGGER.info("Прогнозы сохранены в %s.", path)

    if config["saving"]["save_metrics"]:
        by_fold_path = result_paths["metrics"] / "metrics_by_fold.csv"
        summary_path = result_paths["metrics"] / "metrics_summary.csv"
        metrics_by_fold.to_csv(by_fold_path, index=False)
        metrics_summary.to_csv(summary_path, index=False)
        LOGGER.info("Метрики по шагам проверки сохранены в %s.", by_fold_path)
        LOGGER.info("Сводка метрик сохранена в %s.", summary_path)


def select_best_configuration(metrics_summary: pd.DataFrame, config: dict[str, Any]) -> tuple[str, str]:
    """Выбирает лучшую модель и набор признаков по основной метрике."""
    if metrics_summary.empty:
        raise ValueError("Нельзя выбрать лучшую конфигурацию: сводная таблица метрик пуста.")

    metric_column, base_metric, statistic = resolve_selection_metric(config)

    if metric_column not in metrics_summary.columns:
        raise ValueError(f"В сводной таблице метрик нет колонки для выбора лучшей конфигурации: {metric_column}")

    metric_values = metrics_summary[metric_column].astype(float)
    maximize_metric = base_metric in MAXIMIZED_METRICS and statistic != "std"
    best_index = metric_values.idxmax() if maximize_metric else metric_values.idxmin()
    selection_direction = "максимальному" if maximize_metric else "минимальному"
    best_row = metrics_summary.loc[best_index]
    model_name = str(best_row["model_name"]).lower()
    feature_set = str(best_row["feature_set"]).lower()
    LOGGER.info(
        "Выбрана итоговая конфигурация: %s + %s по %s %s.",
        model_name.upper(),
        feature_set,
        selection_direction,
        metric_column,
    )

    return model_name, feature_set


def train_final_model(
    daily_df: pd.DataFrame,
    config: dict[str, Any],
    project_root: Path,
    model_name: str,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Обучает итоговую модель на всем диапазоне, кроме последнего проверочного интервала, и сохраняет результаты."""
    result_paths = create_result_dirs(config, project_root)
    data = add_log_features(daily_df, config)
    feature_columns = get_feature_columns(data, feature_set)
    experiment_config = config["experiment"]
    input_window = int(experiment_config["input_window"])
    forecast_horizon = int(experiment_config["forecast_horizon"])
    test_window = int(experiment_config["test_window"])
    epsilon = float(experiment_config["epsilon"])
    train_start = 0
    test_end = len(data)
    test_start = test_end - test_window
    train_end = test_start

    if train_end <= input_window:
        raise ValueError(
            "Недостаточно суточных наблюдений для итогового train/test-разреза. "
            f"Доступно: {len(data)}; требуется больше {input_window + test_window}."
        )

    scaled_features, scaler = scale_features_for_window(
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

    if len(x_train) == 0:
        raise ValueError("Для итоговой модели не сформировались обучающие последовательности.")

    if len(x_test) == 0:
        raise ValueError("Для итоговой модели не сформировались тестовые последовательности.")

    LOGGER.info(
        "Обучение итоговой модели %s, набор признаков «%s»: обучающих примеров %s, тестовых примеров %s.",
        model_name.upper(),
        feature_set,
        len(x_train),
        len(x_test),
    )
    model, y_pred_log = fit_and_predict_model(
        model_name=model_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        config=config,
        seed=int(experiment_config["random_seed"]),
        validation_split=0.0,
    )
    y_true_sigma, y_pred_sigma, metrics = evaluate_predictions(
        data=data,
        target_indices=target_indices,
        y_pred_log=y_pred_log,
        config=config,
    )
    final_predictions = pd.DataFrame(
        build_prediction_rows(
            data=data,
            model_name=model_name,
            feature_set=feature_set,
            target_indices=target_indices,
            y_true_log=y_test,
            y_pred_log=y_pred_log,
            y_true_sigma=y_true_sigma,
            y_pred_sigma=y_pred_sigma,
        )
    )
    final_metrics = pd.DataFrame(
        [
            {
                "model_name": model_name.upper(),
                "feature_set": feature_set,
                **metrics,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "training_scope": "all_except_final_test_window",
                "final_train_start_date": format_date(data.loc[train_start, "date_utc"]),
                "final_train_end_date": format_date(data.loc[train_end - 1, "date_utc"]),
                "final_test_start_date": format_date(data.loc[test_start, "date_utc"]),
                "final_test_end_date": format_date(data.loc[test_end - 1, "date_utc"]),
            }
        ]
    )
    metadata = {
        "selected_model": model_name,
        "selected_feature_set": feature_set,
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "final_test_window": test_window,
        "training_scope": "all_except_final_test_window",
        "scaler": experiment_config["scaler"],
        "epsilon": epsilon,
        "feature_columns": feature_columns,
        "target_column": "log_realized_volatility",
        "n_train": int(len(y_train)),
        "validation_split": 0.0,
        "final_train_start_date": format_date(data.loc[train_start, "date_utc"]),
        "final_train_end_date": format_date(data.loc[train_end - 1, "date_utc"]),
        "final_test_start_date": format_date(data.loc[test_start, "date_utc"]),
        "final_test_end_date": format_date(data.loc[test_end - 1, "date_utc"]),
        "final_metrics": metrics,
    }
    save_final_artifacts(
        model=model,
        scaler=scaler,
        final_predictions=final_predictions,
        final_metrics=final_metrics,
        metadata=metadata,
        config=config,
        results_dir=result_paths["base"],
    )

    return final_predictions, final_metrics


def build_empty_final_predictions() -> pd.DataFrame:
    """Создает пустую таблицу итоговых прогнозов с ожидаемыми столбцами."""
    return pd.DataFrame(
        columns=[
            "model_name",
            "feature_set",
            "target_date",
            "y_true_log",
            "y_pred_log",
            "y_true_sigma",
            "y_pred_sigma",
        ]
    )


def save_final_artifacts(
    model: keras.Model,
    scaler: StandardScaler | MinMaxScaler,
    final_predictions: pd.DataFrame,
    final_metrics: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
    results_dir: Path,
) -> None:
    """Сохраняет итоговые прогнозы, метрики, график и файлы обученной модели."""
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / "final_predictions.csv"
    metrics_path = results_dir / "final_metrics.csv"
    final_predictions.to_csv(predictions_path, index=False)
    final_metrics.to_csv(metrics_path, index=False)
    LOGGER.info("Итоговые прогнозы сохранены в %s.", predictions_path)
    LOGGER.info("Итоговые метрики сохранены в %s.", metrics_path)

    save_final_prediction_plot(final_predictions, results_dir / "final_prediction_plot.png")

    if not config["saving"]["save_models"]:
        return

    model_path = results_dir / "final_model.keras"
    scaler_path = results_dir / "final_scaler.pkl"
    metadata_path = results_dir / "final_model_metadata.json"
    model.save(model_path)

    with scaler_path.open("wb") as file:
        pickle.dump(scaler, file)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    LOGGER.info("Итоговая модель сохранена в %s.", model_path)
    LOGGER.info("Итоговый scaler сохранен в %s.", scaler_path)
    LOGGER.info("Метаданные итоговой модели сохранены в %s.", metadata_path)


def save_final_prediction_plot(final_predictions: pd.DataFrame, path: Path) -> None:
    """Сохраняет график фактической и прогнозной волатильности итоговой модели."""
    if final_predictions.empty:
        raise ValueError("Нельзя построить итоговый график: таблица прогнозов пуста.")

    plot_data = final_predictions.copy()
    plot_data["target_date"] = pd.to_datetime(plot_data["target_date"])
    plot_data = plot_data.sort_values("target_date")
    save_volatility_plot(plot_data, "Итоговая модель: фактическая и прогнозная волатильность", path)
    LOGGER.info("Итоговый график сохранен в %s.", path)


def format_date(value: Any) -> str:
    """Возвращает дату в формате ISO."""
    return pd.Timestamp(value).date().isoformat()


def create_result_dirs(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Создает каталоги для результатов эксперимента."""
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


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Проверяет модели, наборы признаков и размеры окон перед запуском."""
    get_enabled_metrics(config)
    resolve_selection_metric(config)
    get_training_loss_name(config)
    models = [name.lower() for name in config["experiment"]["models"]]
    feature_sets = [name.lower() for name in config["experiment"]["feature_sets"]]
    unknown_models = sorted(set(models) - VALID_MODELS)
    unknown_feature_sets = sorted(set(feature_sets) - VALID_FEATURE_SETS)

    if unknown_models:
        raise ValueError(f"В config.json указаны неизвестные модели: {unknown_models}")

    if unknown_feature_sets:
        raise ValueError(f"В config.json указаны неизвестные наборы признаков: {unknown_feature_sets}")

    train_window = int(config["experiment"]["train_window"])
    test_window = int(config["experiment"]["test_window"])
    input_window = int(config["experiment"]["input_window"])
    step_size = int(config["experiment"]["step_size"])

    if train_window <= input_window:
        raise ValueError("Параметр experiment.train_window должен быть больше experiment.input_window, иначе обучающие последовательности не формируются.")

    if test_window < 1:
        raise ValueError("Параметр experiment.test_window должен быть положительным.")

    if step_size < 1:
        raise ValueError("Параметр experiment.step_size должен быть положительным.")

    if "train_final_model" in config["experiment"] and not isinstance(config["experiment"]["train_final_model"], bool):
        raise ValueError("Параметр experiment.train_final_model должен быть логическим значением true или false.")


def set_random_seed(seed: int) -> None:
    """Фиксирует генераторы случайных чисел Python, NumPy и TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if hasattr(keras.utils, "set_random_seed"):
        keras.utils.set_random_seed(seed)


# Точка входа

CONFIG_FILE = globals().get("CONFIG_FILE", "config.json")


def main(config_file: str | Path = CONFIG_FILE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Выполняет подготовку данных, скользящую проверку и обучение итоговой модели."""
    project_root = Path.cwd().resolve()
    config_path = project_root / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    config, project_root = load_config(config_path)
    LOGGER.info("Конфигурация загружена из %s.", config_path)
    LOGGER.info("Активные наборы признаков: %s.", ", ".join(config["experiment"]["feature_sets"]))
    LOGGER.info("Функция потерь: %s.", get_training_loss_name(config))
    LOGGER.info("Корень проекта: %s.", project_root)

    daily_dataset = build_daily_dataset(config, project_root)
    LOGGER.info("Сформирован суточный набор данных, строк: %s.", len(daily_dataset))
    predictions, metrics_by_fold, metrics_summary = run_experiments(daily_dataset, config, project_root)
    final_predictions = build_empty_final_predictions()
    final_metrics = pd.DataFrame()

    if should_train_final_model(config):
        if not config["saving"]["save_models"]:
            LOGGER.warning(
                "Итоговая модель будет обучена, но не будет сохранена, потому что saving.save_models=false."
            )

        best_model_name, best_feature_set = select_best_configuration(metrics_summary, config)
        final_predictions, final_metrics = train_final_model(
            daily_df=daily_dataset,
            config=config,
            project_root=project_root,
            model_name=best_model_name,
            feature_set=best_feature_set,
        )
    else:
        LOGGER.info("Обучение итоговой модели отключено параметром experiment.train_final_model=false.")

    return predictions, metrics_by_fold, metrics_summary, final_predictions, final_metrics


if __name__ == "__main__":
    main()


