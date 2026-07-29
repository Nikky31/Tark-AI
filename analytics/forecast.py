# analytics/forecast.py
# -----------------------------------------------------------------------------
# LAYER 13 : FORECASTING
#
# In simple words: we look at the past values of a number (say monthly sales)
# and guess the next few values. We use a very simple ML model - "Linear
# Regression" from Spark MLlib - which just draws the best straight line through
# the past points and extends it into the future.
#
# We also report how good the model is:
#   RMSE : average error size          (lower is better)
#   MAE  : average absolute error       (lower is better)
#   R2   : how well the line fits, 0..1 (higher is better)
#   MAPE : average error in percent     (lower is better)
#
# forecast_metric() is the only function app.py calls, so its name, inputs and
# outputs stay exactly the same. The work is split into small one-job helpers
# so the code is easy to follow.
# -----------------------------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from config.config import HIVE_DATABASE, HIVE_METASTORE_URI


# -----------------------------------------------------------------------------
# Helper 1 : start Spark
# -----------------------------------------------------------------------------
def create_spark_session():
    """Start (or reuse) a Hive-enabled Spark session - the number-crunching engine."""
    spark = (
        SparkSession.builder
        .appName("TarkAI-Forecast")
        .config("hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )
    return spark


# -----------------------------------------------------------------------------
# Helper 2 : read the past data from Hive
# -----------------------------------------------------------------------------
def load_history(spark, table, dimension_column, metric_column):
    """Read the historical points: SUM(metric) per dimension value, time-ordered.

    Returns a list of Spark Rows with d = the period label and y = the summed value.
    """
    spark.sql(f"USE {HIVE_DATABASE}")

    query = (
        f"SELECT {dimension_column} AS d, SUM({metric_column}) AS y "
        f"FROM {table} "
        f"GROUP BY {dimension_column} "
        f"ORDER BY {dimension_column} ASC"
    )
    grouped_data = spark.sql(query)
    return grouped_data.collect()


# -----------------------------------------------------------------------------
# Helper 3 : train the linear regression model
# -----------------------------------------------------------------------------
def train_model(spark, history_rows):
    """Train Linear Regression on the past points.

    The period might be text, so we replace each one with its position in time
    (0, 1, 2, ...) and let the model learn: value = slope * position + intercept.
    Returns (model, assembler, training_data, numbered_history), where
    numbered_history is a list of (position, value, period_label) tuples.
    """
    # turn every past point into (position, value, label)
    numbered_history = [
        (float(position), float(row["y"]), str(row["d"]))
        for position, row in enumerate(history_rows)
    ]

    training_data = spark.createDataFrame(
        numbered_history, ["position", "value", "period_label"])

    # Spark ML needs all inputs packed into a single "features" column
    assembler = VectorAssembler(inputCols=["position"], outputCol="features")
    training_data = assembler.transform(training_data)

    # fit the straight line through the points
    linear_regression = LinearRegression(featuresCol="features", labelCol="value")
    model = linear_regression.fit(training_data)

    print(f"[Forecast] Model trained on {len(numbered_history)} data points")
    return model, assembler, training_data, numbered_history


# -----------------------------------------------------------------------------
# Helper 4 : measure how good the model is
# -----------------------------------------------------------------------------
def evaluate_model(model, training_data, training_points):
    """Score the model by predicting the same past points and comparing to reality.

    Returns a dict with RMSE, MAE, R2, MAPE, plus the line's intercept,
    coefficients and how many points were used (same keys app.py expects).
    """
    predictions = model.transform(training_data)

    # Spark gives us RMSE, MAE and R2 with a ready-made evaluator
    def score(metric_name):
        evaluator = RegressionEvaluator(
            labelCol="value", predictionCol="prediction", metricName=metric_name)
        return evaluator.evaluate(predictions)

    rmse = round(score("rmse"), 2)
    mae = round(score("mae"), 2)
    r2 = round(score("r2"), 4)

    # MAPE = average of |actual - predicted| / |actual|, shown as a percentage.
    # We compute it ourselves so beginners can see exactly how it works.
    actual_vs_predicted = predictions.select("value", "prediction").collect()
    percentage_errors = []
    for point in actual_vs_predicted:
        actual = point["value"]
        predicted = point["prediction"]
        if actual != 0:                      # avoid dividing by zero
            percentage_errors.append(abs(actual - predicted) / abs(actual))
    if percentage_errors:
        mape = round(sum(percentage_errors) / len(percentage_errors) * 100, 2)
    else:
        mape = None

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "intercept": round(model.intercept, 2),
        "coefficients": [round(c, 4) for c in model.coefficients.toArray().tolist()],
        "training_points": training_points,
    }
    print(f"[Forecast] RMSE={rmse}, MAE={mae}, R2={r2}, MAPE={mape}")
    return metrics


# -----------------------------------------------------------------------------
# Helper 5 : predict the future
# -----------------------------------------------------------------------------
def predict_future(spark, model, assembler, history_length, periods):
    """Predict the next `periods` values.

    Trained on positions 0..(n-1), so future positions are n, n+1, ... We ask
    the model to predict them and clamp negatives to 0 (sales cannot be < 0).
    Returns a list of {"period": "t+1", "value": ...} dicts.
    """
    future_positions = [(float(history_length + i),) for i in range(periods)]
    future_data = spark.createDataFrame(future_positions, ["position"])
    future_data = assembler.transform(future_data)

    predicted_rows = model.transform(future_data).collect()

    forecast = []
    for i, row in enumerate(predicted_rows):
        predicted_value = max(row["prediction"], 0)      # no negative values
        forecast.append({
            "period": f"t+{i + 1}",
            "value": round(predicted_value, 2),
        })
    return forecast


# -----------------------------------------------------------------------------
# Main function used by app.py
# -----------------------------------------------------------------------------
def forecast_metric(table, dim, metric, periods=3):
    """Forecast a numeric metric and report how good the model is.

    The only function app.py calls. It runs the whole pipeline in order:
    start Spark -> load past data -> train -> evaluate -> predict future.
    Returns (history, forecast, metrics). With fewer than 2 points there is
    nothing to fit a line to, so it safely returns ([], [], {}).
    """
    spark = create_spark_session()

    # step 1 & 2 : read the historical points
    history_rows = load_history(spark, table, dim, metric)

    # we need at least 2 points to draw a line through them
    if len(history_rows) < 2:
        print("[Forecast] Not enough data points for regression")
        spark.stop()
        return [], [], {}

    # step 3 : train the model
    model, assembler, training_data, numbered_history = train_model(spark, history_rows)

    # step 4 : measure accuracy
    metrics = evaluate_model(model, training_data, len(numbered_history))

    # build the history list for the chart (period label + real value)
    history = [
        {"period": period_label, "value": round(value, 2)}
        for _, value, period_label in numbered_history
    ]

    # step 5 : predict the future
    forecast = predict_future(spark, model, assembler, len(numbered_history), periods)

    spark.stop()
    return history, forecast, metrics


# TODO: for seasonal data, a polynomial regression or ARIMA model could give
#       better accuracy than a straight line. Keeping it simple for now.


# quick manual test - run this file directly to check it works
if __name__ == "__main__":
    history, forecast, metrics = forecast_metric(
        "sales", "order_id", "sales_amount", periods=3)
    print("History :", history)
    print("Forecast:", forecast)
    print("Metrics :", metrics)
