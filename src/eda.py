# src/eda.py (Part A, task 7): pandas cross-check of Q2 and exploratory statistics.
# Run:  python -m src.eda

import sqlite3

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def main():
    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")
    completed = orders[orders["status"] == "completed"]
    table = completed.merge(customers, on="customer_id").merge(products, on="product_id")

    # (a) Q2 rebuilt with a pivot table; margins=True adds an "All" row and column with totals
    pivot = pd.pivot_table(table, values="order_amount", index="city", columns="category",
                           aggfunc="sum", margins=True)
    pandas_total = round(pivot.loc["All", "All"], 2)
    connection = sqlite3.connect("data/instamart.db")
    sql_total = connection.execute(
        "SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed'").fetchone()[0]
    connection.close()
    print("Pivot table (INR):")
    print(pivot.round(2))
    print("pandas grand total:", pandas_total, "| SQL total:", sql_total, "| equal:", pandas_total == sql_total)

    # Named aggregation per city
    per_city = table.groupby("city").agg(
        total=("order_amount", "sum"),
        mean=("order_amount", "mean"),
        count=("order_amount", "count"),
    ).round(2)
    print("\nPer city:")
    print(per_city)

    # (b) Statistics of completed order_amount
    amount = completed["order_amount"]
    q1 = amount.quantile(0.25)
    q3 = amount.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    print("\nCompleted order_amount statistics:")
    print("mean:", round(amount.mean(), 2), "| median:", round(amount.median(), 2),
          "| skewness:", round(amount.skew(), 2))
    print("Q1:", round(q1, 2), "| Q3:", round(q3, 2), "| IQR:", round(iqr, 2))
    print("lower fence:", round(lower_fence, 2), "| outliers below:", (amount < lower_fence).sum())
    print("upper fence:", round(upper_fence, 2), "| outliers above:", (amount > upper_fence).sum())

    # Correlation matrix of four numeric columns
    columns = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = completed[columns].corr().round(3)
    print("\nCorrelation matrix:")
    print(corr)

    # Find the pair with the largest correlation in absolute value (ignoring the diagonal)
    best_pair = None
    best_value = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr.iloc[i, j]
            if abs(value) > abs(best_value):
                best_pair = (columns[i], columns[j])
                best_value = value
    print("largest absolute correlation:", best_pair, best_value)


if __name__ == "__main__":
    main()
