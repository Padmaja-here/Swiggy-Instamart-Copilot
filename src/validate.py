# src/validate.py (Part A, task 5): check the clean data against the brief's reference checkpoints.
# Run:  python -m src.validate     -> prints ALL CHECKPOINTS PASSED if every check holds

import sqlite3

import pandas as pd


def main():
    customers = pd.read_csv("data/clean/customers.csv")
    orders = pd.read_csv("data/clean/orders.csv")
    completed = orders[orders["status"] == "completed"]
    cancelled = orders[orders["status"] == "cancelled"]

    # Customers (task 3)
    assert customers["city"].value_counts().to_dict() == {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
    assert customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
    assert customers["age"].isna().sum() == 0
    assert customers["age"].between(18, 60).all()

    # Orders (task 4)
    assert len(orders) == 12309
    assert orders["order_id"].is_unique
    assert len(completed) == 11110
    assert len(cancelled) == 1199
    assert (orders["status"] == "pending").sum() == 0
    assert orders["order_amount"].isna().sum() == 0
    assert completed["delivery_minutes"].isna().sum() == 0
    assert cancelled["delivery_minutes"].isna().sum() == 1199
    assert round(completed["order_amount"].sum(), 2) == 2998668.80

    # Database (task 5)
    connection = sqlite3.connect("data/instamart.db")
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    first_ts = connection.execute("SELECT order_ts FROM orders LIMIT 1").fetchone()[0]
    connection.close()
    assert {"customers", "products", "orders"} <= set(tables)
    assert len(first_ts) == 19   # stored as text: YYYY-MM-DD HH:MM:SS

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
