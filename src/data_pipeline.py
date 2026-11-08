# src/data_pipeline.py (Part A, tasks 2-5): audit the raw data, clean it, and store it.
# Run from the main folder:  python -m src.data_pipeline

import os
import sqlite3

import numpy as np
import pandas as pd

RAW = "data/raw/"
CLEAN = "data/clean/"
DB_PATH = "data/instamart.db"


def is_blank(column):
    # True where a value is empty text (or NaN)
    return column.isna() | (column.astype(str).str.strip() == "")


# ---------------- Task 2: audit ----------------

def describe_table(name, df):
    print("\n=== " + name + " ===")
    print("shape:", df.shape)
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": [is_blank(df[col]).sum() for col in df.columns],
    })
    report["missing_pct"] = (report["missing_count"] / len(df) * 100).round(2)
    print(report)


def audit():
    # keep_default_na=False keeps empty cells as "" so we can count them ourselves
    customers = pd.read_csv(RAW + "customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW + "products.csv", keep_default_na=False)
    orders = pd.read_csv(RAW + "orders.csv", keep_default_na=False)

    describe_table("customers", customers)
    describe_table("products", products)
    describe_table("orders", orders)

    age_numbers = pd.to_numeric(customers["age"], errors="coerce")
    delivery_numbers = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    figures = {
        "raw order rows": len(orders),
        "exact duplicate rows": orders.duplicated().sum(),
        "rows with status pending": (orders["status"] == "pending").sum(),
        "distinct order_id": orders["order_id"].nunique(),
        "order_amount not convertible to a number": pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum(),
        "negative delivery_minutes": (delivery_numbers < 0).sum(),
        "blank delivery_minutes": is_blank(orders["delivery_minutes"]).sum(),
        "blank age": is_blank(customers["age"]).sum(),
        "age outside 10-100": ((age_numbers < 10) | (age_numbers > 100)).sum(),
        "distinct raw city spellings": customers["city"].nunique(),
        "distinct non-blank raw gender spellings": customers.loc[~is_blank(customers["gender"]), "gender"].nunique(),
        "blank gender": is_blank(customers["gender"]).sum(),
    }
    print("\n=== Audit figures ===")
    for name, value in figures.items():
        print(f"{name}: {value}")
    return figures


# ---------------- Task 3: clean customers ----------------

def clean_customers():
    customers = pd.read_csv(RAW + "customers.csv", keep_default_na=False)
    print("\n=== Cleaning customers ===")

    # (a) city: remove spaces, Title Case
    before = customers["city"].nunique()
    customers["city"] = customers["city"].str.strip().str.title()
    print("city spellings:", before, "->", customers["city"].nunique())

    # (b) gender: every spelling -> male / female / unknown
    gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
    customers["gender"] = customers["gender"].str.strip().str.lower().map(gender_map).fillna("unknown")
    print("gender counts:", customers["gender"].value_counts().to_dict())

    # (c) age: blank or outside 10-100 -> missing, then fill with the median age of the same city
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan
    print("ages set to missing (blank + impossible):", customers["age"].isna().sum())
    city_median_age = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_median_age)

    # (d) signup_date -> datetime
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    print("city counts:", customers["city"].value_counts().to_dict())
    print("missing ages after:", customers["age"].isna().sum(),
          "| age range:", customers["age"].min(), "-", customers["age"].max())
    return customers


# ---------------- Task 4: clean orders ----------------

def clean_orders(products):
    orders = pd.read_csv(RAW + "orders.csv", keep_default_na=False)
    print("\n=== Cleaning orders ===")

    # 1. order_ts -> datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # 2. remove exact duplicate rows
    before = len(orders)
    orders = orders.drop_duplicates()
    print("rule 2 - exact duplicates removed:", before - len(orders))

    # 3. same order_id more than once (pending + final): keep the row with the latest order_ts
    before = len(orders)
    orders = orders.sort_values(["order_id", "order_ts"])
    orders = orders.drop_duplicates(subset="order_id", keep="last")
    print("rule 3 - older records of the same order removed:", before - len(orders))

    # 4. order_amount -> number; unparseable -> missing; refill from the product price
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    missing_amount = orders["order_amount"].isna()
    price = orders["product_id"].map(products.set_index("product_id")["price_inr"])
    rebuilt = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
    orders.loc[missing_amount, "order_amount"] = rebuilt[missing_amount]
    print("rule 4 - amounts rebuilt from price:", missing_amount.sum())

    # 5. negative delivery -> missing; completed orders: fill with the median of their category
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative = orders["delivery_minutes"] < 0
    orders.loc[negative, "delivery_minutes"] = np.nan
    print("rule 5 - negative delivery times set to missing:", negative.sum())

    orders["category"] = orders["product_id"].map(products.set_index("product_id")["category"])
    is_completed = orders["status"] == "completed"
    completed_delivery = orders["delivery_minutes"].where(is_completed)   # cancelled -> NaN, ignored
    category_median = completed_delivery.groupby(orders["category"]).transform("median")
    to_fill = is_completed & orders["delivery_minutes"].isna()
    orders.loc[to_fill, "delivery_minutes"] = category_median[to_fill]
    print("rule 5 - completed orders given their category's median delivery time:", to_fill.sum())
    orders = orders.drop(columns=["category"])
    # cancelled orders keep a missing delivery time: they were never delivered

    print("rows:", len(orders), "| status:", orders["status"].value_counts().to_dict())
    print("sum of completed order_amount: INR", round(orders.loc[is_completed, "order_amount"].sum(), 2))
    return orders.reset_index(drop=True)


# ---------------- Task 5: store ----------------

def store(customers, products, orders):
    os.makedirs(CLEAN, exist_ok=True)
    # store dates as text in a fixed format
    customers = customers.copy()
    orders = orders.copy()
    customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    customers.to_csv(CLEAN + "customers.csv", index=False)
    products.to_csv(CLEAN + "products.csv", index=False)
    orders.to_csv(CLEAN + "orders.csv", index=False)

    connection = sqlite3.connect(DB_PATH)
    customers.to_sql("customers", connection, if_exists="replace", index=False)
    products.to_sql("products", connection, if_exists="replace", index=False)
    orders.to_sql("orders", connection, if_exists="replace", index=False)
    connection.close()
    print("\nSaved data/clean/*.csv and", DB_PATH)


def main():
    audit()
    products = pd.read_csv(RAW + "products.csv")
    customers = clean_customers()
    orders = clean_orders(products)
    store(customers, products, orders)


if __name__ == "__main__":
    main()
