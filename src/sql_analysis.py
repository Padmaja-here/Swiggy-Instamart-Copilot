# src/sql_analysis.py (Part A, task 6): run the seven queries in sql/queries.sql and report the results.
# Run:  python -m src.sql_analysis

import sqlite3

import pandas as pd


def load_queries():
    # Split the file at each ";" and keep the pieces that contain a query
    text = open("sql/queries.sql", encoding="utf-8").read()
    pieces = text.split(";")
    return [piece.strip() for piece in pieces if "SELECT" in piece]


def main():
    queries = load_queries()
    connection = sqlite3.connect("data/instamart.db")
    results = [pd.read_sql_query(query, connection) for query in queries]
    connection.close()
    q1, q2, q3, q4, q5, q6, q7 = results

    print("Q1 customers with completed orders > 500 adding up to > 8000:", len(q1))

    print("Q2 rows:", len(q2), "| total revenue: INR", round(q2["revenue"].sum(), 2))
    biggest = q2.loc[q2["revenue"].idxmax()]
    print("   largest cell:", biggest["city"], "/", biggest["category"], "INR", biggest["revenue"])

    print("Q3 products above the average price:", len(q3))
    print("Q4 customers with no completed order:", len(q4))
    print("Q5 customers above the average spend:", len(q5), "| average: INR", q5["avg_spend"].iloc[0])

    top = q6[q6["row_number_in_city"] == 1][["city", "customer_id", "total_spend"]]
    print("Q6 top customer per city:")
    print(top.to_string(index=False))
    print("   rows where RANK() differs from ROW_NUMBER():", (q6["rank_in_city"] != q6["row_number_in_city"]).sum())

    print("Q7 daily rows:", len(q7), "| last cumulative value: INR", q7["cumulative_revenue"].iloc[-1])


if __name__ == "__main__":
    main()
