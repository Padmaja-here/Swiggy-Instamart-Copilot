# src/customer_models.py (Part B): RFM segments with K-means, and a lapse (churn) model.
# Run from the main folder:  python -m src.customer_models

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01")
SEPTEMBER_END = pd.Timestamp("2025-10-01")
NUMERIC = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate",
           "avg_delivery", "age", "tenure_days"]
NOMINAL = ["city", "gender", "favourite_category"]

# One recommended action per segment name
ACTIONS = {
    "Premium Big Spenders": "Give early access to new electronics and a dedicated support line.",
    "Loyal Regulars": "Reward with a subscription plan or loyalty points on groceries.",
    "Occasional Buyers": "Send personalised offers on their usual categories to raise order frequency.",
    "Lapsing Customers": "Run a win-back campaign with a time-limited discount.",
}


def load_data():
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("data/clean/products.csv")
    return orders, customers, products


def rfm_table(completed_orders, as_of):
    rfm = completed_orders.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    ).reset_index()
    rfm["recency"] = (as_of - rfm["last_order"]).dt.days
    return rfm[["customer_id", "recency", "frequency", "monetary"]]


# ---------------- Tasks 1-3: segments ----------------

def name_segments(profile):
    # Premium = highest mean monetary. Of the rest: highest recency = Lapsing,
    # highest frequency = Loyal Regulars, anything left = Occasional Buyers.
    names = {}
    rest = profile.copy()
    names[rest["monetary"].idxmax()] = "Premium Big Spenders"
    rest = rest.drop(rest["monetary"].idxmax())
    names[rest["recency"].idxmax()] = "Lapsing Customers"
    rest = rest.drop(rest["recency"].idxmax())
    names[rest["frequency"].idxmax()] = "Loyal Regulars"
    rest = rest.drop(rest["frequency"].idxmax())
    for segment_id in rest.index:
        names[segment_id] = "Occasional Buyers"
    return names


def make_segments(orders, k=4):
    print("=== Task 1: RFM ===")
    completed = orders[orders["status"] == "completed"]
    rfm = rfm_table(completed, SNAPSHOT)
    print("customers:", len(rfm))
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

    print("\n=== Task 2: K-means for K = 2..8 ===")
    scaled = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    rows = []
    for n in range(2, 9):
        model = KMeans(n_clusters=n, n_init=10, random_state=42).fit(scaled)
        rows.append({"K": n, "inertia": round(model.inertia_, 1),
                     "silhouette": round(silhouette_score(scaled, model.labels_), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Task 3: profile of the chosen K =", k, "===")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    rfm["segment_id"] = kmeans.fit_predict(scaled)
    profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    ).round(2)
    names = name_segments(profile)
    profile["segment_name"] = profile.index.map(names)
    profile["action"] = profile["segment_name"].map(ACTIONS)
    print(profile.to_string())
    print("(segment_id numbers are only labels; they have no order)")

    rfm["segment_name"] = rfm["segment_id"].map(names)
    premium_id = profile["monetary"].idxmax()
    rfm["is_premium"] = rfm["segment_id"] == premium_id

    pca = PCA(n_components=2).fit(scaled)
    ratio = pca.explained_variance_ratio_
    print("PCA explained_variance_ratio_:", ratio.round(3), "| total:", round(ratio.sum(), 3))

    os.makedirs("outputs", exist_ok=True)
    rfm.to_csv("outputs/customer_segments.csv", index=False)
    print("saved outputs/customer_segments.csv")
    return rfm


# ---------------- Task 4: the lapse problem ----------------

def build_cohort(orders, customers, products):
    before = orders[orders["order_ts"] < CUTOFF]                    # nothing on or after the cutoff
    completed = before[before["status"] == "completed"]

    cohort = rfm_table(completed, CUTOFF)                            # recency, frequency, monetary
    extra = completed.groupby("customer_id").agg(
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
    ).reset_index()
    cohort = cohort.merge(extra, on="customer_id")

    # cancellation rate = cancelled / all orders with a final status (before the cutoff)
    final = before[before["status"].isin(["completed", "cancelled"])]
    rate = (final["status"] == "cancelled").groupby(final["customer_id"]).mean()
    cohort["cancellation_rate"] = cohort["customer_id"].map(rate)

    # favourite (most frequent) category; ties broken alphabetically
    with_category = completed.merge(products[["product_id", "category"]], on="product_id")
    counts = with_category.groupby(["customer_id", "category"]).size().reset_index(name="n")
    counts = counts.sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
    favourite = counts.drop_duplicates("customer_id").set_index("customer_id")["category"]
    cohort["favourite_category"] = cohort["customer_id"].map(favourite)

    cohort = cohort.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
    cohort["tenure_days"] = (CUTOFF - cohort["signup_date"]).dt.days

    # label: 1 if NO completed order from 2025-09-01 to 2025-09-30
    september = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < SEPTEMBER_END)
                       & (orders["status"] == "completed")]
    cohort["lapsed"] = (~cohort["customer_id"].isin(september["customer_id"])).astype(int)
    return cohort


def scores(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "confusion_matrix [[TN FP] [FN TP]]": confusion_matrix(y_true, y_pred).tolist(),
    }


def make_pipeline(model):
    # The preprocessing lives INSIDE the pipeline, so it is fitted on training rows only
    prep = ColumnTransformer([
        ("numbers", StandardScaler(), NUMERIC),
        ("categories", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def lapse_model(orders, customers, products, segments):
    print("\n=== Task 4: lapse cohort ===")
    cohort = build_cohort(orders, customers, products)
    X = cohort[NUMERIC + NOMINAL]
    y = cohort["lapsed"]
    print("cohort:", len(cohort), "| lapsed:", y.sum(), f"({y.mean() * 100:.2f}%)")
    assert X.isna().sum().sum() == 0, "a feature column has a missing value"
    print("missing values in features: 0")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    ids_test = cohort.loc[X_test.index, "customer_id"]
    print("train rows:", len(X_train), "| test rows:", len(X_test))

    print("\n=== Task 6a: baseline (always 'not lapsed') ===")
    baseline = scores(y_test, np.zeros(len(y_test), dtype=int))
    print(baseline)

    print("\n=== Task 6b: Logistic Regression grid search ===")
    lr_search = GridSearchCV(make_pipeline(LogisticRegression(max_iter=1000)),
                             {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
    lr_search.fit(X_train, y_train)
    print("best params:", lr_search.best_params_, "| best CV F1:", round(lr_search.best_score_, 3))

    print("\n=== Task 6c: Random Forest grid search (45 fits) ===")
    rf_search = GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
                             {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
                             cv=5, scoring="f1")
    rf_search.fit(X_train, y_train)
    print("best params:", rf_search.best_params_, "| best CV F1:", round(rf_search.best_score_, 3))

    print("\n=== Task 5: scaler learned on training rows only ===")
    scaler = lr_search.best_estimator_.named_steps["prep"].named_transformers_["numbers"]
    print("scaler mean of frequency (train rows):", round(scaler.mean_[NUMERIC.index("frequency")], 3),
          "| full-cohort mean:", round(X["frequency"].mean(), 3))

    print("\n=== Task 6d: test-set results (lapsed class) ===")
    lr_test = scores(y_test, lr_search.predict(X_test))
    rf_test = scores(y_test, rf_search.predict(X_test))
    print("Logistic Regression:", lr_test)
    print("Random Forest:      ", rf_test)
    selected = rf_search.best_estimator_
    print("selected model: Random Forest")

    print("\n=== Task 7: over-sampling the lapsed class (training rows only) ===")
    train = X_train.copy()
    train["lapsed"] = y_train
    lapsed_rows = train[train["lapsed"] == 1]
    other_rows = train[train["lapsed"] == 0]
    extra = lapsed_rows.sample(n=len(other_rows) - len(lapsed_rows), replace=True, random_state=42)
    balanced = pd.concat([train, extra])
    print("class counts after over-sampling:", balanced["lapsed"].value_counts().to_dict())
    balanced_lr = make_pipeline(LogisticRegression(max_iter=1000, C=lr_search.best_params_["model__C"]))
    balanced_lr.fit(balanced[NUMERIC + NOMINAL], balanced["lapsed"])
    print("unbalanced LR:", {k: lr_test[k] for k in ["precision", "recall", "f1"]})
    balanced_test = scores(y_test, balanced_lr.predict(X_test))
    print("balanced LR:  ", {k: balanced_test[k] for k in ["precision", "recall", "f1"]})

    print("\n=== Task 8a: what the models learned ===")
    names = selected.named_steps["prep"].get_feature_names_out()
    importance = pd.Series(selected.named_steps["model"].feature_importances_, index=names)
    print("Random Forest top 5 features:")
    print(importance.sort_values(ascending=False).head(5).round(3).to_string())
    lr_best = lr_search.best_estimator_
    coef = pd.Series(lr_best.named_steps["model"].coef_[0],
                     index=lr_best.named_steps["prep"].get_feature_names_out())
    print("Logistic Regression coefficients (positive = more likely to lapse):")
    print(coef[["numbers__recency", "numbers__frequency", "numbers__monetary"]].round(3).to_string())

    print("\n=== Task 8b: save and reload the selected model ===")
    joblib.dump(selected, "outputs/lapse_model.joblib")
    reloaded = joblib.load("outputs/lapse_model.joblib")
    same = (reloaded.predict(X_test) == selected.predict(X_test)).all()
    print("reloaded model gives identical test predictions:", same)
    assert same

    print("\n=== Task 8c: retention list (Premium customers in the test set) ===")
    probability = selected.predict_proba(X_test)[:, 1]
    test_table = pd.DataFrame({"customer_id": ids_test.values,
                               "lapse_probability": probability.round(3),
                               "actually_lapsed": y_test.values})
    premium_ids = segments.loc[segments["is_premium"], "customer_id"]
    retention = test_table[test_table["customer_id"].isin(premium_ids)]
    retention = retention.sort_values("lapse_probability", ascending=False).head(10)
    retention.to_csv("outputs/retention_list.csv", index=False)
    print(retention.to_string(index=False))
    print("truly lapsed in the list:", retention["actually_lapsed"].sum(), "of", len(retention))
    print("Note: the Premium flag uses whole-window RFM (including September), so this list is illustrative only.")


def main():
    orders, customers, products = load_data()
    segments = make_segments(orders, k=4)
    lapse_model(orders, customers, products, segments)


if __name__ == "__main__":
    main()
