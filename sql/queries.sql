-- sql/queries.sql (Part A, task 6). Only completed orders count as revenue or spend.

-- Q1: WHERE then HAVING. Customers whose completed orders above INR 500 add up to more than INR 8000.
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_over_500
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_over_500 DESC;

-- Q2: three-table join. Revenue by city and product category.
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: scalar subquery. Products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN. Customers who have no completed order.
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: common table expressions. Customers whose total completed spend is above the average.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_spend
)
SELECT cs.customer_id, ROUND(cs.total_spend, 2) AS total_spend, ROUND(a.avg_spend, 2) AS avg_spend
FROM customer_spend cs, average_spend a
WHERE cs.total_spend > a.avg_spend
ORDER BY cs.total_spend DESC;

-- Q6: ranking window functions within each city.
WITH spend AS (
    SELECT c.city, o.customer_id, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city, customer_id, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_in_city,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_in_city,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_number_in_city
FROM spend
ORDER BY city, row_number_in_city;

-- Q7: running total. Daily completed revenue with a cumulative column.
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
