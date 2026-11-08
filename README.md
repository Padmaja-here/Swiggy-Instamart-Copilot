# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a helper tool ("copilot") for a grocery delivery app. It has four parts, and each part uses what the previous part made:

- **Part A** cleans messy order data and saves it in a database.
- **Part B** groups customers and predicts who might stop ordering.
- **Part C** reads support tickets, hides private details and sends each ticket to the right team.
- **Part D** answers policy questions and handles refund requests with an AI agent.

All data is fake, made by the supplied `generate_data.py`. The "AI" is the supplied `mock_llm.py`, an offline stand-in for a real language model, so no API key is needed. Both files are used unchanged. All money is in INR.

| File | What it does |
| --- | --- |
| `generate_data.py` | Creates the fake data (supplied) |
| `mock_llm.py` | Fake language model (supplied) |
| `src/data_pipeline.py`, `src/validate.py`, `src/sql_analysis.py`, `src/eda.py`, `sql/queries.sql` | Part A |
| `src/customer_models.py` | Part B |
| `src/triage.py` | Part C |
| `src/policy_rag.py`, `src/refund_agent.py` | Part D |
| `run_project.py` | Runs everything in order |

## Reproduce

Run these commands one by one from the main project folder:

1. `python -m venv .venv`
2. `.venv\Scripts\Activate.ps1` (on macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python -m src.data_pipeline`
6. `python -m src.validate`
7. `python -m src.sql_analysis`
8. `python -m src.eda`
9. `python -m src.customer_models`
10. `python -m src.triage`
11. `python -m src.policy_rag`
12. `python -m src.refund_agent`

Shortcut: after step 3, `python run_project.py` runs steps 4 to 12 together.

## Part A

**Concept:** raw data is messy, with duplicates, different spellings and impossible values. If we use it as it is, every later result will be wrong. So we first check the data (audit), fix it with clear rules (clean), save it (store), and then ask business questions with SQL and pandas.

**How it runs:**
1. `data_pipeline.py` counts the problems, fixes them one rule at a time, and saves clean CSV files and the SQLite database `data/instamart.db`.
2. `validate.py` checks the clean data against expected numbers and prints `ALL CHECKPOINTS PASSED`.
3. `sql_analysis.py` runs the seven queries in `sql/queries.sql`.
4. `eda.py` repeats one query in pandas to cross-check it and calculates simple statistics.

**Cleaning rules and how many values each changed:**
- City: removed spaces and fixed capital letters, so 24 spellings became 6 cities (476 values).
- Gender: every spelling became male, female or unknown (950 values).
- Age: ages outside 10–100 were set as missing, then all missing ages were filled with the city's median age (67 ages).
- Dates: converted text to real dates.
- Orders: removed 119 exact duplicates, and kept only the latest record of each order (656 older "pending" rows removed).
- Amounts: 112 unreadable amounts (like "N/A") were rebuilt as price × quantity × (1 − discount).
- Delivery time: 105 negative times were set to missing, and 774 missing times of completed orders were filled with their category's median. Cancelled orders stay empty because they were never delivered.

**Results:**
- Audit: 13,084 raw order rows; 119 exact duplicates; 656 pending rows; 12,309 distinct orders; 120 unreadable amounts; 106 negative and 2,541 blank delivery times; 61 blank ages; 6 ages outside 10–100; 24 city spellings; 10 gender spellings; 30 blank genders. Missing values: age 5.08%, gender 2.50%, delivery_minutes 19.42%, all other columns 0%. Raw shapes: customers 1,200 × 5, products 40 × 4, orders 13,084 × 9.
- After cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; all ages 18–60; 12,309 orders (11,110 completed, 1,199 cancelled); total completed revenue INR 2,998,668.80.
- SQL: Q1 30 customers; Q2 30 rows, total INR 2,998,668.80, largest Bengaluru / Electronics INR 233,909.45; Q3 11 products; Q4 25 customers; Q5 550 customers above the average of INR 2,552.06; Q6 top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10), with 8 rows where RANK() ≠ ROW_NUMBER(), because tied customers get the same RANK() but always a different ROW_NUMBER(); Q7 183 days, final total INR 2,998,668.80.
- pandas: the pivot-table total equals the SQL total. Per city (sum / mean / count): Bengaluru 523,643.40 / 276.47 / 1,894; Chennai 463,629.15 / 272.72 / 1,700; Delhi 502,829.80 / 243.62 / 2,064; Hyderabad 528,589.90 / 284.95 / 1,855; Mumbai 479,058.50 / 297.00 / 1,613; Pune 500,918.05 / 252.48 / 1,984.
- Statistics of completed amounts: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences −283.50 and 645.30, outliers 0 below and 897 above.

**What the data tells us:** it is six months of grocery orders from 1,175 active customers in six cities. Amounts are right-skewed: most orders are cheap, and a few expensive electronics orders pull the mean far above the median. The 897 outliers are real electronics purchases, so I kept them. All correlations are very weak; the largest is amount vs discount (−0.095), and a correlation this small does not prove causation.

## Part B

**Concept:** customer segmentation and lapse prediction.
- **Segmentation (unsupervised learning):** each customer is described by RFM: Recency (days since the last order), Frequency (number of orders) and Monetary (total spend). K-means then groups similar customers without being told the answer.
- **Lapse prediction (supervised learning):** a model learns from past customers who did or did not stop ordering, then predicts it for new ones.

**How it runs:** `customer_models.py`
1. Builds the RFM table, scales it, and tries K-means with K = 2 to 8.
2. Picks K = 4, names each group, and saves `outputs/customer_segments.csv`.
3. Builds features from orders **before 2025-09-01** and labels each customer "lapsed" if they had no order in September. Using only earlier data avoids **leakage**, meaning the model never sees the answer.
4. Splits customers into training (80%) and test (20%) sets, trains Logistic Regression and Random Forest with GridSearchCV, and compares them on the test set.
5. Saves the best model and a retention list of Premium customers at risk.

**Results:**
- RFM: 1,175 customers; recency median 19 (0–180); frequency mean 9.46 (max 33); monetary mean 2,552.06 (max 14,766.55).
- K-means (inertia / silhouette): K2 1820.8 / 0.465, K3 1268.2 / 0.445, **K4 743.1 / 0.526**, K5 617.4 / 0.480, K6 503.2 / 0.482, K7 408.5 / 0.416, K8 366.5 / 0.412. I chose K = 4 because the elbow is there, its silhouette score is the best, and four groups are easy to act on. Inertia always goes down as K goes up, so it can't choose K by itself.
- Segments (cluster numbers are only labels):

| Segment | Size | Recency | Frequency | Spend (INR) | Action |
| --- | --- | --- | --- | --- | --- |
| Premium Big Spenders (Premium) | 138 | 24.1 | 6.8 | 7,129.70 | Early access to electronics |
| Loyal Regulars | 419 | 8.8 | 19.4 | 3,410.02 | Loyalty points |
| Occasional Buyers | 436 | 28.9 | 3.8 | 1,057.62 | Personal offers |
| Lapsing Customers | 182 | 109.0 | 2.2 | 685.99 | Win-back discount |

- PCA explained variance: 0.634 + 0.218 = 0.852.
- Lapse data: 1,156 customers, 443 lapsed (38.32%), split into 924 training and 232 test. 11 features, no missing values. No data from 2025-09-01 onwards and no segment labels were used as features, and no encoder, scaler or resampler saw test rows (all of them sit inside a Pipeline). The scaler's frequency mean is 8.115 versus the cohort's 8.029, because it only saw training rows.
- Test results for the lapsed class (precision / recall / F1): baseline 0 / 0 / 0 (accuracy 61.6%); Logistic Regression (C = 100, CV F1 0.651) 0.644 / 0.652 / 0.648, confusion [[111, 32], [31, 58]]; **Random Forest** (depth 8, leaf 10, CV F1 0.681) 0.659 / 0.652 / **0.655**, confusion [[113, 30], [31, 58]].
- I chose the Random Forest. It catches the same leavers as logistic regression with fewer false alarms. Missing a customer who leaves costs more than sending one extra coupon to a loyal customer.
- Balancing the training data (570 per class) raised recall from 0.652 to 0.854, while precision fell from 0.644 to 0.589. So we catch more leavers but send more extra offers.
- Most important forest features: frequency 0.280, monetary 0.271, recency 0.115, average delivery 0.061, cancellation rate 0.052. Logistic coefficients: frequency −1.698 (frequent buyers stay), monetary +0.285 (big one-time spenders leave more), recency +0.040 (the longer since the last order, the more likely to leave).
- The model is saved to `outputs/lapse_model.joblib` and gives identical predictions after reloading. 1 of the 10 customers in `outputs/retention_list.csv` actually lapsed. This list is only illustrative, because the Premium label uses data from the whole period.

## Part C

**Concept:** sorting support tickets with an AI model, safely. Before anything goes to the model we block empty or very long tickets and hide private details (emails, phone numbers, order IDs). After the model answers, we check the answer, retry when the model fails, and send anything unclear to a human.

**How it runs:** `triage.py` processes each ticket in this order:
1. **Guardrails:** skip empty tickets and flag very long ones.
2. **Redaction:** replace private details with `[EMAIL]`, `[PHONE]` and `[ORDER_ID]`.
3. **Ask the model:** send the ticket with a prompt asking for a JSON answer. Zero-shot means only instructions; few-shot adds examples.
4. **Retry:** if the model gives a temporary error, wait a little and try again.
5. **Validate:** accept only the four allowed categories; anything else goes to human review.
6. **Route:** fraud → security team, Premium customer → premium desk, otherwise → support queue.
7. Save `outputs/triage_results.csv`, plus `outputs/triage_budget_demo.csv`, which shows what happens when the token budget runs out.

**Results:**
- 4 empty and 4 long tickets (734 tokens each) were never sent to the model.
- Redactions: 55 emails, 91 phones, 132 order IDs (278 in total), with 0 left afterwards.
- 19 temporary errors, all fixed by one retry.
- Answers: 135 valid JSON, 12 read from plain text, 9 invalid (sent to humans).
- Zero-shot: 85.7% accuracy, 9,712 tokens (66.1 per ticket). Its mistakes were 11 fraud → billing and 10 technical → billing. Few-shot: 100% accuracy, 19,072 tokens (129.7 per ticket). Few-shot costs twice the tokens but stops fraud cases going to the wrong team, so it is worth it.
- Budget demo (zero-shot, 6,000 tokens): 59 tickets stopped as `budget_exhausted`.
- Final routes (164 tickets): support queue 102, security team 34, human review 17, premium desk 11. The 147 classified tickets were checked by rules; 17 went to humans (4 empty, 4 long, 9 invalid).

## Part D

**Concept:** policy answers with RAG, and a refund agent.
- **RAG (Retrieval-Augmented Generation):** first *find* the policy text that matches the question, then let the model *answer only from that text*. This stops it from making things up.
- **Agent:** an AI that works in a loop: decide the next step, use a tool (a Python function), look at the result, repeat. It has safety limits, and it hands over to a human when needed.

**How it runs:**
1. `policy_rag.py` cuts the 5 policy files into 40-word pieces (chunks) that overlap by 10 words, turns them into numbers with TF-IDF, finds the 3 closest chunks to each question, and asks the model to answer from them. It tests itself on 14 questions.
2. `refund_agent.py` runs 7 refund requests. For each one, the model picks a tool: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` or `escalate_to_human`. Refunds above INR 1,000 go to a human. Every run is logged, and a checkpoint is saved after each tool so a crashed run can be resumed.

**Results:**
- 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2). For "refund policy late delivery", all top 3 chunks come from the refund policy.
- TF counts how often a word appears in a chunk, and IDF gives rare words more weight. Its weakness is that it only matches exact words, so different wording can miss the right chunk.
- Evaluation with overlap 10: right document 12/12, right fact 12/12, "I could not find" for both off-topic questions 2/2. With no overlap, the right fact drops to 10/12: "What delivery time makes a completed order eligible for a full refund?" breaks because its sentence is cut between two chunks.

| Scenario | Result | Refund (INR) | Model calls | Tokens |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 947 |
| S2 | REFUND | 135.00 | 5 | 947 |
| S3 | ESCALATE | 1,999.00 | 5 | 934 |
| S4 | DENIED | 0 | 4 | 645 |
| S5 | DENIED | 0 | 3 | 367 |
| S6 | NOT_FOUND | 0 | 2 | 92 |
| S7 | FALLBACK_MAX_ITERATIONS | 0 | 6 | 3,060 |

- Safety demos: S1 with a 200-token limit → FALLBACK_TOKEN_LIMIT; a 0-second time limit → FALLBACK_TIMEOUT.
- Crash and resume (S1): tools executed were get_order, search_policy, calculate_refund, issue_refund, ending in REFUND INR 128.25.
- **ReAct vs plan-and-execute:** my agent (ReAct) asks the model after every step, using 5 calls for S1, 3 for S5 and 2 for S6. Plan-and-execute would ask once for a full plan, so it is cheaper when everything goes as expected. But if the order is not found (S6) or cancelled (S5), a fixed plan could keep going wrongly unless it replans, while ReAct reacts to each result. For refunds I would choose ReAct, because many requests take an unexpected turn and correct money decisions matter more than saving tokens.

## Limitations and Next Steps

- The data is fake and simple; real data would be messier, and few-shot reaches 100% partly because tickets come from only 12 templates.
- `mock_llm.py` follows fixed rules; it is not a real AI model.
- The Premium label uses data from the whole period (including September), so the retention list is illustrative and not a fair, leakage-free test.
- The lapse model (F1 0.655) could improve with more features.
- Simple text patterns may miss unusual private details, and TF-IDF only matches exact words.
- No real money moves; refunds are only written to a log file.

### What I learned

This was my first AI/ML project. Here is what I learned from each part:

- **Clean data comes first.** Real data has many small problems, like the same city written five ways, missing ages, or the same order saved twice. If I don't fix these first, every later result is wrong. I also learned to write down every cleaning rule and count how many rows it changed.
- **SQL and pandas can answer the same question.** Running a query both ways and getting the same total gave me confidence my work was correct.
- **Machine learning can find groups by itself.** K-means grouped customers without being told the answer, and the groups made business sense (Premium, Loyal, Occasional, Lapsing).
- **A model must never see the answer while learning.** I only used data from before September to predict September. Otherwise the model looks perfect in testing but fails in real life.
- **Accuracy is not enough.** A lazy model that always says "won't leave" got 61.6% accuracy but caught nobody. Precision, recall and F1 told me much more.
- **AI needs safety checks.** Before calling the model, I blocked empty or very long tickets and hid emails and phone numbers. After the model answered, I checked the answer and sent unclear ones to a human.
- **RAG stops the AI from making things up.** First find the right policy text, then answer only from it. If nothing matches, the assistant should say "I could not find it" instead of guessing.
- **An agent is a loop with limits.** The agent decides a step, uses a tool, checks the result and repeats. Limits on steps, tokens and time, plus a human fallback, keep it safe.
- **Good habits matter.** Virtual environments, small functions, testing with `assert`, fixed random seeds and Git made my project easy to run again and share.

I still have a lot to learn, especially about real language models and improving model accuracy, but this project gave me a strong base to build on.