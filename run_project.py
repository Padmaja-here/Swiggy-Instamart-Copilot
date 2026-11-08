# run_project.py: runs the whole project in order with one command.
#   python run_project.py
# (The README's Reproduce section lists the same steps one by one.)

import subprocess
import sys

STEPS = [
    ["generate_data.py"],              # supplied file: creates data/raw and data/policies
    ["-m", "src.data_pipeline"],       # Part A: audit, clean, store
    ["-m", "src.validate"],            # Part A: prints ALL CHECKPOINTS PASSED
    ["-m", "src.sql_analysis"],        # Part A: the seven SQL queries
    ["-m", "src.eda"],                 # Part A: pandas cross-check and statistics
    ["-m", "src.customer_models"],     # Part B: segments and lapse model
    ["-m", "src.triage"],              # Part C: ticket triage
    ["-m", "src.policy_rag"],          # Part D: policy RAG
    ["-m", "src.refund_agent"],        # Part D: refund agent
]

for step in STEPS:
    print("\n" + "=" * 70)
    print("RUNNING: python", " ".join(step))
    print("=" * 70, flush=True)
    # sys.executable is the Python of the active virtual environment
    subprocess.run([sys.executable] + step, check=True)

print("\nAll parts finished.")
