#!/usr/bin/env python
import csv

# Find the question with ID 25636
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_SCRIPT_DIR, "..", "data", "metaculus_conditionals_2026-03.csv")

with open(_CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('child_post_id') == '25636' or row.get('condition_post_id') == '25636':
            print("Found question:")
            print(f"  conditional_post_id: {row.get('conditional_post_id')}")
            print(f"  condition_post_id: {row.get('condition_post_id')}")
            print(f"  condition_title: {row.get('condition_title')[:80]}")
            print(f"  condition_description length: {len(row.get('condition_description', ''))}")
            print(f"  condition_description: {row.get('condition_description')[:100]}")
            print(f"  child_post_id: {row.get('child_post_id')}")
            print(f"  child_title: {row.get('child_title')[:80]}")
            print(f"  child_description length: {len(row.get('child_description', ''))}")
            print(f"  child_description: {row.get('child_description')[:100]}")
            print(f"  child_resolution_criteria length: {len(row.get('child_resolution_criteria', ''))}")
            break
