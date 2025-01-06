import json

file_path = '../notebooks/ML-Halpha-wise_unique.ipynb'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        json.load(file)
    print("The notebook file is a valid JSON.")
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
