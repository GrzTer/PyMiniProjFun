# ====================================================
# File: json_to_csv.py
# Author: G_T
# Version: 1.0
# Description: Reads JSON data from 'input.json' and converts it to CSV format.
#              The CSV output is written to 'output.csv'.
# ====================================================


import json
import csv


with open('2-JSON_to_CSV/input.json', 'r') as f:
    data = json.load(f)
out = ','.join([*data[0]])
for obj in data:
    out += f"\n{obj["Name"]},{obj["age"]},{obj["birthyear"]}"

with open('2-JSON_to_CSV/output.csv', 'w') as f:
    f.write(out)
#--------------------------------------------------
def json_to_csv(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        output = ','.join([*data[0]])
        for obj in data:
            output += f'\n{obj["Name"]},{obj["age"]},{obj["birthyear"]}'

        with open(output_file, 'w') as f:
            f.write(output)

    except Exception as ex:
        print(f"Error: {str(ex)}")
#--------------------------------------------------

with open('2-JSON_to_CSV/input.json', 'r') as f:
    data = json.load(f)

# Define the CSV field names (keys expected in each dictionary)
fieldnames = ["Name", "age", "birthyear"]

# Write to CSV using DictWriter
with open('2-JSON_to_CSV/output2.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for obj in data:
        writer.writerow(obj)

if __name__ == '__main__':
    json_to_csv('2-JSON_to_CSV/input.json', '2-JSON_to_CSV/output1.csv')