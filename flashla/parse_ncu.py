#!/usr/bin/env python3
import subprocess, csv, sys, io

rep_file = sys.argv[1]
kernel_id = sys.argv[2] if len(sys.argv) > 2 else "0"
filter_pattern = sys.argv[3] if len(sys.argv) > 3 else None

result = subprocess.run(
    ["ncu", "--import", rep_file, "--csv", "--page", "raw"],
    capture_output=True, text=True
)

reader = csv.reader(io.StringIO(result.stdout))
header = next(reader)

for row in reader:
    if row[0] == kernel_id:
        metric = row[12]
        unit = row[13]
        value = row[14]
        if filter_pattern:
            if filter_pattern.lower() in metric.lower():
                print(f"{metric}: {value} {unit}")
        else:
            print(f"{metric}: {value} {unit}")
