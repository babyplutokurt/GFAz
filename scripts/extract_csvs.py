import re
import csv
import os

latex_path = 'paper/table/compressors_evaluation.tex'
output_dir = 'paper/table/csvs'
os.makedirs(output_dir, exist_ok=True)

compressors = ['Gzip', 'Zstd', 'sqz', 'sqz+bgzip', 'sqz+Zstd', 'GBZ', 'gfaz(CPU)', 'gfaz(GPU)']

with open(latex_path, 'r') as f:
    lines = f.readlines()

current_dataset = None
data = {} # {dataset: {metric: [vals]}}

rows_processed = 0

for line in lines:
    line = line.strip()
    if not line: continue
    
    # Check for dataset start
    # Matches \TabDataName{chr1.} or similar
    match = re.search(r'\\TabDataName\{(.*?)\}', line)
    if match:
        raw_name = match.group(1)
        # Clean name: remove trailing dot, replace odd chars
        current_dataset = raw_name.replace('.', '')
        # Special case for E.coli -> Ecoli
        if 'E.coli' in raw_name:
            current_dataset = 'Ecoli'
            
        data[current_dataset] = {}
    
    if current_dataset:
        metric = None
        if '& Ratio' in line:
            metric = 'Ratio'
        elif '& Co.' in line:
            metric = 'Co.'
        elif '& De.' in line:
            metric = 'De.'
            
        if metric:
            # Parse values
            # The line is expected to be split by '&'
            parts = line.split('&')
            
            # parts[0] is dataset column
            # parts[1] is metric column
            # parts[2..9] are the compressors
            
            if len(parts) < 10:
                print(f"Skipping malformed line for {current_dataset} {metric}: {line}")
                continue
                
            vals_raw = parts[2:10] 
            
            clean_vals = []
            for v in vals_raw:
                # Remove trailing \\ and comments
                v = v.split('\\\\')[0]
                v = v.split('%')[0]
                
                # Remove macros
                v = re.sub(r'\\NoBcBestCr\{([^\}]+)\}', r'\1', v)
                v = re.sub(r'\\SecondbestCr\{([^\}]+)\}', r'\1', v)
                
                # Remove any other latex stuff if needed? 
                # e.g., \color{...} - seemingly not present in data cells
                
                v = v.strip()
                clean_vals.append(v)
            
            data[current_dataset][metric] = clean_vals

# Write CSVs
for dataset, metrics in data.items():
    filename = os.path.join(output_dir, f"{dataset}.csv")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Compressor', 'Ratio', 'Compression_Speed_MBps', 'Decompression_Speed_MBps'])
        
        for i, comp in enumerate(compressors):
            r = metrics.get('Ratio', ['']*8)[i] if i < len(metrics.get('Ratio', [])) else ''
            c = metrics.get('Co.', ['']*8)[i] if i < len(metrics.get('Co.', [])) else ''
            d = metrics.get('De.', ['']*8)[i] if i < len(metrics.get('De.', [])) else ''
            writer.writerow([comp, r, c, d])
    print(f"Created {filename}")
