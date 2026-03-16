import csv

def format_val(val_str):
    if not val_str or val_str.lower() == 'na':
        return None
    
    # Specific hack for malformed value in copy.csv
    if "529.1714361058.3428738" in val_str:
        return 1058.34
        
    try:
        f = float(val_str)
        return round(f, 2)
    except ValueError:
        return None

def get_formatted_string(val, rank):
    if val is None:
        return ""
    
    if val >= 100:
       s = f"{val:.0f}"
    elif val >= 10:
        s = f"{val:.1f}"
    else:
        s = f"{val:.2f}"
    
    if rank == 1:
        return f"\\NoBcBestCr{{{s}}}"
    elif rank == 2:
        return f"\\SecondbestCr{{{s}}}"
    else:
        return s

def process_dataset(rows, dataset_key, latex_name):
    row = next((r for r in rows if r['dataset'].startswith(dataset_key)), None)
    if not row:
        return None

    # Define groups of columns (Ratio, Comp, Decomp)
    # The order must match the table columns: Gzip, Zstd, Sqz, Sqz+bgz, Sqz+zstd, GBZ, Gfaz, Gfaz-GPU
    
    compressors = [
        ('gzip', 'gzip'),
        ('zstd', 'zstd'),
        ('sqz', 'sqz'),
        ('sqz_bgzip', 'sqz_bgzip'),
        ('sqz_zstd', 'sqz_zstd'),
        ('gbz', 'gbz'),
        ('gfaz', 'gfaz'),
        ('gfaz_gpu', 'gfaz_gpu')
    ]
    
    metrics = [
        ('ratio', 'Ratio'),
        ('comp_MBps', 'Co.'),
        ('decomp_MBps', 'De.')
    ]
    
    latex_rows = []
    
    for metric_suffix, metric_name in metrics:
        values = []
        for _, prefix in compressors:
            key = f"{prefix}_{metric_suffix}"
            val = format_val(row.get(key, ''))
            values.append(val)
        
        # Determine strict ranking (ignoring Nones)
        # Sort descending
        valid_vals = [v for v in values if v is not None]
        valid_vals.sort(reverse=True)
        
        best = valid_vals[0] if len(valid_vals) > 0 else None
        second = valid_vals[1] if len(valid_vals) > 1 else None
        
        latex_cells = []
        for v in values:
            if v is None:
                latex_cells.append("")
            elif v == best:
                latex_cells.append(get_formatted_string(v, 1))
            elif v == second:
                latex_cells.append(get_formatted_string(v, 2))
            else:
                latex_cells.append(get_formatted_string(v, 0))
                
        # Construct line
        line_content = " & ".join(latex_cells)
        latex_rows.append(f"& {metric_name} & {line_content} \\\\")

    return latex_rows

if __name__ == "__main__":
    # Using 'eval_results copy.csv' as it seems to contain the updated data
    with open('scripts/eval_results copy.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    datasets = [
        ('chr1.', 'chr1.'),
        ('chr6.', 'chr6.'),
        ('Ecoli', 'EcoliGraph_MGC'),
        ('HPRCv1.1', 'hprc-v1.1'),
        ('HPRCv2.0', 'hprc-v2.0'),
        ('HPRCv2.1', 'hprc-v2.1')
    ]

    for latex_label, csv_key in datasets:
        res = process_dataset(rows, csv_key, latex_label)
        if res:
            print(f"--- {latex_label} ---")
            for l in res:
                print(l)
