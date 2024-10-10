import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def parse_results_file(result_file, debug=False):
    with open(result_file, 'r') as file:
        lines = file.readlines()
        print(f"Parsed file has: {len(str(lines))} lines")

    data = []
    current_block = None
    block_counter = 0
    
    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Processing lines"):
        line = line.strip()

        # Debug information for each line
        if debug:
            print(f"Processing line {i+1}: {line}")

        if line.endswith('.mp4,'):
            block_counter += 1
            current_block = {
                'query_filename': line.split(',')[0].strip(),
                'query_gloss': None,
                'query_dataset': None,
                'query_embedding_model': None,
                'search_results': []
            }
            if debug:
                print(f"Detected query block {block_counter} at line {i+1}: {current_block['query_filename']}")

        elif '* gloss:' in line:
            current_block['query_gloss'] = line.split(':')[1].strip().replace(",","")
            if debug:
                print(f"  Found gloss at line {i+1}: {current_block['query_gloss']}")

        elif '* dataset:' in line:
            current_block['query_dataset'] = line.split(':')[1].strip().replace(",","")
            if debug:
                print(f"  Found dataset at line {i+1}: {current_block['query_dataset']}")

        elif '* embedded with' in line:
            current_block['query_embedding_model'] = line.split('with')[1].strip()
            if debug:
                print(f"  Found embedding model at line {i+1}: {current_block['query_embedding_model']}")

        elif 'i ,' in line:
            if debug:
                print(f"  Found search result header at line {i+1}")

        elif ',' in line and current_block:
            result_line = line.replace('MATCH!', '').strip()
            result_parts = [x.strip() for x in result_line.split(',')]
            if len(result_parts) >= 4:
                search_result = {
                    'search_result_rank': result_parts[0],
                    'search_result_filename': result_parts[1],
                    'search_result_dataset': result_parts[2],
                    'search_result_gloss': result_parts[3],
                    'search_result_embedding_model': result_parts[4] if len(result_parts) > 4 else None
                }
                current_block['search_results'].append(search_result)
                if debug:
                    print(f"    Found search result at line {i+1}: {search_result['search_result_filename']}")

        # When end of a block is reached
        if line.startswith("0/") or i == len(lines) - 1:
            if current_block:
                for result in current_block['search_results']:
                    data.append([
                        current_block['query_filename'],
                        current_block['query_gloss'],
                        current_block['query_dataset'],
                        current_block['query_embedding_model'],
                        result['search_result_rank'],
                        result['search_result_filename'],
                        result['search_result_dataset'],
                        result['search_result_gloss'],
                        result['search_result_embedding_model']
                    ])
                current_block = None

    # Create DataFrame and return
    columns = ['query_filename', 'query_gloss', 'query_dataset', 'query_embedding_model',
               'search_result_rank', 'search_result_filename', 'search_result_dataset', 
               'search_result_gloss', 'search_result_embedding_model']
    df = pd.DataFrame(data, columns=columns)
    
    print(f"Total query blocks processed: {block_counter}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Parse result file and generate a CSV.")
    parser.add_argument('result_file', type=Path, help="The result file to parse")
    parser.add_argument('--output', type=Path, default=None, help="Output CSV file. Default: input filename with .csv")
    parser.add_argument('--debug', action='store_true', help="Print debug information")
    
    args = parser.parse_args()


    if args.output is None:
        
        args.output = args.result_file.with_suffix(".csv")
    # else: 
    #     output =args.output




    # Parse the result file
    print(f"Processing file: {args.result_file}")

    
    df = parse_results_file(args.result_file, debug=args.debug)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"CSV saved to {args.output}")

if __name__ == "__main__":
    main()
