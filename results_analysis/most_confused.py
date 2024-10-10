import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
sns.set_theme()

def load_confusion_data(input_csv):
    """Load the data from CSV file"""
    df = pd.read_csv(input_csv)
    return df

def build_confusion_matrix(df):
    """Build confusion matrix based on search_result_gloss vs query_gloss"""
    confusion_data = df.groupby(['query_gloss', 'search_result_gloss']).size().reset_index(name='confusion_count')
    return confusion_data

def top_most_confused(confusion_data, top_n=10, min_confusion=1):
    """Filter and get the top N most confused glosses"""
    # Filter confusions above min_confusion threshold
    filtered_confusions = confusion_data[confusion_data['confusion_count'] >= min_confusion]
    # Sort by confusion count
    top_confusions = filtered_confusions.sort_values('confusion_count', ascending=False).head(top_n)
    return top_confusions

def analyze_gloss_confusion(confusion_data, gloss, min_confusion=1):
    """Analyze and return confusions for a specific gloss, sorted by confusion count"""
    gloss_confusions = confusion_data[(confusion_data['query_gloss'] == gloss) & 
                                      (confusion_data['confusion_count'] >= min_confusion)]
    return gloss_confusions.sort_values('confusion_count', ascending=False)

def explore_top_confusions(confusion_data, top_n=10, min_confusion=1):
    """Print the top N most confused glosses and then analyze further confusions"""
    top_confusions = top_most_confused(confusion_data, top_n, min_confusion)
    
    print("Top Confusions:")
    for _, row in top_confusions.iterrows():
        query_gloss = row['query_gloss']
        search_result_gloss = row['search_result_gloss']
        confusion_count = row['confusion_count']
        print(f'{query_gloss} -> {search_result_gloss}: {confusion_count}')
    
    # Separately analyze and print further confusions for the top confusions
    print("\nFurther Confusions for Top Glosses:")
    for _, row in top_confusions.iterrows():
        query_gloss = row['query_gloss']
        detailed_confusions = analyze_gloss_confusion(confusion_data, query_gloss, min_confusion)
        print(f'Further confusions for {query_gloss}:')
        for _, detail_row in detailed_confusions.iterrows():
            print(f"  {detail_row['search_result_gloss']}: {detail_row['confusion_count']}")

def plot_top_confusions(confusion_data, top_n=10, min_confusion=1, dataset='', model='', show_plot=True, save_plot=True):
    """Plot the top N most confused glosses as a bar plot"""
    top_confusions = top_most_confused(confusion_data, top_n, min_confusion)
    title = f'Top {top_n} Most Confused Glosses, {model} on {dataset}'

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(top_confusions['query_gloss'] + ' -> ' + top_confusions['search_result_gloss'], 
             top_confusions['confusion_count'], color='skyblue')
    plt.xlabel('Confusion Count')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_plot:        
        plt.savefig(title.replace(" ","_").replace("/","").replace("\\","")+".png")
    if show_plot:
        plt.show()


def sort_gloss_pairs(df):
    stacked = df.stack()

    # Reset the index to convert the Series back to a DataFrame
    result_df = stacked.reset_index()
    print("Stacked")
    

    # Rename the columns for clarity
    result_df.columns = ['search_result_gloss', 'query_gloss', 'value']
    
    # Sort the DataFrame by 'value' in descending order
    result_df = result_df.sort_values(by='value', ascending=False)
    print(result_df)

    
    # Create categorical types to maintain order
    result_df['search_result_gloss'] = pd.Categorical(result_df['search_result_gloss'], 
                                                    ordered=True,
                                                    categories=result_df['search_result_gloss'].unique())
    result_df['query_gloss'] = pd.Categorical(result_df['query_gloss'], 
                                            ordered=True,
                                            categories=result_df['query_gloss'].unique())


    # Create a pivot table maintaining the original order
    pivot_table = pd.pivot_table(result_df, values='value', 
                                index='search_result_gloss', 
                                columns='query_gloss', 
                                fill_value=np.nan)
    
    return pivot_table




def plot_confusion_matrix(confusion_data, top_n=10, min_confusion=1, dataset='', model='', show_plot=True, save_plot=True):
    """Plot the confusion matrix for top confused glosses"""
    # Filter to get the top most confused glosses
    top_confusions = top_most_confused(confusion_data, top_n, min_confusion)
    title = f'Matrix for Top {top_n} Most Confused Glosses, {model} on {dataset}'

    # Create a pivot table to use as the confusion matrix
    confusion_matrix = top_confusions.pivot(index="query_gloss", columns="search_result_gloss", values="confusion_count")
    print(confusion_matrix)
    print(confusion_matrix.info())
    print(type(confusion_matrix))

    sorted_confusion_matrix = sort_gloss_pairs(confusion_matrix)
    print(sorted_confusion_matrix)
    confusion_matrix = sorted_confusion_matrix
    
    # Replace NaNs with 0 for better visualization
    confusion_matrix = confusion_matrix.fillna(0)



    # Plot the matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt=".0f", cbar_kws={'label': 'Confusion Count'})
    plt.title(title)
    plt.xlabel('Predicted Gloss')
    plt.ylabel('Actual Gloss')
    plt.tight_layout()
    if save_plot:        
        plt.savefig(title.replace(" ","_").replace("/","").replace("\\","")+".png")
    if show_plot:
        plt.show()
    

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Analyze most confused glosses.")
    parser.add_argument('input_path', type=Path, help='Path to input CSV file, or folder of CSV files')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top confused glosses to display.')
    parser.add_argument('--min_confusion', type=int, default=1, help='Minimum confusion count to consider.')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name for plot title.')
    parser.add_argument('--model', type=str, default='', help='Model name for plot title.')
    parser.add_argument('--save_plots', action="store_true", help='Save plots')
    parser.add_argument('--show_plots', action="store_true", help='Show plots')
    args = parser.parse_args()

    
    if args.input_path.is_dir():
        input_files = list(args.input_path.glob("*.csv"))
        print(f"Found {len(input_files)} CSV files to parse")
    else:
        input_files = [args.input_path]
            
        
            
    for input_file in input_files:
        # Load and process data
        print(f'Loading data from {input_file}')
        if args.model:
            model = args.model
        else:
            model = input_file.stem.split("model_")[-1]
            print(f"Model parsed from filename: {model}")
            
        # if args.dataset:
        #     dataset = args.dataset
        # else:
        #     dataset=input_file.stem.split("search_results")[-1].split("using_model")[0]
        #     print(f"parsed dataset from filename: {dataset}")
        df = load_confusion_data(input_file)
        confusion_data = build_confusion_matrix(df)
        confusion_data.to_csv(f"confusion_for_{model}_on_{args.dataset}.csv")
        
        # Explore top confusions
        explore_top_confusions(confusion_data, args.top_n, args.min_confusion)
        
        # Plot top confusions
        plot_top_confusions(confusion_data, args.top_n, args.min_confusion, args.dataset, model, args.show_plots, args.save_plots)
        
        # Plot confusion matrix
        plot_confusion_matrix(confusion_data, args.top_n, args.min_confusion, args.dataset, model, args.show_plots, args.save_plots)

if __name__ == "__main__":
    main()
