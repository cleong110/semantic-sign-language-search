import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def parse_csv(result_csv):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(result_csv)
    
    # Strip possible extra whitespace in gloss columns
    df['query_gloss'] = df['query_gloss'].str.strip()
    df['query_gloss'] = df['query_gloss'].str.replace(",","")
    df['search_result_gloss'] = df['search_result_gloss'].str.strip()
    
    # Count confused glosses (non-identical gloss pairs)
    confused_glosses = df[df['query_gloss'] != df['search_result_gloss']]
    
    # Group by (query_gloss, search_result_gloss) to count confusion occurrences
    gloss_confusions = confused_glosses.groupby(['query_gloss', 'search_result_gloss']).size().reset_index(name='confusion_count')
    
    return gloss_confusions
    
def get_most_confused(gloss_confusions, min_val=1, top_n=0):
    # Sort by confusion count in descending order
    gloss_confusions_sorted = gloss_confusions.sort_values(by='confusion_count', ascending=False)
    
    # Filter to only gloss pairs where the confusion count is >= min_val
    filtered_confusions = gloss_confusions_sorted[gloss_confusions_sorted['confusion_count'] >= min_val]
    
    if top_n >0:
        filtered_confusions = filtered_confusions.head(top_n)
    return filtered_confusions


def plot_most_confused(gloss_confusions, top_n=10):
    # Sort by the confusion count in descending order
    #gloss_confusions_sorted = gloss_confusions.sort_values(by='confusion_count', ascending=False).head(top_n)
    gloss_confusions_sorted = get_most_confused(gloss_confusions, top_n=top_n)
    
    # Plot the top N most confused glosses
    plt.figure(figsize=(10, 6))
    sns.barplot(x='confusion_count', y='query_gloss', hue='search_result_gloss', data=gloss_confusions_sorted, dodge=False)
    plt.title(f"Top {top_n} Most Confused Glosses")
    plt.xlabel('Confusion Count')
    plt.ylabel('Query Gloss')
    plt.legend(title='Confused With')
    plt.tight_layout()
    plt.show()

# TODO: fix
def analyze_gloss_confusion(df, gloss, min_val=1):
    """
    Analyzes what other glosses the given gloss is confused with.
    Returns a list of glosses it is confused with and the counts.
    """
    filtered_df = df[df['query_gloss'] == gloss]
    confused_with = filtered_df.groupby('search_result_gloss').size().reset_index(name='confusion_count')
    confused_with = confused_with[confused_with['confusion_count'] >= min_val].sort_values(by='confusion_count', ascending=False)
    
    if confused_with.empty:
        print(f"{gloss} is not confused with any other gloss.")
    else:
        print(f"Gloss '{gloss}' is confused with:")
        for _, row in confused_with.iterrows():
            print(f"  {row['search_result_gloss']}: {row['confusion_count']}")
    return confused_with

# TODO: fix
def explore_top_confusions(gloss_confusions, top_n=0, min_val=1):
    """
    Explores the top N most confused glosses, analyzing what each one is confused with.
    """
    most_confused_glosses = get_most_confused(gloss_confusions, min_val=min_val, top_n=top_n)
#    if N > 0:
#        most_confused_glosses = most_confused_glosses[:top_n]
    print(most_confused_glosses.head())
    # Iterating over the DataFrame rows
    for index, row in most_confused_glosses.iterrows():
        query_gloss = row['query_gloss']
        search_result_gloss = row['search_result_gloss']
        confusion_count = row['confusion_count']
        print(f"\nAnalyzing '{query_gloss}' (most confused with '{search_result_gloss}': {confusion_count})")
        analyze_gloss_confusion(gloss_confusions, query_gloss, min_val=min_val)

# Call explore_top_confusions(df, N=5) to explore the top 5 most confused glosses

def print_most_confused(gloss_confusions, min_val=1, top_n=10, specific_gloss=None):
    most_confused_glosses = get_most_confused(gloss_confusions, min_val=min_val)
    
    top_most_confused_glosses = []
    
    # Print the most confused glosses in text format
    n = 0
    if specific_gloss is None:        
        print("\nTop Most Confused Glosses (Actual -> Predicted: Count):")
    else:
        print(f"\nTop Most Confused Glosses for {specific_gloss} (Actual -> Predicted: Count):")
    for __, row in most_confused_glosses.iterrows():
        if (top_n > 0) and (n >= top_n):
                return top_most_confused_glosses
        if (specific_gloss is None):
            print(f"{row['query_gloss']} -> {row['search_result_gloss']}: {row['confusion_count']}")
            top_most_confused_glosses.append(row['query_gloss'])
            n = n+1
        elif (row['query_gloss']==specific_gloss):
            print(f"{row['query_gloss']} -> {row['search_result_gloss']}: {row['confusion_count']}")
            n = n+1
    return top_most_confused_glosses
#            print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#            print(f"AHA! {specific_gloss} at {row}")
#            print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#        else: 
#            print("*** AWW**")
#            print(f"can't find {specific_gloss} at")
#            print(f"{__}, {row}")
#            print(f"gloss on this row: {row['search_result_gloss']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze most confused glosses from result CSV.")
    parser.add_argument("result_csv", type=Path, help="Path to the CSV file containing result data")
    parser.add_argument("--top_n", type=int, default=10, help="Number of most confused glosses to plot")
    parser.add_argument("--min_val", type=int, default=1, help="Minimum confusion count for text output")
    args = parser.parse_args()
    
    # Parse CSV and calculate confusions
    gloss_confusions = parse_csv(args.result_csv)
    
    # Plot the most confused glosses
    plot_most_confused(gloss_confusions, args.top_n)
    
    # Print the most confused glosses in text format
    top_most_confused_glosses = set(print_most_confused(gloss_confusions, args.min_val, top_n=11))
    top_most_confused_glosses =list( top_most_confused_glosses)[:10]
    print(top_most_confused_glosses)
    
    #for most_confused in ["COMMUNITY", "BOOTS", "EARTH", "FINE_2", "HAMSTER", "STARE", "ADMIT", "BLOW_CANDLE_2", "TOILET", "POSITIVE"]:
    for most_confused in top_most_confused_glosses:
        print_most_confused(gloss_confusions, top_n=0, min_val=args.min_val,specific_gloss = most_confused)
#        print_most_confused(gloss_confusions, top_n=0, min_val=1,specific_gloss = "BOOTS")
    
    #explore_top_confusions(gloss_confusions, top_n=args.top_n)

if __name__ == "__main__":
    main()
