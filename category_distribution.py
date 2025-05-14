# Topic Category Distribution Analysis
# -----------------------------------
# This notebook analyzes the distribution of arXiv categories across each topic
# and determines the primary category for each topic.

import pandas as pd
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

# Define list of all math categories
math_categories = [
    'math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA', 'math.CO', 'math.CT',
    'math.CV', 'math.DG', 'math.DS', 'math.FA', 'math.GM', 'math.GN', 'math.GR',
    'math.GT', 'math.HO', 'math.IT', 'math.KT', 'math.LO', 'math.MG', 'math.MP',
    'math.NA', 'math.NT', 'math.OA', 'math.OC', 'math.PR', 'math.QA', 'math.RA',
    'math.RT', 'math.SG', 'math.ST'
]

# Load data files
papers_df = pd.read_parquet('results/topics/papers_by_topic_no_outliers.parquet')
topics_df = pd.read_csv('results/topics/common_topics.csv')

print(f"Loaded {len(papers_df)} papers and {len(topics_df)} topics")

# Get list of all topics
topic_list = list(topics_df['topic'].unique())
print(f"Processing {len(topic_list)} unique topics")

# Create category distribution for each topic
print("Calculating category distribution for each topic...")
topic_category_distribution = {}

for topic in topic_list:
    # Initialize topic dictionary 
    topic_dist = {}
    topic_count = topics_df.loc[topics_df['topic'] == topic, 'count'].iloc[0]
    
    # Calculate distribution across categories
    topic_papers_df = papers_df[papers_df['topic'] == topic]
    for cat in math_categories:
        cat_papers_count = len(topic_papers_df[topic_papers_df['categories'].str.contains(cat, na=False)])
        topic_dist[cat] = cat_papers_count / topic_count
    
    topic_category_distribution[str(topic)] = topic_dist

print("Category distribution calculation complete")

# Helper function to save the distribution as JSON
def save_topic_category_distribution(topic_category_distribution, filename):
    """
    Save the topic category distribution dictionary to a JSON file.
    
    Args:
        topic_category_distribution: Nested dictionary mapping topics to category distributions
        filename: Path where to save the JSON file
    """
    # Create a custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Ensure all values are serializable
    serializable_dict = {}
    for topic, cat_dist in topic_category_distribution.items():
        serializable_dict[str(topic)] = {cat: float(value) for cat, value in cat_dist.items()}
    
    # Create directory if it doesn't exist
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file with pretty formatting
    with open(filename, 'w') as f:
        json.dump(serializable_dict, f, cls=NumpyEncoder, indent=2)
    
    print(f"Saved topic category distribution to {filename}")

# Save the distribution to JSON
output_path = 'results/topics/topic_category_distribution.json'
save_topic_category_distribution(topic_category_distribution, output_path)

# Determine primary category for each topic
print("Determining primary category for each topic...")
primary_category = {topic: max(cat_dist, key=cat_dist.get) 
                   for topic, cat_dist in topic_category_distribution.items()}

# Convert to DataFrame
primary_category_df = pd.DataFrame(list(primary_category.items()), 
                                   columns=['topic', 'primary_category'])
primary_category_df['topic'] = primary_category_df['topic'].astype(int)

# Merge with existing topics dataframe
enhanced_topics_df = pd.merge(
    topics_df,
    primary_category_df,
    on="topic",
    how="left"
)

print(f"Enhanced topics dataframe shape: {enhanced_topics_df.shape}")

# Save the improved dataframe with primary categories
filepath = Path('results/topics/common_topics.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
enhanced_topics_df.to_csv(filepath, index=False)
print(f"Saved enhanced topics data to {filepath}")

# Display category distribution for a sample topic as validation
sample_topic = str(topic_list[0])
print(f"\nSample distribution for Topic {sample_topic}:")
sorted_dist = sorted(topic_category_distribution[sample_topic].items(), 
                     key=lambda x: x[1], reverse=True)
for cat, value in sorted_dist[:5]:  # Show top 5 categories
    print(f"  {cat}: {value:.4f}")

print(f"Primary category: {primary_category[sample_topic]}")
