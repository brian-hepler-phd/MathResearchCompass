#!/usr/bin/env python3
"""
Topic Trend Analysis for Math Research Compass
---------------------------------------------
This script performs temporal topic modeling analysis on arXiv papers:
1. Fetches and processes data from arXiv for specified categories
2. Analyzes topics using BERTopic
3. Tracks topic prevalence over time
4. Identifies emerging and declining research areas
5. Creates interactive visualizations

Usage:
    python topic_trends_analyzer.py --subjects math.AG math.AT --years 5
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Import BERTopic and dependencies
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("topic_trends")

# Constants
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
TOPIC_DIR = RESULTS_DIR / "topics"
for dir_path in [RAW_DIR, CLEANED_DIR, RESULTS_DIR, PLOTS_DIR, TOPIC_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Common English stopwords plus math/research specific ones
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", 
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", 
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", 
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", 
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", 
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", 
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", 
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", 
    "you've", "your", "yours", "yourself", "yourselves",
    # Research/math specific stopwords
    "prove", "paper", "show", "result", "consider", "using", "use", "given", "thus",
    "therefore", "hence", "obtain", "we", "our", "propose", "method", "approach",
    "introduce", "study", "analyze", "present", "develop", "data", "set", "model",
    "algorithm", "equation", "function", "theorem", "lemma", "define", "definition",
    "example", "problem", "solution", "property", "application"
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze topic trends in arXiv math papers over time"
    )
    parser.add_argument(
        "--subjects", 
        nargs="+", 
        default=["math.AG", "math.AT", "math.RT", "math.SG"],
        help="arXiv subject categories to analyze"
    )
    parser.add_argument(
        "--years", 
        type=int, 
        default=5,
        help="Number of years of data to analyze"
    )
    parser.add_argument(
        "--min-topic-size", 
        type=int, 
        default=15,
        help="Minimum cluster size for topics"
    )
    parser.add_argument(
        "--time-interval", 
        type=str, 
        choices=["month", "quarter", "year"],
        default="quarter",
        help="Time interval for temporal analysis"
    )
    parser.add_argument(
        "--force-download", 
        action="store_true",
        help="Force re-download of data even if it exists"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def run_script(script_path, args):
    """
    Run a Python script with the given arguments.
    
    Args:
        script_path: Path to the script to run
        args: List of command-line arguments
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [sys.executable, script_path] + args
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")
        return False


def ensure_data_ready(subjects, force_download=False):
    """
    Ensure data is downloaded and processed for all specified subjects.
    
    Args:
        subjects: List of arXiv subject categories
        force_download: Whether to force re-download even if data exists
    
    Returns:
        True if successful, False otherwise
    """
    # Find script paths
    fetch_script = Path("fetch_data.py")
    process_script = Path("process_data.py")
    
    if not fetch_script.exists() or not process_script.exists():
        logger.error(f"Required scripts not found: {fetch_script} or {process_script}")
        return False
    
    # Check if we need to download or process data
    missing_data = False
    for subject in subjects:
        safe_subject = subject.replace(".", "_")
        raw_file = RAW_DIR / f"{safe_subject}.json"
        cleaned_file = CLEANED_DIR / f"{safe_subject}.csv"
        
        if force_download or not raw_file.exists():
            missing_data = True
            break
            
        if not cleaned_file.exists():
            missing_data = True
            break
    
    if not missing_data:
        logger.info("All data already available, skipping download and processing")
        return True
    
    # Run fetch_data.py for all subjects
    fetch_args = ["--categories"] + subjects
    if force_download:
        fetch_args.append("--overwrite")
    
    success = run_script(fetch_script, fetch_args)
    if not success:
        return False
    
    # Run process_data.py for all subjects
    process_args = ["--categories"] + subjects
    if force_download:
        process_args.append("--overwrite")
    
    success = run_script(process_script, process_args)
    return success


def load_data(subjects):
    """
    Load processed data for all specified subjects.
    
    Args:
        subjects: List of arXiv subject categories
    
    Returns:
        Combined DataFrame with data from all subjects
    """
    logger.info(f"Loading data for subjects: {', '.join(subjects)}")
    
    frames = []
    for subject in subjects:
        safe_subject = subject.replace(".", "_")
        csv_path = CLEANED_DIR / f"{safe_subject}.csv"
        
        if not csv_path.exists():
            logger.error(f"Processed data file {csv_path} not found")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            frames.append(df)
            logger.debug(f"Loaded {len(df)} papers from {csv_path}")
        except Exception as e:
            logger.error(f"Error loading {csv_path}: {e}")
    
    if not frames:
        logger.error("No data loaded for any subject")
        return pd.DataFrame()
    
    combined_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} papers across all subjects")
    
    # Convert string representations of lists back to actual lists
    list_columns = ["authors", "categories", "all_categories"]
    for col in list_columns:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Convert date columns to datetime
    date_columns = ["published_date", "updated_date"]
    for col in date_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_datetime(combined_df[col])
    
    # Create time fields for analysis
    if "published_date" in combined_df.columns:
        combined_df["year"] = combined_df["published_date"].dt.year
        combined_df["month"] = combined_df["published_date"].dt.month
        combined_df["quarter"] = combined_df["published_date"].dt.quarter
        combined_df["yearmonth"] = combined_df["published_date"].dt.to_period("M")
        combined_df["yearquarter"] = combined_df["published_date"].apply(
            lambda x: f"{x.year}-Q{x.quarter}"
        )
    
    # Create text field for NLP if not already present
    if "text_for_nlp" not in combined_df.columns and "title" in combined_df.columns and "abstract" in combined_df.columns:
        combined_df["text_for_nlp"] = combined_df["title"] + " " + combined_df["abstract"]
    
    return combined_df


def prepare_data(subjects, force_download=False):
    """
    Prepare data for all specified subjects.
    
    Args:
        subjects: List of arXiv math subject categories
        force_download: Whether to force re-download even if data exists
    
    Returns:
        DataFrame with preprocessed paper data
    """
    # Ensure data is ready
    success = ensure_data_ready(subjects, force_download)
    if not success:
        logger.error("Failed to prepare data")
        return pd.DataFrame()
    
    # Load and combine data
    df = load_data(subjects)
    return df


def filter_recent_years(df, years):
    """Filter dataframe to only include papers from the last N years."""
    if df.empty or "year" not in df.columns:
        return df
        
    current_year = datetime.now().year
    start_year = current_year - years
    
    filtered_df = df[df["year"] >= start_year].copy()
    logger.info(f"Filtered to {len(filtered_df)} papers from {start_year}-{current_year}")
    
    return filtered_df


def build_advanced_topic_model(
    min_topic_size=15,
    n_gram_range=(1, 3)
):
    """
    Build an enhanced BERTopic model with improved parameters.
    
    Args:
        min_topic_size: Minimum size of topics
        n_gram_range: Range of n-grams to consider
    
    Returns:
        Configured BERTopic model (unfitted)
    """
    # Set up the vectorizer with n-grams
    vectorizer_model = CountVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=n_gram_range,
        min_df=5,
        max_df=0.8
    )
    
    # Use a better-tuned embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # UMAP - Better parameters for mathematical topics
    umap_model = UMAP(
        n_neighbors=15,
        n_components=10,  # More components can capture more subtle variations
        min_dist=0.05,
        metric="cosine",
        low_memory=True,
        random_state=42
    )
    
    # HDBSCAN - Improved clustering settings
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    
    # Create custom representation model
    representation_model = KeyBERTInspired()
    
    # Create and return the model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        verbose=True
    )
    
    return topic_model


def run_topic_modeling(
    df, 
    min_topic_size=15
):
    """
    Run topic modeling on the dataset.
    
    Args:
        df: DataFrame with preprocessed paper data
        min_topic_size: Minimum size of topics
    
    Returns:
        Tuple of (fitted topic model, document topics, document topic probabilities)
    """
    logger.info("Building topic model...")
    model = build_advanced_topic_model(min_topic_size=min_topic_size)
    
    # Prepare documents
    docs = df["text_for_nlp"].tolist()
    logger.info(f"Running topic modeling on {len(docs)} documents")
    
    # Fit the model
    topics, probs = model.fit_transform(docs)
    
    # Log topic statistics
    topic_info = model.get_topic_info()
    num_topics = len(topic_info[topic_info["Topic"] != -1])
    logger.info(f"Found {num_topics} topics (excluding outliers)")
    
    return model, topics, probs


def analyze_temporal_trends(
    model,
    df,
    topics,
    time_interval="quarter"
):
    """
    Analyze how topics evolve over time.
    
    Args:
        model: Fitted BERTopic model
        df: DataFrame with paper data
        topics: Document topic assignments
        time_interval: Time interval for analysis ("month", "quarter", or "year")
    
    Returns:
        Dictionary with temporal analysis results
    """
    logger.info(f"Analyzing temporal trends using {time_interval} intervals")
    
    # Map time interval to the appropriate dataframe column
    interval_column_map = {
        "month": "yearmonth",
        "quarter": "yearquarter",
        "year": "year"
    }
    
    # Convert timestamps to strings
    timestamps_col = df[interval_column_map[time_interval]]
    logger.info(f"Timestamp column type: {timestamps_col.dtype}")
    logger.info(f"Sample values: {timestamps_col.head().tolist()}")
    
    # Convert to strings with extra validation
    timestamps = []
    for t in timestamps_col:
        timestamps.append(str(t))
    
    # Verify conversion
    logger.info(f"All timestamps are now strings: {all(isinstance(t, str) for t in timestamps)}")
    
    # Get documents for BERTopic (it needs the original text documents)
    docs = df["text_for_nlp"].tolist()
    logger.info(f"Document sample: {docs[0][:50]}...")
    
    # Get temporal topic representations
    try:
        # Try with docs parameter
        topics_over_time = model.topics_over_time(
            topics, 
            timestamps, 
            docs=docs,  # Explicitly pass the docs parameter
            nr_bins=None
        )
    except Exception as e:
        logger.warning(f"Error with docs parameter: {e}")
        try:
            # Try without docs parameter
            topics_over_time = model.topics_over_time(
                topics, 
                timestamps,
                nr_bins=None
            )
        except Exception as e:
            logger.error(f"Failed to generate topics over time: {e}")
            # Return empty results so the rest of the pipeline can continue
            return {
                "topics_over_time": [],
                "trends": {},
                "emerging_topics": [],
                "declining_topics": []
            }
    
    # Calculate topic growth and trend metrics
    topic_info = model.get_topic_info()
    trends = {}
    
    # Only analyze actual topics (not outliers)
    for topic_id in topic_info[topic_info["Topic"] != -1]["Topic"]:
        topic_data = topics_over_time[topics_over_time["Topic"] == topic_id]
        
        if len(topic_data) > 1:  # Ensure enough data points
            # Calculate growth metrics
            topic_freq = topic_data["Frequency"].values
            growth_abs = topic_freq[-1] - topic_freq[0]
            growth_rel = (topic_freq[-1] / topic_freq[0] - 1) * 100 if topic_freq[0] > 0 else 0
            
            # Calculate trend indicators (using simple linear regression)
            x = np.arange(len(topic_freq))
            slope = np.polyfit(x, topic_freq, 1)[0]
            normalized_slope = slope / np.mean(topic_freq) if np.mean(topic_freq) > 0 else 0
            
            trends[topic_id] = {
                "name": topic_info[topic_info["Topic"] == topic_id]["Name"].iloc[0],
                "frequency": topic_freq.tolist(),
                "timestamps": topic_data["Timestamp"].tolist(),
                "growth_absolute": float(growth_abs),
                "growth_relative": float(growth_rel),
                "trend_slope": float(slope),
                "normalized_slope": float(normalized_slope)
            }
    
    # Identify top emerging and declining topics
    sorted_topics = sorted(trends.items(), key=lambda x: x[1]["normalized_slope"], reverse=True)
    
    emerging = [
        {"id": t_id, **t_data} 
        for t_id, t_data in sorted_topics[:10]
    ]
    
    declining = [
        {"id": t_id, **t_data} 
        for t_id, t_data in sorted_topics[-10:]
    ]
    
    return {
        "topics_over_time": topics_over_time.to_dict(orient="records"),
        "trends": trends,
        "emerging_topics": emerging,
        "declining_topics": declining
    }


def create_visualizations(
    model, 
    df, 
    topics, 
    temporal_data,
    top_n_topics=15
):
    """
    Create and save visualizations for the dashboard.
    
    Args:
        model: Fitted BERTopic model
        df: DataFrame with paper data
        topics: Document topic assignments
        temporal_data: Dictionary with temporal analysis results
        top_n_topics: Number of top topics to visualize
    """
    logger.info("Generating visualizations")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Topic hierarchy visualization
    logger.info("Creating hierarchical topic visualization")
    try:
        hierarchical_topics = model.hierarchical_topics(df["text_for_nlp"].tolist())
        hierarchy_fig = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        hierarchy_fig.write_html(PLOTS_DIR / f"topic_hierarchy_{timestamp}.html")
        
        # Also save hierarchy data as JSON for the dashboard
        hierarchical_data = hierarchical_topics.to_dict(orient="records")
        with open(TOPIC_DIR / f"hierarchical_topics_{timestamp}.json", "w") as f:
            json.dump(hierarchical_data, f)
    except Exception as e:
        logger.error(f"Failed to create hierarchy visualization: {e}")
    
    # 2. Topic prevalence over time
    logger.info("Creating topic prevalence visualization")
    try:
        # Get topic info for top topics by size (excluding outliers)
        topic_info = model.get_topic_info()
        top_topics = topic_info[topic_info["Topic"] != -1].head(top_n_topics)["Topic"].tolist()
        
        # Convert topics_over_time to DataFrame for visualization
        tot_df = pd.DataFrame(temporal_data["topics_over_time"])
        tot_filtered = tot_df[tot_df["Topic"].isin(top_topics)]
        
        # Create interactive plot
        fig = px.line(
            tot_filtered, 
            x="Timestamp", 
            y="Frequency", 
            color="Topic",
            hover_name="Topic",
            title="Topic Prevalence Over Time",
            labels={"Timestamp": "Time Period", "Frequency": "Relative Frequency"},
        )
        
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Topic Prevalence",
            legend_title="Topic",
            hovermode="closest"
        )
        
        fig.write_html(PLOTS_DIR / f"topic_prevalence_{timestamp}.html")
    except Exception as e:
        logger.error(f"Failed to create prevalence visualization: {e}")
    
    # 3. Topic distribution visualization
    logger.info("Creating topic distribution visualization")
    try:
        fig = model.visualize_topics(top_n_topics=top_n_topics)
        fig.write_html(PLOTS_DIR / f"topic_distribution_{timestamp}.html")
    except Exception as e:
        logger.error(f"Failed to create topic distribution visualization: {e}")
    
    # 4. Emerging and declining topics
    logger.info("Creating emerging and declining topics visualization")
    try:
        # Emerging topics
        emerging_data = temporal_data["emerging_topics"][:5]
        emerging_ids = [item["id"] for item in emerging_data]
        
        em_fig = go.Figure()
        
        for topic_id in emerging_ids:
            topic_data = [t for t in temporal_data["topics_over_time"] if t["Topic"] == topic_id]
            if topic_data:
                topic_name = model.get_topic(topic_id)[0][0]  # Get most representative term
                timestamps = [d["Timestamp"] for d in topic_data]
                frequencies = [d["Frequency"] for d in topic_data]
                
                em_fig.add_trace(go.Scatter(
                    x=timestamps, 
                    y=frequencies,
                    mode="lines+markers",
                    name=f"Topic {topic_id}: {topic_name}"
                ))
        
        em_fig.update_layout(
            title="Emerging Research Topics",
            xaxis_title="Time Period",
            yaxis_title="Topic Frequency",
            hovermode="closest"
        )
        
        em_fig.write_html(PLOTS_DIR / f"emerging_topics_{timestamp}.html")
        
        # Declining topics
        declining_data = temporal_data["declining_topics"][:5]
        declining_ids = [item["id"] for item in declining_data]
        
        de_fig = go.Figure()
        
        for topic_id in declining_ids:
            topic_data = [t for t in temporal_data["topics_over_time"] if t["Topic"] == topic_id]
            if topic_data:
                topic_name = model.get_topic(topic_id)[0][0]  # Get most representative term
                timestamps = [d["Timestamp"] for d in topic_data]
                frequencies = [d["Frequency"] for d in topic_data]
                
                de_fig.add_trace(go.Scatter(
                    x=timestamps, 
                    y=frequencies,
                    mode="lines+markers",
                    name=f"Topic {topic_id}: {topic_name}"
                ))
        
        de_fig.update_layout(
            title="Declining Research Topics",
            xaxis_title="Time Period",
            yaxis_title="Topic Frequency",
            hovermode="closest"
        )
        
        de_fig.write_html(PLOTS_DIR / f"declining_topics_{timestamp}.html")
    except Exception as e:
        logger.error(f"Failed to create emerging/declining visualization: {e}")
    
    # 5. Generate topic word clouds for top topics
    logger.info("Creating topic word clouds")
    try:
        for topic_id in top_topics[:10]:  # Create word clouds for the top 10 topics
            topic_words = model.get_topic(topic_id)
            if topic_words:
                # Create word dictionary with weights
                word_dict = {word: weight for word, weight in topic_words}
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=30
                ).generate_from_frequencies(word_dict)
                
                # Plot and save
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f"Topic {topic_id}: {topic_words[0][0]}")
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / f"wordcloud_topic_{topic_id}_{timestamp}.png", dpi=300)
                plt.close()
    except Exception as e:
        logger.error(f"Failed to create word clouds: {e}")


def save_topic_data(model, df, topics, temporal_data):
    """
    Save topic modeling results for later use in the dashboard.
    
    Args:
        model: Fitted BERTopic model
        df: DataFrame with paper data
        topics: Document topic assignments
        temporal_data: Dictionary with temporal analysis results
    """
    logger.info("Saving topic modeling results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save topic info
    topic_info = model.get_topic_info()
    topic_info.to_csv(TOPIC_DIR / f"topic_info_{timestamp}.csv", index=False)
    
    # 2. Save topic keywords (top terms for each topic)
    keywords = {}
    for topic_id in topic_info["Topic"]:
        if topic_id != -1:  # Skip outlier topic
            keywords[topic_id] = model.get_topic(topic_id)
    
    with open(TOPIC_DIR / f"topic_keywords_{timestamp}.json", "w") as f:
        # Convert NumPy values to Python native types for JSON serialization
        serializable_keywords = {}
        for topic_id, terms in keywords.items():
            serializable_keywords[int(topic_id)] = [
                [term, float(weight)] for term, weight in terms
            ]
        json.dump(serializable_keywords, f)
    
    # 3. Save document-topic mappings
    doc_topics = pd.DataFrame({
        "id": df["id"].tolist(),
        "title": df["title"].tolist(),
        "topic": topics,
        "published_date": df["published_date"].dt.strftime("%Y-%m-%d").tolist(),
        "primary_category": df["primary_category"].tolist()
    })
    doc_topics.to_csv(TOPIC_DIR / f"document_topics_{timestamp}.csv", index=False)
    
    # 4. Save temporal trend data
    with open(TOPIC_DIR / f"topic_trends_{timestamp}.json", "w") as f:
        json.dump(temporal_data, f, cls=NumpyJSONEncoder)
    
    # 5. Create a metadata file with information about this run
    metadata = {
        "timestamp": timestamp,
        "num_documents": len(df),
        "num_topics": len(topic_info[topic_info["Topic"] != -1]),
        "subjects": df["primary_category"].unique().tolist(),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "file_references": {
            "topic_info": f"topic_info_{timestamp}.csv",
            "topic_keywords": f"topic_keywords_{timestamp}.json",
            "document_topics": f"document_topics_{timestamp}.csv",
            "topic_trends": f"topic_trends_{timestamp}.json",
        }
    }
    
    with open(TOPIC_DIR / f"metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f)
    
    logger.info(f"All topic data saved with timestamp: {timestamp}")


# Helper class for JSON serialization of NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Period):
            return str(obj)
        return super().default(obj)


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare data
    df = prepare_data(args.subjects, force_download=args.force_download)
    
    if df.empty:
        logger.error("No data available for analysis. Exiting.")
        return
    
    # Filter to recent years
    df_recent = filter_recent_years(df, args.years)
    
    if df_recent.empty:
        logger.error(f"No data found for the last {args.years} years. Exiting.")
        return
    
    # Run topic modeling
    model, topics, probs = run_topic_modeling(df_recent, min_topic_size=args.min_topic_size)
    
    # Analyze temporal trends
    temporal_data = analyze_temporal_trends(model, df_recent, topics, time_interval=args.time_interval)
    
    # Create visualizations
    create_visualizations(model, df_recent, topics, temporal_data)
    
    # Save topic data for dashboard
    save_topic_data(model, df_recent, topics, temporal_data)
    
    logger.info("Topic analysis complete!")


if __name__ == "__main__":
    main()