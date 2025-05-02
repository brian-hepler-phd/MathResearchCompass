#!/usr/bin/env python3
"""
Math Research Compass - arXiv Topic Trends Dashboard
---------------------------------------------------
Streamlit dashboard to visualize research trends in mathematical fields based on arXiv data.

Features:
- Topic prevalence over time
- Hierarchical topic visualization
- Emerging and declining research areas
- Collaboration network visualization

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
import ast
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Optional imports - these are used if available but not required
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except ImportError:
    PLOTLY_EVENTS_AVAILABLE = False

try:
    import streamlit_calendar
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Math Research Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TOPIC_DIR = RESULTS_DIR / "topics"
IMAGE_DIR = RESULTS_DIR / "images"

# Ensure directories exist
for dir_path in [DATA_DIR, CLEANED_DIR, RESULTS_DIR, PLOTS_DIR, TOPIC_DIR, IMAGE_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #3B82F6;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .trend-up {
        color: #10B981;
        font-weight: 500;
    }
    .trend-down {
        color: #EF4444;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

def on_page_change():
    """Handle page navigation"""
    page = st.session_state.navigation
    st.session_state.current_page = page

def apply_filters():
    """Apply selected filters to the data"""
    if "filters_applied" not in st.session_state:
        st.session_state.filters_applied = False
    
    st.session_state.filters_applied = True
    # This will force a rerun with the new filters
    st.rerun()  # Changed from experimental_rerun which is deprecated

# Helper functions
def load_subject_data(subject):
    """
    Load cleaned data for a specific subject.
    
    Args:
        subject: arXiv subject category (e.g., "math.AG")
    
    Returns:
        DataFrame with the cleaned data, or empty DataFrame if not found
    """
    safe_subject = subject.replace(".", "_")
    csv_path = CLEANED_DIR / f"{safe_subject}.csv"
    
    if not csv_path.exists():
        st.warning(f"Data file for {subject} not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()


def load_subjects(subjects):
    """
    Load data for all specified subjects.
    
    Args:
        subjects: List of arXiv subject categories
    
    Returns:
        Combined DataFrame with data from all subjects
    """
    frames = []
    for subject in subjects:
        df = load_subject_data(subject)
        if not df.empty:
            frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    combined_df = pd.concat(frames, ignore_index=True)
    
    # Convert string representations of lists back to actual lists
    list_columns = ["authors", "categories", "all_categories"]
    for col in list_columns:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    
    # Convert date columns to datetime
    date_columns = ["published_date", "updated_date"]
    for col in date_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_datetime(combined_df[col])
    
    return combined_df


def load_metadata_files() -> List[Dict]:
    """Load all available metadata files and return as list of dictionaries sorted by timestamp."""
    metadata_files = glob.glob(str(TOPIC_DIR / "metadata_*.json"))
    metadata_list = []
    
    for mfile in metadata_files:
        try:
            with open(mfile, "r") as f:
                metadata = json.load(f)
                
                # Add the file path itself
                metadata["filepath"] = mfile
                metadata_list.append(metadata)
        except Exception as e:
            st.error(f"Error loading metadata file {mfile}: {e}")
    
    # Sort the metadata list by timestamp in descending order (newest first)
    if metadata_list:
        # Sort by timestamp (assuming format YYYYmmdd_HHMMSS)
        metadata_list.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Print information about the most recent dataset for debugging
        st.sidebar.info(f"Most recent dataset: {metadata_list[0]['timestamp']} - {len(metadata_list[0].get('subjects', []))} subjects, {metadata_list[0].get('num_documents', 0)} documents")
    
    return metadata_list


def load_topic_data(metadata: Dict) -> Dict:
    """
    Load all topic data files referenced in the metadata.
    
    Args:
        metadata: Metadata dictionary with file references
    
    Returns:
        Dictionary with loaded data
    """
    data = {"metadata": metadata}
    timestamp = metadata["timestamp"]
    
    # Load topic info
    try:
        topic_info_path = TOPIC_DIR / metadata["file_references"]["topic_info"]
        data["topic_info"] = pd.read_csv(topic_info_path)
    except Exception as e:
        st.warning(f"Could not load topic info: {e}")
    
    # Load topic keywords
    try:
        keywords_path = TOPIC_DIR / metadata["file_references"]["topic_keywords"]
        with open(keywords_path, "r") as f:
            data["topic_keywords"] = json.load(f)
    except Exception as e:
        st.warning(f"Could not load topic keywords: {e}")
    
    # Load document topics
    try:
        doc_topics_path = TOPIC_DIR / metadata["file_references"]["document_topics"]
        data["document_topics"] = pd.read_csv(doc_topics_path)
        # Convert published_date to datetime if it's not already
        if "published_date" in data["document_topics"].columns:
            data["document_topics"]["published_date"] = pd.to_datetime(
                data["document_topics"]["published_date"]
            )
    except Exception as e:
        st.warning(f"Could not load document topics: {e}")
    
    # Load topic trends
    try:
        trends_path = TOPIC_DIR / metadata["file_references"]["topic_trends"]
        with open(trends_path, "r") as f:
            data["topic_trends"] = json.load(f)
    except Exception as e:
        st.warning(f"Could not load topic trends: {e}")
    
    # Load hierarchical topics if available
    hierarchy_path = TOPIC_DIR / f"hierarchical_topics_{timestamp}.json"
    if hierarchy_path.exists():
        try:
            with open(hierarchy_path, "r") as f:
                data["hierarchical_topics"] = json.load(f)
        except Exception as e:
            st.warning(f"Could not load hierarchical topics: {e}")
    
    # Find and load any available visualizations
    plot_files = glob.glob(str(PLOTS_DIR / f"*_{timestamp}.html"))
    data["visualizations"] = {
        os.path.basename(f).split("_")[0]: f for f in plot_files
    }
    
    wordcloud_files = glob.glob(str(PLOTS_DIR / f"wordcloud_topic_*_{timestamp}.png"))
    data["wordclouds"] = {
        int(os.path.basename(f).split("_")[2]): f for f in wordcloud_files
    }
    
    return data


def create_sample_data():
    """Create sample data if no real data is available yet"""
    st.info("No topic analysis data found. Creating sample data for demonstration.")
    
    # Create sample directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sample metadata
    metadata = {
        "timestamp": timestamp,
        "num_documents": 26741,
        "num_topics": 290,
        "subjects": ["math.AG", "math.AT", "math.RT", "math.SG"],
        "year_range": [2019, 2023],
        "file_references": {
            "topic_info": f"topic_info_{timestamp}.csv",
            "topic_keywords": f"topic_keywords_{timestamp}.json",
            "document_topics": f"document_topics_{timestamp}.csv",
            "topic_trends": f"topic_trends_{timestamp}.json",
        }
    }
    
    # Save sample metadata
    with open(TOPIC_DIR / f"metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f)
    
    # Create sample topic info
    topic_info = pd.DataFrame({
        "Topic": range(1, 11),
        "Count": [500, 450, 400, 350, 300, 250, 200, 150, 100, 50],
        "Name": [f"Topic_{i}_sample_topic" for i in range(1, 11)]
    })
    topic_info.to_csv(TOPIC_DIR / f"topic_info_{timestamp}.csv", index=False)
    
    # Create sample keywords
    keywords = {}
    for i in range(1, 11):
        keywords[str(i)] = [
            ["algebra", 0.9],
            ["geometry", 0.8],
            ["topology", 0.7],
            ["manifold", 0.6],
            ["group", 0.5]
        ]
    
    with open(TOPIC_DIR / f"topic_keywords_{timestamp}.json", "w") as f:
        json.dump(keywords, f)
    
    # Create sample document topics
    doc_topics = pd.DataFrame({
        "id": [f"sample_{i}" for i in range(1, 101)],
        "title": [f"Sample Paper {i}" for i in range(1, 101)],
        "topic": np.random.randint(1, 11, 100),
        "published_date": pd.date_range(start="2019-01-01", end="2023-12-31", periods=100).strftime("%Y-%m-%d").tolist(),
        "primary_category": np.random.choice(["math.AG", "math.AT", "math.RT", "math.SG"], 100)
    })
    doc_topics.to_csv(TOPIC_DIR / f"document_topics_{timestamp}.csv", index=False)
    
    # Create sample topic trends
    topic_trends = {
        "topics_over_time": [],
        "trends": {},
        "emerging_topics": [],
        "declining_topics": []
    }
    
    # Generate sample data for each topic over time
    periods = ["2019-Q1", "2019-Q2", "2019-Q3", "2019-Q4", 
               "2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4",
               "2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4", 
               "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
               "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"]
    
    for topic_id in range(1, 11):
        # Generate a trend line with some randomness
        if topic_id <= 5:  # Emerging topics
            base = np.linspace(0.05, 0.15, len(periods))  # Increasing trend
            growth = 80 + np.random.randint(-20, 20)
        else:  # Declining topics
            base = np.linspace(0.15, 0.05, len(periods))  # Decreasing trend
            growth = -50 + np.random.randint(-20, 20)
        
        noise = np.random.normal(0, 0.01, len(periods))
        frequencies = base + noise
        frequencies = np.maximum(frequencies, 0.01)  # Ensure no negative values
        
        # Add to topics_over_time
        for period, freq in zip(periods, frequencies):
            topic_trends["topics_over_time"].append({
                "Topic": topic_id,
                "Timestamp": period,
                "Frequency": float(freq)
            })
        
        # Add to trends
        topic_trends["trends"][str(topic_id)] = {
            "name": f"Topic_{topic_id}_sample_topic",
            "frequency": frequencies.tolist(),
            "timestamps": periods,
            "growth_absolute": float(frequencies[-1] - frequencies[0]),
            "growth_relative": float(growth),
            "trend_slope": float((frequencies[-1] - frequencies[0]) / len(periods)),
            "normalized_slope": float((frequencies[-1] - frequencies[0]) / np.mean(frequencies))
        }
        
        # Add to emerging or declining topics
        if topic_id <= 5:
            topic_trends["emerging_topics"].append({
                "id": topic_id,
                "name": f"Topic_{topic_id}_sample_topic",
                "frequency": frequencies.tolist(),
                "timestamps": periods,
                "growth_relative": float(growth)
            })
        else:
            topic_trends["declining_topics"].append({
                "id": topic_id,
                "name": f"Topic_{topic_id}_sample_topic",
                "frequency": frequencies.tolist(),
                "timestamps": periods,
                "growth_relative": float(growth)
            })
    
    with open(TOPIC_DIR / f"topic_trends_{timestamp}.json", "w") as f:
        json.dump(topic_trends, f)
    
    return metadata


def display_header():
    """Display dashboard header and description."""
    st.markdown('<p class="main-header">Math Research Compass 🧭</p>', unsafe_allow_html=True)
    st.markdown(
        """
        An interactive dashboard exploring research trends, emerging topics, and collaboration structures
        across mathematical subfields on arXiv. This dashboard visualizes topic modeling results from recent
        mathematical preprints to identify research patterns and emerging areas of interest.
        """
    )
    
    with st.expander("About this project"):
        st.markdown(
            """
            ### Project Details
            
            Math Research Compass is a data science project designed to provide insights into the recent landscape of mathematical research published on arXiv.
            
            #### Key Features:
            * **Topic Modeling**: Uses advanced NLP (BERTopic) to identify research themes and their hierarchical relationships
            * **Temporal Analysis**: Tracks how research interests evolve over time
            * **Emerging Topics**: Identifies growing and declining research areas
            * **Collaboration Analysis**: Examines co-authorship patterns (when available)
            
            #### How It Works:
            The dashboard analyzes metadata from arXiv preprints in selected mathematical subfields using
            natural language processing to identify coherent research topics and track their evolution over time.
            Topics are extracted using a combination of transformer-based embeddings (Sentence-BERT) and clustering (HDBSCAN).
            
            #### Data Scope:
            Currently tracking papers from math.AG (Algebraic Geometry), math.AT (Algebraic Topology),
            math.RT (Representation Theory), and math.SG (Symplectic Geometry).
            """
        )


def display_sidebar(metadata_list: List[Dict]) -> Dict:
    """
    Display sidebar with dataset selection and filters.
    
    Args:
        metadata_list: List of available metadata files
    
    Returns:
        Selected metadata dictionary
    """
    st.sidebar.title("Navigation")
    
    # Dataset selection if multiple are available
    selected_metadata = None
    if metadata_list:
        if len(metadata_list) > 1:
            # Create options for the selectbox
            options = []
            for meta in metadata_list:
                year_range = f"{meta['year_range'][0]}-{meta['year_range'][1]}"
                subjects = ", ".join(meta["subjects"])
                timestamp = datetime.strptime(meta["timestamp"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
                option_text = f"{subjects} | {year_range} | {timestamp}"
                options.append((meta, option_text))
            
            selected_option = st.sidebar.selectbox(
                "Select Dataset:",
                options,
                format_func=lambda x: x[1],
                key="dataset_selector"
            )
            selected_metadata = selected_option[0]
        else:
            selected_metadata = metadata_list[0]
            # Show dataset info
            year_range = f"{selected_metadata['year_range'][0]}-{selected_metadata['year_range'][1]}"
            subjects = ", ".join(selected_metadata["subjects"])
            st.sidebar.info(f"Dataset: {subjects}, {year_range}")
    else:
        st.sidebar.error("No topic analysis data found. Please run the topic_trends_analyzer.py script first.")
    
    # Navigation
    st.sidebar.markdown("## Sections")

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"

    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "Topic Explorer", "Temporal Trends", "Emerging Topics", "Papers by Topic", "Collaboration Network"],
        key="navigation",
        index=["Overview", "Topic Explorer", "Temporal Trends", "Emerging Topics", "Papers by Topic", "Collaboration Network"].index(st.session_state.current_page)
    )
    st.session_state.current_page = page
    
    # Additional filters based on selected page
    st.sidebar.markdown("## Filters")
    
    # Create filter states in session state if they don't exist
    if "filters" not in st.session_state:
        st.session_state.filters = {
            "min_topic_size": 15,
            "max_topics": 10,
            "selected_topics": [],
            "selected_subjects": [],
            "date_range": None,
        }
    
    # Common filters
    if selected_metadata:
        # Topic size filter
        st.session_state.filters["min_topic_size"] = st.sidebar.slider(
            "Min Topic Size:", 
            min_value=5, 
            max_value=50, 
            value=st.session_state.filters["min_topic_size"],
            step=5,
            key="min_topic_size_filter"
        )
        
        # Number of topics to display
        st.session_state.filters["max_topics"] = st.sidebar.slider(
            "Max Topics to Display:", 
            min_value=5, 
            max_value=20, 
            value=st.session_state.filters["max_topics"],
            step=5,
            key="max_topics_filter"
        )
        
        # Subject filter (if multiple subjects are in the data)
        if len(selected_metadata["subjects"]) > 1:
            # Define the target categories of interest
            target_categories = ["math.AG", "math.AT", "math.RT", "math.SG"]
    
            # Filter the metadata subjects to only include target categories
            available_categories = [subject for subject in selected_metadata["subjects"] 
                                    if subject in target_categories]
    
            selected_subjects = st.sidebar.multiselect(
                "Filter by Subject:",
                options=available_categories,
                default=st.session_state.filters["selected_subjects"] or available_categories,
                key="subject_filter"
            )
            st.session_state.filters["selected_subjects"] = selected_subjects
    
    # Add button for applying filters
    if st.sidebar.button("Apply Filters", key="apply_filters_button"):
        apply_filters()

    # Link to GitHub
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[GitHub Repository](https://github.com/brian-hepler-phd/MathResearchCompass)"
    )
    
    return selected_metadata


def display_overview(data: Dict):
    """Display overview page with key metrics and visualizations."""
    st.markdown('<p class="sub-header">Overview</p>', unsafe_allow_html=True)
    
    # Display summary statistics
    if "document_topics" in data and "topic_info" in data:
        col1, col2, col3, col4 = st.columns(4)
        
        doc_topics = data["document_topics"]
        topic_info = data["topic_info"]
        
        total_papers = len(doc_topics)
        num_topics = len(topic_info[topic_info["Topic"] != -1])
        time_span = f"{data['metadata']['year_range'][0]}-{data['metadata']['year_range'][1]}"
        
        # Calculate % of papers assigned to meaningful topics (not outliers)
        papers_with_topics = len(doc_topics[doc_topics["topic"] != -1])
        coverage_pct = round(papers_with_topics / total_papers * 100, 1)
        
        col1.metric("Total Papers", total_papers)
        col2.metric("Discovered Topics", num_topics)
        col3.metric("Topic Coverage", f"{coverage_pct}%")
        col4.metric("Time Span", time_span)
        
        # Display top topics
        st.markdown("### 📊 Top Research Topics")
        
        top_topics_df = topic_info[topic_info["Topic"] != -1].head(10)
        
        # Create a bar chart of top topics
        fig = px.bar(
            top_topics_df,
            x="Count",
            y="Name",
            orientation="h",
            labels={"Count": "Number of Papers", "Name": "Topic"},
            title="Most Common Research Topics",
            color="Count",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Display topic distribution visualization if available
        if "visualizations" in data and "topic_distribution" in data["visualizations"]:
            st.markdown("### 🔍 Topic Distribution")
            with open(data["visualizations"]["topic_distribution"], "r") as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=600)
    
    # Display hierarchical topics if available
    if "visualizations" in data and "topic_hierarchy" in data["visualizations"]:
        st.markdown("### 🌳 Topic Hierarchy")
        st.markdown(
            """
            This visualization shows how research topics relate to each other hierarchically.
            Topics that are closer together or connected share more semantic similarities.
            """
        )
        with open(data["visualizations"]["topic_hierarchy"], "r") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=700)
    
    # Display topic prevalence over time if available
    if "visualizations" in data and "topic_prevalence" in data["visualizations"]:
        st.markdown("### 📈 Topic Prevalence Over Time")
        st.markdown(
            """
            This chart shows how the prevalence of different research topics 
            has changed over the analyzed time period.
            """
        )
        with open(data["visualizations"]["topic_prevalence"], "r") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=600)


def display_topic_explorer(data: Dict):
    """Display topic explorer page with detailed topic information."""
    st.markdown('<p class="sub-header">Topic Explorer</p>', unsafe_allow_html=True)
    
    if "topic_info" in data and "topic_keywords" in data:
        topic_info = data["topic_info"]
        topic_keywords = data["topic_keywords"]
        
        # Filter out the outlier topic (-1)
        filtered_topics = topic_info[topic_info["Topic"] != -1]
        
        # Get list of topics
        topics = filtered_topics["Topic"].astype(str).tolist()
        topic_names = filtered_topics["Name"].tolist()
        
        # Create a selectbox with topic names
        topic_options = [f"Topic {t}: {n}" for t, n in zip(topics, topic_names)]
        selected_topic_idx = st.selectbox("Select a topic to explore:", range(len(topic_options)), format_func=lambda i: topic_options[i], key="topic_explorer_selector")
        
        selected_topic_id = int(topics[selected_topic_idx])
        
        # Display topic details
        st.markdown(f"### Topic {selected_topic_id}: {topic_names[selected_topic_idx]}")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Display top keywords
            st.markdown("#### Top Keywords")
            
            if str(selected_topic_id) in topic_keywords:
                keywords = topic_keywords[str(selected_topic_id)]
            else:
                keywords = topic_keywords.get(selected_topic_id, [])
            
            # Create a dataframe for better display
            if keywords:
                df_keywords = pd.DataFrame(keywords, columns=["Term", "Weight"])
                
                # Create a horizontal bar chart
                fig = px.bar(
                    df_keywords.head(15),
                    x="Weight",
                    y="Term",
                    orientation="h",
                    title="Top Terms by Relevance",
                    color="Weight",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No keyword data available for this topic.")
        
        with col2:
            # Display word cloud if available
            if "wordclouds" in data and selected_topic_id in data["wordclouds"]:
                st.markdown("#### Word Cloud")
                image_path = data["wordclouds"][selected_topic_id]
                image = Image.open(image_path)
                st.image(image, use_column_width=True)
            
            # Display topic size
            topic_size = filtered_topics[filtered_topics["Topic"] == selected_topic_id]["Count"].iloc[0]
            st.markdown(f"#### Topic Size: {topic_size} papers")
            
            # If temporal data is available, show trend
            if "topic_trends" in data and "trends" in data["topic_trends"]:
                trends = data["topic_trends"]["trends"]
                if str(selected_topic_id) in trends:
                    trend_data = trends[str(selected_topic_id)]
                    growth = trend_data.get("growth_relative", 0)
                    
                    trend_class = "trend-up" if growth > 0 else "trend-down"
                    trend_icon = "📈" if growth > 0 else "📉"
                    
                    st.markdown(
                        f"#### Trend: <span class='{trend_class}'>{trend_icon} {growth:.1f}% growth</span>", 
                        unsafe_allow_html=True
                    )
        
        # Display representative papers
        if "document_topics" in data:
            st.markdown("### Representative Papers")
            doc_topics = data["document_topics"]
            topic_papers = doc_topics[doc_topics["topic"] == selected_topic_id].sort_values(
                by="published_date", ascending=False
            )
            
            if not topic_papers.empty:
                for i, (_, paper) in enumerate(topic_papers.head(5).iterrows()):
                    with st.expander(paper["title"], key=f"paper_expander_{i}"):
                        st.markdown(f"**ID**: {paper['id']}")
                        st.markdown(f"**Publication Date**: {paper['published_date']}")
                        st.markdown(f"**Category**: {paper['primary_category']}")
                        st.markdown(f"**arXiv Link**: [View on arXiv](https://arxiv.org/abs/{paper['id']})")
            else:
                st.info("No papers found for this topic.")


def display_temporal_trends(data: Dict):
    """Display temporal trends page with topic evolution over time."""
    st.markdown('<p class="sub-header">Temporal Trends</p>', unsafe_allow_html=True)
    
    if "topic_trends" not in data:
        st.warning("Temporal trend data not available.")
        return
    
    # Get trend data
    trends = data["topic_trends"]
    
    # Display topics over time
    st.markdown("### Topic Evolution Over Time")
    
    if "topics_over_time" in trends and "topic_info" in data:
        topic_info = data["topic_info"]
        
        # Get topics for the dropdown
        filtered_topics = topic_info[topic_info["Topic"] != -1]
        
        # Get list of topics
        topics = filtered_topics["Topic"].astype(str).tolist()
        topic_names = filtered_topics["Name"].tolist()
        topic_counts = filtered_topics["Count"].tolist()
        
        # Create a multiselect with topic names
        topic_options = [f"Topic {t}: {n} ({c} papers)" for t, n, c in zip(topics, topic_names, topic_counts)]
        
        # Calculate the top topics by size
        top_topic_indices = filtered_topics["Count"].nlargest(min(5, len(filtered_topics))).index
        default_topics = [i for i, idx in enumerate(filtered_topics.index) if idx in top_topic_indices]
        
        selected_topic_indices = st.multiselect(
            "Select topics to display:",
            range(len(topic_options)),
            default=default_topics,
            format_func=lambda i: topic_options[i],
            key="temporal_trends_topic_selector"
        )
        
        if selected_topic_indices:
            selected_topic_ids = [int(topics[i]) for i in selected_topic_indices]
            
            # Create a dataframe for plotting from topics_over_time
            tot_df = pd.DataFrame(trends["topics_over_time"])
            filtered_tot = tot_df[tot_df["Topic"].isin(selected_topic_ids)]
            
            if not filtered_tot.empty:
                # Create a time series plot
                fig = px.line(
                    filtered_tot,
                    x="Timestamp",
                    y="Frequency",
                    color="Topic",
                    labels={"Timestamp": "Time Period", "Frequency": "Topic Frequency", "Topic": "Topic ID"},
                    title="Topic Frequency Over Time",
                    markers=True
                )
                
                # Replace topic IDs with short names in legend
                for topic_id in selected_topic_ids:
                    topic_name = topic_info[topic_info["Topic"] == topic_id]["Name"].iloc[0].split("_")[0]
                    fig.update_traces(
                        name=f"Topic {topic_id}: {topic_name}",
                        selector=dict(name=str(topic_id))
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No temporal data available for the selected topics.")
    
    # Display growth rate comparison
    st.markdown("### Topic Growth Comparison")
    
    if "trends" in trends and "topic_info" in data:
        topic_info = data["topic_info"]
        trend_data = trends["trends"]
        
        # Extract growth rates for all topics
        growth_data = []
        
        for topic_id, data in trend_data.items():
            topic_id = int(topic_id)
            if topic_id == -1:  # Skip outlier topic
                continue
                
            # Get topic info
            topic_row = topic_info[topic_info["Topic"] == topic_id]
            if topic_row.empty:
                continue
                
            topic_name = topic_row["Name"].iloc[0]
            topic_size = topic_row["Count"].iloc[0]
            
            growth_data.append({
                "Topic": topic_id,
                "Name": topic_name,
                "Count": topic_size,
                "Growth": data.get("growth_relative", 0)
            })
        
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            
            # Sort by growth rate
            growth_df = growth_df.sort_values("Growth", ascending=False)
            
            # Create a horizontal bar chart
            fig = px.bar(
                growth_df,
                x="Growth",
                y="Topic",
                orientation="h",
                labels={"Growth": "Relative Growth (%)", "Topic": "Topic ID"},
                title="Topics by Growth Rate",
                color="Growth",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                hover_data=["Name", "Count"]
            )
            
            # Format y-axis to show topic IDs and names
            fig.update_layout(
                yaxis={"categoryorder": "array", "categoryarray": growth_df["Topic"].tolist()},
                yaxis_tickformat=",d"
            )
            
            # Update hover template
            fig.update_traces(
                hovertemplate="<b>Topic %{y}</b><br>%{customdata[0]}<br>Growth: %{x:.1f}%<br>Papers: %{customdata[1]}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No growth data available for topics.")


def display_emerging_topics(data: Dict):
    """Display emerging and declining topics page."""
    st.markdown('<p class="sub-header">Emerging & Declining Research Areas</p>', unsafe_allow_html=True)
    
    if "topic_trends" not in data:
        st.warning("Trend data not available.")
        return
    
    trends = data["topic_trends"]
    
    # Display emerging topics
    st.markdown("### 🚀 Emerging Research Topics")
    
    if "emerging_topics" in trends and "topic_keywords" in data:
        emerging = trends["emerging_topics"]
        topic_keywords = data["topic_keywords"]
        
        if emerging:
            # Create columns for each emerging topic
            cols = st.columns(min(3, len(emerging)))
            
            for i, (col, topic) in enumerate(zip(cols, emerging[:3])):
                with col:
                    topic_id = topic["id"]
                    # Split with underscore and get all parts after the first one
                    topic_name_parts = topic["name"].split("_")
                    topic_name = " ".join(topic_name_parts[1:]) if len(topic_name_parts) > 1 else topic_name_parts[0]
                    growth = topic["growth_relative"]
                    
                    # Display card for the topic
                    st.markdown(f'<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"#### Topic {topic_id}: {topic_name}")
                    st.markdown(f"**Growth Rate**: <span class='trend-up'>+{growth:.1f}%</span>", unsafe_allow_html=True)
                    
                    # Show top keywords
                    if str(topic_id) in topic_keywords:
                        keywords = topic_keywords[str(topic_id)]
                    else:
                        keywords = topic_keywords.get(topic_id, [])
                    
                    if keywords:
                        st.markdown("**Top Keywords:**")
                        keyword_text = ", ".join([word for word, _ in keywords[:5]])
                        st.markdown(f"*{keyword_text}*")
                    
                    # Show trend visualization
                    timestamps = topic["timestamps"]
                    frequencies = topic["frequency"]
                    
                    df = pd.DataFrame({
                        "Timestamp": timestamps,
                        "Frequency": frequencies
                    })
                    
                    fig = px.line(
                        df,
                        x="Timestamp",
                        y="Frequency",
                        markers=True,
                        title="Trend"
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=200)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Show full emerging topics visualization if available
            if "visualizations" in data and "emerging_topics" in data["visualizations"]:
                st.markdown("### Detailed Trends for Emerging Topics")
                with open(data["visualizations"]["emerging_topics"], "r") as f:
                    html_content = f.read()
                    st.components.v1.html(html_content, height=500)
        else:
            st.info("No emerging topics identified in the dataset.")
    
    # Display declining topics
    st.markdown("### 📉 Declining Research Areas")
    
    if "declining_topics" in trends and "topic_keywords" in data:
        declining = trends["declining_topics"]
        topic_keywords = data["topic_keywords"]
        
        if declining:
            # Create columns for each declining topic
            cols = st.columns(min(3, len(declining)))
            
            for i, (col, topic) in enumerate(zip(cols, declining[:3])):
                with col:
                    topic_id = topic["id"]
                    # Split with underscore and get all parts after the first one
                    topic_name_parts = topic["name"].split("_")
                    topic_name = " ".join(topic_name_parts[1:]) if len(topic_name_parts) > 1 else topic_name_parts[0]
                    growth = topic["growth_relative"]
                    
                    # Display card for the topic
                    st.markdown(f'<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"#### Topic {topic_id}: {topic_name}")
                    st.markdown(f"**Growth Rate**: <span class='trend-down'>{growth:.1f}%</span>", unsafe_allow_html=True)
                    
                    # Show top keywords
                    if str(topic_id) in topic_keywords:
                        keywords = topic_keywords[str(topic_id)]
                    else:
                        keywords = topic_keywords.get(topic_id, [])
                    
                    if keywords:
                        st.markdown("**Top Keywords:**")
                        keyword_text = ", ".join([word for word, _ in keywords[:5]])
                        st.markdown(f"*{keyword_text}*")
                    
                    # Show trend visualization
                    timestamps = topic["timestamps"]
                    frequencies = topic["frequency"]
                    
                    df = pd.DataFrame({
                        "Timestamp": timestamps,
                        "Frequency": frequencies
                    })
                    
                    fig = px.line(
                        df,
                        x="Timestamp",
                        y="Frequency",
                        markers=True,
                        title="Trend"
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=200)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Show full declining topics visualization if available
            if "visualizations" in data and "declining_topics" in data["visualizations"]:
                st.markdown("### Detailed Trends for Declining Topics")
                with open(data["visualizations"]["declining_topics"], "r") as f:
                    html_content = f.read()
                    st.components.v1.html(html_content, height=500)
        else:
            st.info("No declining topics identified in the dataset.")


def display_papers_by_topic(data: Dict):
    """Display papers by topic page with paper listings."""
    st.markdown('<p class="sub-header">Papers by Topic</p>', unsafe_allow_html=True)
    
    if "document_topics" not in data:
        st.warning("Document topic data not available.")
        return
    
    doc_topics = data["document_topics"]
    
    # Create filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique primary categories from the data
        categories = sorted(doc_topics["primary_category"].unique())
        
        selected_categories = st.multiselect(
            "Filter by Category:",
            options=categories,
            default=categories[:4] if len(categories) > 4 else categories,
            key="papers_category_filter"
        )
    
    with col2:
        # Topic filter
        if "topic_info" in data:
            topic_info = data["topic_info"]
            filtered_topics = topic_info[topic_info["Topic"] != -1]
            
            # Sort topics by size
            sorted_topics = filtered_topics.sort_values("Count", ascending=False)
            
            # Create topic options
            topic_options = [
                (row["Topic"], f"Topic {row['Topic']}: {row['Name']} ({row['Count']} papers)")
                for _, row in sorted_topics.iterrows()
            ]
            
            selected_topic_option = st.selectbox(
                "Select Topic:",
                options=topic_options,
                format_func=lambda x: x[1],
                key="papers_topic_filter"
            )
            
            selected_topic = selected_topic_option[0]
        else:
            # If topic info is not available, use a simple numeric input
            selected_topic = st.number_input("Enter Topic ID:", min_value=-1, value=0, key="topic_id_input")
    
    # Filter papers
    filtered_papers = doc_topics.copy()
    
    if selected_categories:
        filtered_papers = filtered_papers[filtered_papers["primary_category"].isin(selected_categories)]
    
    filtered_papers = filtered_papers[filtered_papers["topic"] == selected_topic]
    
    # Display paper count
    st.markdown(f"### Found {len(filtered_papers)} Papers")
    
    # Sort papers by date (newest first)
    sorted_papers = filtered_papers.sort_values("published_date", ascending=False)
    
    # Display papers
    for i, (_, paper) in enumerate(sorted_papers.iterrows()):
        if i >= 50:  # Limit to 50 papers for performance
            st.info(f"Showing 50 of {len(sorted_papers)} papers. Please add more filters to narrow down results.")
            break
            
        # Create a card for each paper
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        
        # Paper title with link
        arxiv_id = paper["id"]
        paper_url = f"https://arxiv.org/abs/{arxiv_id}"
        st.markdown(f"#### [{paper['title']}]({paper_url})")
        
        # Paper metadata
        st.markdown(f"**ID**: {arxiv_id} | **Date**: {paper['published_date']} | **Category**: {paper['primary_category']}")
        
        # Links
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"[PDF](https://arxiv.org/pdf/{arxiv_id}.pdf) | [Abstract](https://arxiv.org/abs/{arxiv_id})")
        
        st.markdown('</div>', unsafe_allow_html=True)


def display_collaboration_network(data: Dict):
    """Display collaboration network with co-authorship visualization."""
    st.markdown('<p class="sub-header">Collaboration Network</p>', unsafe_allow_html=True)
    
    # Check if document data is available
    if "document_topics" not in data:
        st.warning("Document data not available for network analysis.")
        return
    
    doc_topics = data["document_topics"]
    papers = None
    
    # If document topics doesn't contain author information, we need the original data
    if "id" in doc_topics.columns:
        # Load original papers data to get author information
        try:
            # Try to get subjects from metadata
            subjects = data["metadata"]["subjects"]
            
            # Load raw data for all subjects
            dfs = []
            for subject in subjects:
                df = load_subject_data(subject)
                if not df.empty:
                    dfs.append(df)
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                
                # Make sure authors column is parsed correctly
                if "authors" in df.columns:
                    if isinstance(df["authors"].iloc[0], str):
                        df["authors"] = df["authors"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                
                # Merge with document topics to get author information
                papers = pd.merge(
                    doc_topics[["id", "topic"]],
                    df[["id", "authors", "primary_category"]],
                    on="id",
                    how="inner"
                )
            else:
                st.error("Could not load paper data with author information")
                return
                
        except Exception as e:
            st.error(f"Could not load paper author data: {e}")
            st.info("To enable collaboration network visualization, please ensure your data includes author information.")
            return
    
    if papers is None or papers.empty:
        st.warning("No author data available for analysis.")
        return
    
    # Allow filtering by topic
    if "topic_info" in data:
        topic_info = data["topic_info"]
        
        # Get filtered topics (excluding outliers)
        filtered_topics = topic_info[topic_info["Topic"] != -1]
        
        # Sort by count
        sorted_topics = filtered_topics.sort_values("Count", ascending=False)
        
        # Create options
        topic_options = [
            (row["Topic"], f"Topic {row['Topic']}: {row['Name']} ({row['Count']} papers)")
            for _, row in sorted_topics.iterrows()
        ]
        
        # Add "All Topics" option
        topic_options = [(None, "All Topics")] + topic_options
        
        selected_topic_option = st.selectbox(
            "Select Topic for Collaboration Analysis:",
            options=topic_options,
            format_func=lambda x: x[1],
            key="collab_topic_selector"
        )
        
        selected_topic = selected_topic_option[0]
    else:
        selected_topic = None
    
    # Filter papers by selected topic if needed
    if selected_topic is not None:
        filtered_papers = papers[papers["topic"] == selected_topic]
    else:
        filtered_papers = papers
    
    # Set a limit on the number of papers to process for performance
    max_papers = 1000
    if len(filtered_papers) > max_papers:
        st.warning(f"Processing only the first {max_papers} papers for network visualization due to performance constraints.")
        filtered_papers = filtered_papers.head(max_papers)
    
    # Build co-authorship network
    try:
        # Create graph
        G = nx.Graph()
        
        # Add edges for co-authors
        for _, paper in filtered_papers.iterrows():
            authors = paper["authors"]
            
            if not authors or len(authors) <= 1:
                continue
                
            # Add edges between all pairs of authors
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    author1 = authors[i]
                    author2 = authors[j]
                    
                    # Skip if either author is empty
                    if not author1 or not author2:
                        continue
                    
                    # Add or increment edge weight
                    if G.has_edge(author1, author2):
                        G[author1][author2]["weight"] += 1
                    else:
                        G.add_edge(author1, author2, weight=1)
        
        # Display network statistics
        st.markdown("### Collaboration Network Statistics")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Authors", len(G.nodes()))
        col2.metric("Collaborations", len(G.edges()))
        
        # Calculate connected components
        connected_components = list(nx.connected_components(G))
        col3.metric("Research Groups", len(connected_components))
        
        # If there are no nodes, stop here
        if len(G.nodes()) == 0:
            st.info("No collaboration data available for the selected topic.")
            return
        
        # Get largest connected component if it exists
        if connected_components:
            largest_cc = max(connected_components, key=len)
        
        # Calculate some network metrics
        st.markdown("### Network Insights")
        
        # Most collaborative authors (by number of collaborations)
        degree_dict = dict(G.degree())
        top_authors = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        df_top_authors = pd.DataFrame(top_authors, columns=["Author", "Collaborations"])
        
        fig = px.bar(
            df_top_authors,
            x="Collaborations",
            y="Author",
            orientation="h",
            title="Most Collaborative Authors",
            labels={"Collaborations": "Number of Collaborators", "Author": "Author Name"}
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a subgraph of only the most collaborative authors for visualization
        # This improves performance and readability of the network
        top_author_names = [author for author, _ in top_authors]
        
        # Get collaborators of top authors (extend the network slightly)
        expanded_author_set = set(top_author_names)
        for author in top_author_names:
            expanded_author_set.update(G.neighbors(author))
        
        # Limit the total number of authors for visualization
        if len(expanded_author_set) > 50:
            # Keep all top authors and sample their collaborators
            collaborators = expanded_author_set - set(top_author_names)
            sample_size = 50 - len(top_author_names)
            sampled_collaborators = list(collaborators)[:sample_size]
            vis_authors = set(top_author_names) | set(sampled_collaborators)
        else:
            vis_authors = expanded_author_set
        
        vis_graph = G.subgraph(vis_authors)
        
        # Generate positions for nodes
        try:
            # Use a force-directed layout
            pos = nx.spring_layout(vis_graph, k=0.15, iterations=50)
            
            # Convert to a format usable with Plotly
            edge_traces = []
            node_traces = []
            
            # Create edges
            for edge in vis_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = vis_graph[edge[0]][edge[1]].get("weight", 1)
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=weight, color="#888"),
                    hoverinfo="none",
                    mode="lines",
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            # Create nodes
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in vis_graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"Author: {node}<br>Collaborations: {vis_graph.degree(node)}")
                node_size.append(10 + 5 * vis_graph.degree(node))
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=node_text,
                marker=dict(
                    size=node_size,
                    color=[vis_graph.degree(node) for node in vis_graph.nodes()],
                    colorscale="Viridis",
                    colorbar=dict(title="Collaborations"),
                    line=dict(width=1)
                ),
                showlegend=False
            )
            node_traces.append(node_trace)
            
            # Create layout
            layout = go.Layout(
                title="Collaboration Network Visualization",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            # Create figure
            fig = go.Figure(data=edge_traces + node_traces, layout=layout)
            
            # Display network visualization
            st.markdown("### Collaboration Network Visualization")
            st.markdown(
                """
                This network shows the collaborative relationships between authors.
                Larger nodes represent authors with more collaborators, and thicker lines
                indicate stronger collaborative relationships (more papers together).
                """
            )
            
            # Use plotly_events if available, otherwise use regular plotly chart
            if PLOTLY_EVENTS_AVAILABLE:
                selected_points = plotly_events(fig, override_height=600)
                
                # Display information for selected author if any
                if selected_points:
                    point = selected_points[0]
                    idx = point["pointIndex"]
                    author = list(vis_graph.nodes())[idx]
                    
                    st.markdown(f"### Selected Author: {author}")
                    
                    # Get collaborators
                    collaborators = list(vis_graph.neighbors(author))
                    
                    # Get papers with this author
                    author_papers = filtered_papers[filtered_papers["authors"].apply(lambda x: author in x)]
                    
                    st.markdown(f"**Number of papers:** {len(author_papers)}")
                    st.markdown(f"**Number of collaborators:** {len(collaborators)}")
                    
                    if collaborators:
                        st.markdown("**Top collaborators:**")
                        collab_counts = []
                        for collab in collaborators:
                            collab_count = vis_graph[author][collab].get("weight", 0)
                            collab_counts.append((collab, collab_count))
                        
                        collab_df = pd.DataFrame(
                            sorted(collab_counts, key=lambda x: x[1], reverse=True)[:5],
                            columns=["Collaborator", "Papers Together"]
                        )
                        st.table(collab_df)
                    
                    # Display recent papers
                    if not author_papers.empty:
                        st.markdown("**Recent papers:**")
                        for i, (_, paper) in enumerate(author_papers.sort_values("published_date", ascending=False).head(3).iterrows()):
                            paper_id = paper["id"]
                            st.markdown(f"- [{paper_id}](https://arxiv.org/abs/{paper_id})")
            else:
                st.plotly_chart(fig, use_container_width=True)
                st.info("Install streamlit-plotly-events for interactive network exploration: pip install streamlit-plotly-events")
        
        except Exception as e:
            st.error(f"Error generating network visualization: {e}")
            st.info("Try selecting a different topic with fewer authors for better visualization.")
    
    except Exception as e:
        st.error(f"Error building collaboration network: {e}")


def main():
    """Main function to run the dashboard."""
    # Display the header
    display_header()
    
    # Load available metadata files
    metadata_list = load_metadata_files()
    
    # If no metadata is available, show a message and create sample data if requested
    if not metadata_list:
        if "create_sample" not in st.session_state:
            st.session_state.create_sample = False
            
        create_sample = st.button("Create sample data for testing")
        if create_sample or st.session_state.create_sample:
            st.session_state.create_sample = True
            sample_metadata = create_sample_data()
            metadata_list = [sample_metadata]  # Set the metadata_list to include our sample
        else:
            st.warning("No topic analysis data found. Please run the topic analysis script first.")
            # Show sample command to run the analysis
            st.code("python topic_trends_analyzer.py --subjects math.AG math.AT --years 5", language="bash")
            return
    
    # Display sidebar and get selected dataset
    selected_metadata = display_sidebar(metadata_list)
    
    # Load data for selected metadata
    data = load_topic_data(selected_metadata)
    
    # Use the page from session state that was set in the sidebar
    page = st.session_state.current_page
    
    # Display the selected page
    if page == "Overview":
        display_overview(data)
    elif page == "Topic Explorer":
        display_topic_explorer(data)
    elif page == "Temporal Trends":
        display_temporal_trends(data)
    elif page == "Emerging Topics":
        display_emerging_topics(data)
    elif page == "Papers by Topic":
        display_papers_by_topic(data)
    elif page == "Collaboration Network":
        display_collaboration_network(data)


if __name__ == "__main__":
    main()