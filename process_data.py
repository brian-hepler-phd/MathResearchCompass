#!/usr/bin/env python3
"""
process_data.py - Clean and preprocess arXiv metadata

This script processes raw JSON files containing arXiv metadata,
cleans the data, and prepares it for analysis.

Usage:
    python process_data.py --categories math.AG math.AT
    python process_data.py --categories math.RT --overwrite
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("process_data")

# Constants
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process and clean arXiv papers metadata"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["math.AG", "math.AT", "math.RT", "math.SG"],
        help="arXiv categories to process (e.g., math.AG math.AT)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def load_raw_data(category, raw_dir=RAW_DIR):
    """
    Load raw data for a category.
    
    Args:
        category: arXiv category (e.g., math.AG)
        raw_dir: Directory containing raw data files
    
    Returns:
        List of dictionaries containing paper metadata
    """
    safe_category = category.replace(".", "_")
    input_file = raw_dir / f"{safe_category}.json"
    
    if not input_file.exists():
        logger.error(f"Raw data file {input_file} not found. Run fetch_data.py first.")
        return []
    
    with open(input_file, "r") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} entries from {input_file}")
    return data


def process_data(data):
    """
    Process and clean the raw data.
    
    Args:
        data: List of dictionaries containing paper metadata
    
    Returns:
        Cleaned pandas DataFrame
    """
    logger.info(f"Processing {len(data)} entries")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Basic cleaning
    if df.empty:
        logger.warning("Empty DataFrame after conversion")
        return df
    
    # Convert dates to datetime
    date_columns = ["published_date", "updated_date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Handle missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            logger.debug(f"Column {col} has {missing} missing values")
    
    # Extract year, month for easier analysis
    if "published_date" in df.columns:
        df["year"] = df["published_date"].dt.year
        df["month"] = df["published_date"].dt.month
        df["quarter"] = df["published_date"].dt.quarter
    
    # Ensure authors and categories are stored as strings (for CSV compatibility)
    list_columns = ["authors", "categories", "all_categories"]
    for col in list_columns:
        if col in df.columns:
            # Convert empty lists to empty strings
            df[col] = df[col].apply(lambda x: json.dumps(x) if x else "[]")
    
    # Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["id"])
    n_after = len(df)
    if n_before > n_after:
        logger.info(f"Removed {n_before - n_after} duplicate entries")
    
    # Only keep papers with primary category in math.*
    n_before = len(df)
    if "primary_category" in df.columns:
        df = df[df["primary_category"].str.startswith("math.")]
        n_after = len(df)
        if n_before > n_after:
            logger.info(f"Removed {n_before - n_after} entries with non-math primary categories")
    
    # Create additional fields for analysis
    df["text_for_nlp"] = df["title"] + " " + df["abstract"]
    
    logger.info(f"Processed data: {len(df)} entries remaining")
    return df


def save_processed_data(df, category, output_dir=CLEANED_DIR, overwrite=False):
    """
    Save processed data to file.
    
    Args:
        df: Processed DataFrame
        category: arXiv category
        output_dir: Directory to save data
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to saved file
    """
    safe_category = category.replace(".", "_")
    output_file = output_dir / f"{safe_category}.csv"
    
    # Check if file exists and overwrite is False
    if output_file.exists() and not overwrite:
        logger.info(f"File {output_file} already exists. Use --overwrite to replace.")
        return output_file
    
    # Save data
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(df)} entries to {output_file}")
    return output_file


def main():
    """Main execution function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process each category
    for category in args.categories:
        # Load raw data
        data = load_raw_data(category)
        
        if not data:
            continue
        
        # Process data
        df = process_data(data)
        
        if df.empty:
            logger.warning(f"No data to save for {category}")
            continue
        
        # Save processed data
        save_processed_data(df, category, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
