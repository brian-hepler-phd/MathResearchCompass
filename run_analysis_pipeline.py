#!/usr/bin/env python3
"""
Math Research Compass - Complete Analysis Pipeline
-------------------------------------------------
This script runs the entire analysis pipeline for the Math Research Compass project:
1. Fetches data from arXiv for specified math categories
2. Processes and cleans the data
3. Runs topic modeling analysis
4. Launches the Streamlit dashboard

Usage:
    python run_analysis_pipeline.py [--fetch] [--analyze] [--dashboard]
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pipeline")

# Define paths
SCRIPT_DIR = Path(__file__).parent


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete Math Research Compass analysis pipeline"
    )
    parser.add_argument(
        "--fetch", 
        action="store_true",
        help="Fetch and process data from arXiv"
    )
    parser.add_argument(
        "--analyze", 
        action="store_true",
        help="Run topic modeling analysis"
    )
    parser.add_argument(
        "--dashboard", 
        action="store_true",
        help="Launch the Streamlit dashboard"
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
        "--time-interval", 
        type=str, 
        choices=["month", "quarter", "year"],
        default="quarter",
        help="Time interval for temporal analysis"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download of data even if it exists"
    )
    
    args = parser.parse_args()
    
    # If no specific actions are specified, run everything
    if not any([args.fetch, args.analyze, args.dashboard]):
        args.fetch = True
        args.analyze = True
        args.dashboard = True
    
    return args


def fetch_and_process_data(subjects, force_download=False):
    """
    Fetch and process data for the specified subjects.
    
    Args:
        subjects: List of arXiv subject categories
        force_download: Whether to force re-download even if data exists
    """
    logger.info(f"Starting data fetch and processing for subjects: {', '.join(subjects)}")
    
    # Check if fetch_data.py and process_data.py exist
    fetch_script = SCRIPT_DIR / "fetch_data.py"
    process_script = SCRIPT_DIR / "process_data.py"
    
    if not fetch_script.exists():
        logger.error(f"Could not find fetch_data.py script at {fetch_script}")
        return False
        
    if not process_script.exists():
        logger.error(f"Could not find process_data.py script at {process_script}")
        return False
    
    # Run fetch_data.py for each subject
    for subject in subjects:
        logger.info(f"Fetching data for subject: {subject}")
        
        cmd = [sys.executable, str(fetch_script), "--categories", subject]
        if force_download:
            cmd.append("--overwrite")
        
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error fetching data for {subject}: {e}")
            return False
    
    # Run process_data.py for each subject
    for subject in subjects:
        logger.info(f"Processing data for subject: {subject}")
        
        cmd = [sys.executable, str(process_script), "--categories", subject]
        if force_download:
            cmd.append("--overwrite")
        
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing data for {subject}: {e}")
            return False
    
    logger.info("Data fetch and processing completed successfully")
    return True


def run_topic_analysis(subjects, years, time_interval, force_download=False):
    """
    Run topic modeling analysis on the processed data.
    
    Args:
        subjects: List of arXiv subject categories
        years: Number of years of data to analyze
        time_interval: Time interval for temporal analysis
        force_download: Whether to force re-download of data
    """
    logger.info("Starting topic modeling analysis")
    
    # Find the topic_trends_analyzer.py script
    analyzer_script = SCRIPT_DIR / "topic_trends_analyzer.py"
    
    if not analyzer_script.exists():
        logger.error(f"Could not find topic_trends_analyzer.py script at {analyzer_script}")
        return False
    
    # Build command
    cmd = [
        sys.executable, 
        str(analyzer_script),
        "--subjects"
    ] + subjects + [
        "--years", str(years),
        "--time-interval", time_interval
    ]
    
    if force_download:
        cmd.append("--force-download")
    
    # Run the analysis
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Topic analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running topic analysis: {e}")
        return False


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard")
    
    # Find the app.py script
    app_script = SCRIPT_DIR / "app.py"
    
    if not app_script.exists():
        logger.error(f"Could not find Streamlit dashboard script at {app_script}")
        return False
    
    # Build command
    cmd = ["streamlit", "run", str(app_script)]
    
    # Launch the dashboard
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error launching dashboard: {e}")
        return False
    except FileNotFoundError:
        logger.error("Streamlit not found. Please install it with: pip install streamlit")
        return False


def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    
    if args.fetch:
        success = fetch_and_process_data(args.subjects, args.force)
        if not success:
            logger.error("Data fetch and processing failed, exiting")
            return
    
    if args.analyze:
        success = run_topic_analysis(args.subjects, args.years, args.time_interval, args.force)
        if not success:
            logger.error("Topic analysis failed, exiting")
            return
    
    if args.dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
