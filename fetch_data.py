#!/usr/bin/env python3
"""
fetch_data.py - Robust arXiv data fetcher with date chunking

This script downloads metadata for arXiv papers in specified mathematical categories
using the arXiv API via the arxiv Python package. It splits date ranges into smaller
chunks to improve reliability.

Usage:
    python fetch_data.py --categories math.AG --start_date 2020-01-01 --overwrite
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
import arxiv
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fetch_data")

# Constants
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fetch arXiv papers metadata for specified categories"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["math.AG", "math.AT", "math.RT", "math.SG"],
        help="arXiv categories to fetch (e.g., math.AG math.AT)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2019-01-01",
        help="Start date for papers (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for papers (YYYY-MM-DD), defaults to current date"
    )
    parser.add_argument(
        "--chunk_months",
        type=int,
        default=3,
        help="Number of months per chunk for date range"
    )
    parser.add_argument(
        "--max_results_per_chunk",
        type=int,
        default=2000,
        help="Maximum number of results to fetch per date chunk"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set end date to current date if not specified
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    return args


def date_chunks(start_date_str, end_date_str, months_per_chunk=3):
    """
    Split a date range into smaller chunks.
    
    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        months_per_chunk: Number of months per chunk
    
    Returns:
        List of (chunk_start, chunk_end) date tuples
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    chunks = []
    chunk_start = start_date
    
    while chunk_start <= end_date:
        # Calculate chunk end date (chunk_start + months_per_chunk months)
        year = chunk_start.year + ((chunk_start.month - 1 + months_per_chunk) // 12)
        month = ((chunk_start.month - 1 + months_per_chunk) % 12) + 1
        chunk_end = date(year, month, 1) - timedelta(days=1)
        
        # Ensure chunk_end is not after the overall end date
        if chunk_end > end_date:
            chunk_end = end_date
        
        chunks.append((chunk_start, chunk_end))
        
        # Move to next chunk
        chunk_start = chunk_end + timedelta(days=1)
    
    return chunks


def fetch_papers_for_chunk(category, start_date, end_date, max_results=2000, delay=3.0):
    """
    Fetch papers for a given category and date chunk.
    
    Args:
        category: arXiv category (e.g., "math.AG")
        start_date: Start date object
        end_date: End date object
        max_results: Maximum results per chunk
        delay: Delay between API calls
    
    Returns:
        List of dictionaries containing paper metadata
    """
    # Format dates for the query
    start_date_query = start_date.strftime("%Y%m%d")
    end_date_query = end_date.strftime("%Y%m%d")
    
    # Build query
    search_query = f'cat:{category} AND submittedDate:[{start_date_query} TO {end_date_query}]'
    
    logger.info(f"Fetching chunk: {start_date} to {end_date}")
    logger.info(f"Using query: {search_query}")
    
    # Setup arxiv client
    client = arxiv.Client(
        page_size=100,
        delay_seconds=delay,
        num_retries=10
    )
    
    # Create search object
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending
    )
    
    # Fetch results
    papers_data = []
    papers_found = 0
    papers_kept = 0
    
    try:
        # Get results with progress bar
        results_generator = client.results(search)
        
        for result in tqdm(results_generator, desc=f"Scanning {category} {start_date_query}-{end_date_query}"):
            papers_found += 1
            
            # Keep paper if category is in its categories list
            if category in result.categories:
                # Format paper data
                paper_data = {
                    'id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary.replace("\n", " "),
                    'categories': result.categories,
                    'primary_category': result.primary_category,
                    'all_categories': result.categories,
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'updated_date': result.updated.strftime('%Y-%m-%d'),
                    'pdf_url': result.pdf_url,
                    'comment': result.comment if hasattr(result, 'comment') else None
                }
                papers_data.append(paper_data)
                papers_kept += 1
        
        logger.info(f"Chunk {start_date} to {end_date}: Scanned {papers_found} papers, kept {papers_kept}")
    
    except Exception as e:
        logger.error(f"Error fetching chunk {start_date} to {end_date}: {e}")
    
    return papers_data


def fetch_papers(category, start_date_str, end_date_str, chunk_months=3, max_results_per_chunk=2000, delay=3.0):
    """
    Fetch papers for a given category by splitting date range into chunks.
    
    Args:
        category: arXiv category
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        chunk_months: Number of months per chunk
        max_results_per_chunk: Maximum results per chunk
        delay: Delay between API calls
    
    Returns:
        List of dictionaries containing paper metadata
    """
    logger.info(f"Fetching papers for {category} from {start_date_str} to {end_date_str}")
    
    # Generate date chunks
    chunks = date_chunks(start_date_str, end_date_str, chunk_months)
    logger.info(f"Split date range into {len(chunks)} chunks")
    
    # Fetch papers for each chunk
    all_papers = []
    
    for chunk_start, chunk_end in chunks:
        chunk_papers = fetch_papers_for_chunk(
            category=category,
            start_date=chunk_start,
            end_date=chunk_end,
            max_results=max_results_per_chunk,
            delay=delay
        )
        
        # Add to all papers
        all_papers.extend(chunk_papers)
        
        # Add extra delay between chunks
        time.sleep(delay * 2)
    
    # Remove duplicates (based on paper ID)
    unique_papers = {}
    for paper in all_papers:
        unique_papers[paper['id']] = paper
    
    unique_paper_list = list(unique_papers.values())
    logger.info(f"Total unique papers found: {len(unique_paper_list)}")
    
    return unique_paper_list


def save_data(papers_data, category, output_dir=RAW_DIR, overwrite=False):
    """
    Save papers data to a JSON file.
    
    Args:
        papers_data: List of paper dictionaries
        category: arXiv category
        output_dir: Directory to save the file
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to the saved file
    """
    # Create safe filename
    safe_category = category.replace(".", "_")
    output_file = output_dir / f"{safe_category}.json"
    
    # Check if file exists and overwrite flag
    if output_file.exists() and not overwrite:
        logger.info(f"File {output_file} already exists. Use --overwrite to replace.")
        return output_file
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(papers_data, f, indent=2)
    
    logger.info(f"Saved {len(papers_data)} papers to {output_file}")
    return output_file


def main():
    """Main execution function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process each category
    for category in args.categories:
        # Fetch papers
        papers = fetch_papers(
            category=category,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            chunk_months=args.chunk_months,
            max_results_per_chunk=args.max_results_per_chunk,
            delay=args.delay
        )
        
        if papers:
            # Save data
            save_data(papers, category, overwrite=args.overwrite)
        else:
            logger.warning(f"No papers found for {category}")


if __name__ == "__main__":
    main()