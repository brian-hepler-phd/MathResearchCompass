#!/usr/bin/env python3
"""
claude_topic_labeler.py - Generate AI-powered labels for BERTopic topics using Claude API

This script loads topic keywords from existing BERTopic results and uses
Anthropic's Claude API to generate concise, human-readable labels.

Usage:
    python claude_topic_labeler.py --metadata_file path/to/metadata_file.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import textwrap
import time
from tqdm import tqdm
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("topic_labeler")

# Define paths
RESULTS_DIR = Path("results")
TOPIC_DIR = RESULTS_DIR / "topics"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate Claude AI-powered labels for BERTopic topics"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="Path to the metadata JSON file for the analysis run"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="claude_labeled",
        help="Suffix to add to output files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of topics to process in a batch (for API calls)"
    )
    parser.add_argument(
        "--claude_model",
        type=str,
        default="claude-3-opus-20240229",
        help="Claude model to use for labeling (e.g., claude-3-opus-20240229, claude-3-haiku-20240307)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    # New arguments for improved function
    parser.add_argument(
        "--max_api_calls",
        type=int,
        default=2000,
        help="Maximum number of API calls to make (budget limit)"
    )
    parser.add_argument(
        "--start_topic",
        type=int,
        default=0,
        help="Topic ID to start from (for resuming previous runs)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="How often to save progress (in batches)"
    )
    
    args = parser.parse_args()
    
    # If no metadata file is specified, try to find the most recent one
    if not args.metadata_file:
        metadata_files = list(TOPIC_DIR.glob("metadata_*.json"))
        if metadata_files:
            # Sort by timestamp (newest first)
            metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            args.metadata_file = str(metadata_files[0])
            logger.info(f"Using most recent metadata file: {args.metadata_file}")
        else:
            logger.error("No metadata files found and none specified.")
            sys.exit(1)
    
    return args

def load_data(metadata_path):
    """
    Load topic data from the specified metadata file.
    
    Args:
        metadata_path: Path to the metadata JSON file
    
    Returns:
        Dictionary with loaded data
    """
    logger.info(f"Loading data from {metadata_path}")
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata file: {e}")
        return None
    
    data = {"metadata": metadata}
    
    # Load topic info
    try:
        topic_info_path = TOPIC_DIR / metadata["file_references"]["topic_info"]
        data["topic_info"] = pd.read_csv(topic_info_path)
        logger.info(f"Loaded topic info with {len(data['topic_info'])} topics")
    except Exception as e:
        logger.error(f"Could not load topic info: {e}")
        return None
    
    # Load topic keywords
    try:
        keywords_path = TOPIC_DIR / metadata["file_references"]["topic_keywords"]
        with open(keywords_path, "r") as f:
            data["topic_keywords"] = json.load(f)
        logger.info(f"Loaded keywords for {len(data['topic_keywords'])} topics")
    except Exception as e:
        logger.error(f"Could not load topic keywords: {e}")
        return None
    
    return data

def generate_fallback_label(topic_id, keywords):
    """
    Generate a simple rule-based label for a topic based on its keywords.
    
    Args:
        topic_id: ID of the topic
        keywords: List of (term, weight) tuples for the topic
    
    Returns:
        A concise label for the topic
    """
    # Extract the top keywords
    terms = []
    seen_roots = set()
    
    for keyword, _ in keywords[:4]:  # Use top 4 keywords
        clean_keyword = re.sub(r'[_\s]+', ' ', keyword).strip()
        
        # Skip if we already have a related term (basic stemming)
        root_form = clean_keyword.split()[0] if clean_keyword.split() else ""
        if root_form and len(root_form) > 3:
            if root_form in seen_roots:
                continue
            seen_roots.add(root_form)
        
        terms.append(clean_keyword)
    
    # Create a descriptive name from keywords
    if terms:
        topic_name = " & ".join(terms[:2])  # Just use first 2 terms
        # Capitalize first letter of each word
        topic_name = ' '.join(word.capitalize() for word in topic_name.split())
    else:
        topic_name = f"Topic {topic_id}"
    
    return topic_name

def create_prompt_for_topic(topic_id, keywords, subjects):
    """
    Create a prompt for Claude to generate a topic label.
    
    Args:
        topic_id: ID of the topic
        keywords: List of (term, weight) tuples for the topic
        subjects: List of arXiv subjects for context
    
    Returns:
        A prompt for Claude
    """
    # Format the keywords as a list with weights
    keywords_text = "\n".join([f"- {word} ({weight:.4f})" for word, weight in keywords[:15]])
    
    # Create the prompt
    prompt = textwrap.dedent(f"""
    You are a mathematician and data scientist specializing in interpreting topic modeling results.
    
    I have a set of topics generated from a BERTopic model analyzing mathematical research papers from arXiv.
    The papers are primarily from these mathematical subject areas: {', '.join(subjects)}.
    
    For Topic {topic_id}, the top terms (with their weights) are:
    {keywords_text}
    
    Based on these keywords, please identify what mathematical research topic this represents.
    Provide two labels:
    1. A concise label (3-5 words max) that captures the essence of this topic
    2. A more descriptive name that specifies the mathematical subfield (e.g., "Algebraic Topology: Persistent Homology")
    
    Format your response like this:
    SHORT_LABEL: [Your concise label]
    DESCRIPTIVE_LABEL: [Your more descriptive label]
    """).strip()
    
    return prompt

def get_labels_from_claude(prompts, topic_keywords, batch_size=5, model="claude-3-opus-20240229", 
                          max_api_calls=2000, start_topic=0, checkpoint_interval=20,
                          max_retries=3, batch_delay=2.0, request_delay=0.8):
    """
    Get topic labels using Claude API with improved error handling and rate limiting.
    
    Args:
        prompts: Dictionary mapping topic_ids to prompts
        topic_keywords: Dictionary mapping topic_ids to keywords for fallback labeling
        batch_size: Number of topics to process in a batch
        model: Claude model version to use
        max_api_calls: Maximum number of API calls to make (budget limit)
        start_topic: Topic ID to start from (for resuming previous runs)
        checkpoint_interval: How often to save progress (in batches)
        max_retries: Maximum number of retry attempts per API call
        batch_delay: Delay between batches in seconds
        request_delay: Delay between requests in seconds
    
    Returns:
        Dictionary mapping topic_ids to (short_label, descriptive_label) tuples
    """
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return {}
    
    # Base API URL and headers
    api_url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Get topic IDs and filter if starting from a specific topic
    topic_ids = list(prompts.keys())
    if start_topic > 0:
        topic_ids = [t for t in topic_ids if int(t) >= start_topic]
        logger.info(f"Starting from topic {start_topic}, {len(topic_ids)} topics remaining")
    
    # Initialize results dictionary and load any existing checkpoint
    results = {}
    checkpoint_file = f"topic_labels_checkpoint.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                saved_results = json.load(f)
                # Convert string keys back to integers
                for k, v in saved_results.items():
                    results[int(k)] = v
            logger.info(f"Loaded {len(results)} topics from checkpoint")
            
            # Skip topics that are already in the checkpoint
            topic_ids = [t for t in topic_ids if t not in results]
            logger.info(f"After filtering already processed topics: {len(topic_ids)} remaining")
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
    
    if not topic_ids:
        logger.info("No topics to process")
        return results
    
    # Initialize API call counter and progress reporting
    api_call_count = 0
    successful_calls = 0
    failed_calls = 0
    
    logger.info(f"Processing {len(topic_ids)} topics with Claude API in batches of {batch_size}")
    
    # Process topics in batches
    for i in tqdm(range(0, len(topic_ids), batch_size)):
        batch_ids = topic_ids[i:i+batch_size]
        
        # Add a delay between batches to avoid rate limits
        if i > 0:
            time.sleep(batch_delay)
        
        batch_results = {}  # Track results for this batch
        batch_success = 0   # Track successful calls in this batch
        
        for topic_id in batch_ids:
            # Skip if we've already exceeded the API call budget
            if api_call_count >= max_api_calls:
                logger.warning(f"Reached maximum API call limit of {max_api_calls}")
                # Apply fallback labeling to all remaining topics
                for remaining_id in [t for t in topic_ids if t not in results]:
                    keyword_list = topic_keywords.get(str(remaining_id), [])
                    fallback_label = generate_fallback_label(remaining_id, keyword_list)
                    results[remaining_id] = (fallback_label, f"Mathematics: {fallback_label}")
                
                # Save final checkpoint and return
                with open(checkpoint_file, "w") as f:
                    json.dump({str(k): v for k, v in results.items()}, f)
                
                logger.info(f"Final statistics: {successful_calls} successful API calls, {failed_calls} failed calls")
                return results
            
            prompt = prompts[topic_id]
            
            # Try to get labels with retries
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    # Increment API call counter before making the call
                    api_call_count += 1
                    
                    # Prepare the request payload
                    payload = {
                        "model": model,
                        "max_tokens": 100,
                        "temperature": 0.3,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }
                    
                    # Make the API request
                    response = requests.post(api_url, headers=headers, json=payload)
                    response.raise_for_status()  # Raise exception for HTTP errors
                    
                    # Parse the response
                    response_data = response.json()
                    response_text = response_data["content"][0]["text"]
                    
                    # Extract labels
                    short_label = None
                    descriptive_label = None
                    
                    for line in response_text.split('\n'):
                        if line.startswith("SHORT_LABEL:"):
                            short_label = line.replace("SHORT_LABEL:", "").strip()
                        elif line.startswith("DESCRIPTIVE_LABEL:"):
                            descriptive_label = line.replace("DESCRIPTIVE_LABEL:", "").strip()
                    
                    if short_label and descriptive_label:
                        batch_results[topic_id] = (short_label, descriptive_label)
                        success = True
                        successful_calls += 1
                        batch_success += 1
                    else:
                        # Parsing issue, use fallback
                        logger.warning(f"Could not parse response for topic {topic_id}")
                        keyword_list = topic_keywords.get(str(topic_id), [])
                        fallback_label = generate_fallback_label(topic_id, keyword_list)
                        batch_results[topic_id] = (fallback_label, f"Mathematics: {fallback_label}")
                        failed_calls += 1
                    
                except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as e:
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        # Final failure, use fallback
                        logger.error(f"Error getting labels for topic {topic_id} after {max_retries} retries: {e}")
                        keyword_list = topic_keywords.get(str(topic_id), [])
                        fallback_label = generate_fallback_label(topic_id, keyword_list)
                        batch_results[topic_id] = (fallback_label, f"Mathematics: {fallback_label}")
                        failed_calls += 1
                    else:
                        # Exponential backoff for retry
                        wait_time = 2 ** retry_count
                        logger.warning(f"API error for topic {topic_id}, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                        time.sleep(wait_time)
                
                except Exception as e:
                    # Unexpected error, use fallback
                    logger.error(f"Unexpected error for topic {topic_id}: {e}")
                    keyword_list = topic_keywords.get(str(topic_id), [])
                    fallback_label = generate_fallback_label(topic_id, keyword_list)
                    batch_results[topic_id] = (fallback_label, f"Mathematics: {fallback_label}")
                    failed_calls += 1
                    break  # Exit retry loop
            
            # Add a delay between requests to avoid rate limits
            time.sleep(request_delay)
        
        # Update results with batch results
        results.update(batch_results)
        
        # Log batch statistics
        logger.info(f"Batch {i//batch_size + 1}/{len(topic_ids)//batch_size + 1}: "
                    f"{batch_success}/{len(batch_ids)} successful, "
                    f"total API calls: {api_call_count}/{max_api_calls}")
        
        # Save checkpoint at intervals
        if (i // batch_size) % checkpoint_interval == 0 and i > 0:
            logger.info(f"Saving checkpoint after {i + len(batch_ids)} topics")
            with open(checkpoint_file, "w") as f:
                json.dump({str(k): v for k, v in results.items()}, f)
    
    # Save final checkpoint
    with open(checkpoint_file, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f)
    
    logger.info(f"Final statistics: {successful_calls} successful API calls, {failed_calls} failed calls")
    return results
    
    return results

def update_topic_info(data, topic_labels, output_suffix):
    """
    Update topic info with generated labels.
    
    Args:
        data: Dictionary with loaded data
        topic_labels: Dictionary mapping topic_ids to (short_label, descriptive_label) tuples
        output_suffix: Suffix to add to output files
    
    Returns:
        Updated topic info DataFrame
    """
    logger.info("Updating topic info with generated labels")
    
    topic_info = data["topic_info"]
    
    # Add new label columns
    topic_info["ShortLabel"] = ""
    topic_info["DescriptiveLabel"] = ""
    
    # Update labels
    for i, row in topic_info.iterrows():
        topic_id = row["Topic"]
        
        # Skip outlier topic
        if topic_id == -1:
            topic_info.at[i, "ShortLabel"] = "Outlier Topics"
            topic_info.at[i, "DescriptiveLabel"] = "Miscellaneous Documents"
            continue
        
        # Get labels for this topic
        if topic_id in topic_labels:
            short_label, descriptive_label = topic_labels[topic_id]
            topic_info.at[i, "ShortLabel"] = short_label
            topic_info.at[i, "DescriptiveLabel"] = descriptive_label
        else:
            # Use generic labels
            topic_info.at[i, "ShortLabel"] = f"Topic {topic_id}"
            topic_info.at[i, "DescriptiveLabel"] = f"Mathematics Topic {topic_id}"
    
    # Save updated topic info
    timestamp = data["metadata"]["timestamp"]
    output_path = TOPIC_DIR / f"topic_info_{timestamp}_{output_suffix}.csv"
    topic_info.to_csv(output_path, index=False)
    logger.info(f"Saved updated topic info to {output_path}")
    
    return topic_info

def main():
    """Main execution function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load data
    data = load_data(args.metadata_file)
    
    if not data:
        logger.error("Failed to load required data. Exiting.")
        sys.exit(1)
    
    # Get arXiv subjects from metadata
    subjects = data["metadata"]["subjects"]
    
    # Prepare prompts for all topics
    prompts = {}
    topic_keywords = data["topic_keywords"]
    topic_info = data["topic_info"]
    
    for i, row in topic_info.iterrows():
        topic_id = row["Topic"]
        
        # Skip outlier topic (-1)
        if topic_id == -1:
            continue
        
        # Skip topics below the start_topic if resuming
        if args.start_topic > 0 and topic_id < args.start_topic:
            continue
        
        # Get keywords
        topic_str = str(topic_id)
        if topic_str in topic_keywords:
            keywords = topic_keywords[topic_str]
        else:
            # Try with integer keys
            keywords = topic_keywords.get(topic_id, [])
        
        if not keywords:
            logger.warning(f"No keywords found for topic {topic_id}")
            continue
        
        # Create prompt
        prompts[topic_id] = create_prompt_for_topic(topic_id, keywords, subjects)
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY environment variable not set. Using fallback method.")
        print("ANTHROPIC_API_KEY not found. Please set it to use Claude API:")
        print("export ANTHROPIC_API_KEY=your_claude_api_key")
        
        # Use fallback rule-based method
        logger.info("Using rule-based fallback method for topic labeling")
        topic_labels = {}
        for topic_id, keywords in topic_keywords.items():
            if str(topic_id) == "-1":
                continue
            
            topic_id_int = int(topic_id)
            short_label = generate_fallback_label(topic_id_int, keywords)
            descriptive_label = f"Mathematics: {short_label}"
            topic_labels[topic_id_int] = (short_label, descriptive_label)
    else:
        # Use Claude API with the improved function
        topic_labels = get_labels_from_claude(
            prompts=prompts,
            topic_keywords=topic_keywords,
            batch_size=args.batch_size,
            model=args.claude_model,
            max_api_calls=args.max_api_calls,
            start_topic=args.start_topic,
            checkpoint_interval=args.checkpoint_interval,
            batch_delay=2.0,  # 2 seconds between batches
            request_delay=0.8  # 0.8 seconds between requests
        )
    
    # Update topic info with generated labels
    updated_info = update_topic_info(data, topic_labels, args.output_suffix)
    
    # Print example labels
    print("\nExample topic labels:")
    sample = updated_info[updated_info["Topic"] != -1].sample(min(5, len(updated_info)))
    for _, row in sample.iterrows():
        print(f"Topic {row['Topic']}:")
        print(f"  Original name: {row['Name']}")
        print(f"  Short label: {row['ShortLabel']}")
        print(f"  Descriptive label: {row['DescriptiveLabel']}")
        print()
    
    # Count topics that got API-generated labels vs fallback labels
    if os.environ.get("ANTHROPIC_API_KEY"):
        api_labeled = sum(1 for _, row in updated_info.iterrows() 
                         if row["Topic"] != -1 and not row["ShortLabel"].startswith(f"Topic {row['Topic']}"))
        fallback_labeled = sum(1 for _, row in updated_info.iterrows() 
                             if row["Topic"] != -1 and row["ShortLabel"].startswith(f"Topic {row['Topic']}"))
        print(f"Labels generated for {len(updated_info[updated_info['Topic'] != -1])} topics")
        print(f"  {api_labeled} topics received API-generated labels")
        print(f"  {fallback_labeled} topics used fallback labeling")
    else:
        print(f"Labels generated for {len(updated_info[updated_info['Topic'] != -1])} topics using fallback method")
    
    print(f"Updated topic info saved to {TOPIC_DIR}/topic_info_{data['metadata']['timestamp']}_{args.output_suffix}.csv")
    
    # Suggest next steps
    print("\nNext steps:")
    print("1. Update your app.py to use the new labeled topic info file")
    print("2. To run the labeler with a different Claude model:")
    print(f"   ANTHROPIC_API_KEY=your_key python {sys.argv[0]} --claude_model=claude-3-haiku-20240307")
    
    # Add suggestions for resuming if there were failures
    if os.environ.get("ANTHROPIC_API_KEY") and fallback_labeled > 0:
        # Find the lowest topic ID that used fallback labeling
        min_fallback_id = min((row["Topic"] for _, row in updated_info.iterrows() 
                               if row["Topic"] != -1 and row["ShortLabel"].startswith(f"Topic {row['Topic']}")),
                               default=0)
        if min_fallback_id > 0:
            print("\nTo resume labeling for topics that used fallback method:")
            print(f"   python {sys.argv[0]} --start_topic={min_fallback_id} --max_api_calls=1000")

if __name__ == "__main__":
    main()
