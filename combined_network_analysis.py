#!/usr/bin/env python3
"""
Combined Collaboration Network Analysis with Built-in Storage
----------------------------------------------------------
Runs collaboration network analysis across topics with integrated storage.
All functionality combined in a single file for easy deployment.

Usage:
    python combined_network_analysis.py --test-run
    python combined_network_analysis.py --start-topic 0 --end-topic 100
    python combined_network_analysis.py --topic-list 0,5,10,15
"""

import pandas as pd
import networkx as nx
import numpy as np
import ast 
import argparse
import json
import time
import sqlite3
import pickle
import gzip
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional
import traceback
import warnings
warnings.filterwarnings('ignore')

class NetworkDataManager:
    """
    Manages storage and retrieval of collaboration network analysis data.
    Lightweight version with essential storage capabilities.
    """
    
    def __init__(self, base_dir: str = "results/network_analysis"):
        """Initialize the data storage system."""
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_database()
        self.logger = self._setup_logging()
        
    def setup_directories(self):
        """Create directory structure."""
        self.dirs = {
            'base': self.base_dir,
            'sqlite': self.base_dir / "sqlite_db",
            'networks': self.base_dir / "network_files",
            'summaries': self.base_dir / "topic_summaries",
            'exports': self.base_dir / "exports",
            'logs': self.base_dir / "logs"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_database(self):
        """Initialize SQLite database."""
        self.db_path = self.dirs['sqlite'] / "network_analysis.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS topic_summary (
                    topic_id INTEGER PRIMARY KEY,
                    total_papers INTEGER,
                    collab_papers INTEGER,
                    unique_authors INTEGER,
                    collaboration_edges INTEGER,
                    network_density REAL,
                    largest_component_size INTEGER,
                    num_components INTEGER,
                    most_productive_author TEXT,
                    max_author_papers INTEGER,
                    strongest_collab_weight INTEGER,
                    quarters_active INTEGER,
                    first_quarter TEXT,
                    last_quarter TEXT,
                    peak_quarter TEXT,
                    peak_quarter_papers INTEGER,
                    analysis_timestamp TEXT
                )
            ''')
    
    def validate_dataset(self):
        """Validate and inspect the dataset structure."""
        try:
            data_file = Path("data/cleaned/author_topic_networks.csv")
            if not data_file.exists():
                self.logger.error(f"Dataset file not found: {data_file}")
                return False
            
            self.logger.info("Validating dataset structure...")
            df = pd.read_csv(data_file, nrows=5)  # Just read first 5 rows for inspection
            
            self.logger.info(f"Dataset columns: {list(df.columns)}")
            self.logger.info(f"Dataset shape (first 5 rows): {df.shape}")
            
            # Check required columns
            required_columns = ['topic', 'authors_parsed']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Show sample data
            self.logger.info("Sample row:")
            for col in df.columns:
                self.logger.info(f"  {col}: {df[col].iloc[0]}")
            
            # Check topics available
            full_df = pd.read_csv(data_file, usecols=['topic'] if 'topic' in df.columns else [])
            if 'topic' in full_df.columns:
                unique_topics = sorted(full_df['topic'].unique())
                self.logger.info(f"Total topics available: {len(unique_topics)}")
                self.logger.info(f"Topic range: {min(unique_topics)} to {max(unique_topics)}")
                
                # Show topic distribution
                topic_counts = full_df['topic'].value_counts().sort_index()
                self.logger.info("Sample topic counts:")
                for topic in unique_topics[:10]:  # Show first 10 topics
                    count = topic_counts.get(topic, 0)
                    self.logger.info(f"  Topic {topic}: {count} papers")
                if len(unique_topics) > 10:
                    self.logger.info(f"  ... and {len(unique_topics) - 10} more topics")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating dataset: {e}")
            return False
    
    def _setup_logging(self):
        """Setup logging for the storage manager."""
        log_file = self.dirs['logs'] / f"network_storage_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('NetworkDataManager')
    
    def store_topic_results(self, 
                           topic_id: int, 
                           quarterly_networks: Dict,
                           quarterly_metrics: Dict,
                           overall_stats: Dict,
                           author_data: List[Dict],
                           collaboration_data: List[Dict]):
        """Store analysis results for a topic."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Store networks as pickle files (simpler than GraphML for now)
            self._store_network_files(topic_id, quarterly_networks)
            
            # Store summary data as JSON
            summary_data = {
                'topic_id': topic_id,
                'timestamp': timestamp,
                'overall_stats': overall_stats,
                'quarterly_metrics': quarterly_metrics,
                'author_data': author_data[:50],  # Limit to top 50 authors
                'collaboration_data': collaboration_data[:100]  # Limit to top 100 collaborations
            }
            self._store_summary_json(topic_id, summary_data)
            
            # Update SQLite database
            self._update_database(topic_id, overall_stats, timestamp)
            
            self.logger.info(f"Stored results for topic {topic_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing topic {topic_id}: {e}")
            raise
    
    def _store_network_files(self, topic_id: int, quarterly_networks: Dict):
        """Store networks as compressed pickle files."""
        topic_dir = self.dirs['networks'] / f"topic_{topic_id}"
        topic_dir.mkdir(exist_ok=True)
        
        for quarter, network in quarterly_networks.items():
            if network.number_of_nodes() > 0:
                filename = f"network_{quarter.replace('-', '_')}.pkl.gz"
                filepath = topic_dir / filename
                
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(network, f)
    
    def _store_summary_json(self, topic_id: int, summary_data: Dict):
        """Store summary as JSON."""
        json_file = self.dirs['summaries'] / f"summary_topic_{topic_id}.json"
        
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
    
    def _update_database(self, topic_id: int, overall_stats: Dict, timestamp: str):
        """Update SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO topic_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                topic_id,
                overall_stats.get('total_papers', 0),
                overall_stats.get('collab_papers', 0),
                overall_stats.get('unique_authors', 0),
                overall_stats.get('collaboration_edges', 0),
                overall_stats.get('network_density', 0.0),
                overall_stats.get('largest_component_size', 0),
                overall_stats.get('num_components', 0),
                overall_stats.get('most_productive_author', ''),
                overall_stats.get('max_author_papers', 0),
                overall_stats.get('strongest_collab_weight', 0),
                overall_stats.get('quarters_active', 0),
                overall_stats.get('first_quarter', ''),
                overall_stats.get('last_quarter', ''),
                overall_stats.get('peak_quarter', ''),
                overall_stats.get('peak_quarter_papers', 0),
                timestamp
            ))
    
    def get_cross_topic_summary(self) -> pd.DataFrame:
        """Get summary across all topics."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query('SELECT * FROM topic_summary ORDER BY topic_id', conn)
    
    def export_summary(self) -> str:
        """Export summary to CSV."""
        summary_df = self.get_cross_topic_summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.dirs['exports'] / f"network_analysis_summary_{timestamp}.csv"
        summary_df.to_csv(output_file, index=False)
        return str(output_file)


class BatchNetworkAnalyzer:
    """Batch network analysis with integrated storage."""
    
    def __init__(self, data_path: str, storage_dir: str = "results/network_analysis"):
        """Initialize analyzer."""
        self.data_path = Path(data_path)
        self.storage_manager = NetworkDataManager(storage_dir)
        self.logger = self._setup_logging()
        
        # Progress tracking
        self.checkpoint_file = Path(storage_dir) / "batch_progress.json"
        self.progress = self._load_checkpoint()
        
    def _setup_logging(self):
        """Setup logging."""
        log_dir = Path(self.storage_manager.base_dir) / "logs"
        log_file = log_dir / f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('BatchNetworkAnalyzer')
    
    def _load_checkpoint(self) -> Dict:
        """Load progress checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            'completed_topics': [],
            'failed_topics': [],
            'last_completed': None,
            'start_time': None,
            'total_topics_planned': 0
        }
    
    def _save_checkpoint(self):
        """Save progress checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def validate_dataset(self):
        """Validate dataset through storage manager."""
        return self.storage_manager.validate_dataset()
    
    def parse_authors(self, authors_str):
        """Parse authors from data format."""
        if pd.isna(authors_str) or not authors_str:
            return []
        
        try:
            authors_list = ast.literal_eval(authors_str)
            names = []
            for author_parts in authors_list:
                if isinstance(author_parts, list) and len(author_parts) >= 2:
                    last_name = str(author_parts[0]).strip()
                    first_name = str(author_parts[1]).strip()
                    if first_name and last_name:
                        names.append(f"{first_name} {last_name}")
                    elif last_name:
                        names.append(last_name)
            return names
        except (ValueError, SyntaxError, TypeError):
            return []
    
    def load_topic_data(self, topic_id: int) -> Optional[pd.DataFrame]:
        """Load data for a specific topic from the consolidated CSV file."""
        try:
            # Check if we have already loaded the full dataset
            if not hasattr(self, '_full_dataset'):
                # Try to find the main data file
                possible_files = [
                    "data/cleaned/author_topic_networks.csv",
                    self.data_path / "author_topic_networks.csv", 
                    "sample_topic_networks.csv",  # For testing
                    self.data_path / "sample_topic_networks.csv"
                ]
                
                data_file = None
                for file_path in possible_files:
                    file_path = Path(file_path)
                    if file_path.exists():
                        data_file = file_path
                        break
                
                if data_file is None:
                    self.logger.error("Could not find author_topic_networks.csv or sample data file")
                    self.logger.info("Looking for files in these locations:")
                    for file_path in possible_files:
                        self.logger.info(f"  - {file_path}")
                    return None
                
                self.logger.info(f"Loading full dataset from: {data_file}")
                self._full_dataset = pd.read_csv(data_file)
                self.logger.info(f"Loaded dataset with {len(self._full_dataset)} total papers")
                
                # Check what topics are available
                if 'topic' in self._full_dataset.columns:
                    available_topics = sorted(self._full_dataset['topic'].unique())
                    self.logger.info(f"Available topics: {len(available_topics)} topics "
                                   f"(range: {min(available_topics)} to {max(available_topics)})")
                else:
                    self.logger.warning("No 'topic' column found in dataset")
            
            # Filter for the specific topic
            if 'topic' not in self._full_dataset.columns:
                self.logger.error("Dataset missing 'topic' column")
                return None
                
            topic_df = self._full_dataset[self._full_dataset['topic'] == topic_id].copy()
            
            if topic_df.empty:
                self.logger.warning(f"No papers found for topic {topic_id}")
                return None
            
            self.logger.debug(f"Topic {topic_id}: Found {len(topic_df)} papers")
            
            # Parse authors
            if 'authors_parsed' in topic_df.columns:
                topic_df['authors_list'] = topic_df['authors_parsed'].apply(self.parse_authors)
            else:
                self.logger.error("Dataset missing 'authors_parsed' column")
                return None
            
            # Filter for collaboration papers only
            topic_df = topic_df[topic_df['authors_list'].apply(lambda x: len(x) > 1)]
            
            if topic_df.empty:
                self.logger.info(f"Topic {topic_id}: No collaboration papers found")
                return None
            
            # Add time information
            if 'year' in topic_df.columns and 'quarter' in topic_df.columns:
                topic_df['year_quarter'] = topic_df['year'].astype(str) + '-Q' + topic_df['quarter'].astype(str)
            else:
                self.logger.warning(f"Topic {topic_id}: Missing year/quarter information")
                # Still proceed without temporal analysis
                topic_df['year_quarter'] = '2020-Q1'  # Default value
            
            self.logger.info(f"Topic {topic_id}: {len(topic_df)} collaboration papers ready for analysis")
            return topic_df
            
        except Exception as e:
            self.logger.error(f"Error loading data for topic {topic_id}: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def build_collaboration_network(self, papers_df: pd.DataFrame) -> nx.Graph:
        """Build collaboration network."""
        G = nx.Graph()
        
        for _, paper in papers_df.iterrows():
            authors = paper['authors_list']
            
            # Add nodes
            for author in authors:
                if author not in G.nodes():
                    G.add_node(author, papers=0)
                G.nodes[author]['papers'] += 1
            
            # Add edges
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    if G.has_edge(author1, author2):
                        G[author1][author2]['weight'] += 1
                        G[author1][author2]['papers'].append(paper['id'])
                    else:
                        G.add_edge(author1, author2, weight=1, papers=[paper['id']])
        
        return G
    
    def calculate_network_metrics(self, G: nx.Graph) -> Dict:
        """Calculate network metrics."""
        if len(G) == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'density': 0,
                'num_components': 0,
                'largest_component_size': 0
            }
        
        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'num_components': nx.number_connected_components(G)
        }
        
        if G.number_of_nodes() > 0:
            # Largest connected component
            components = list(nx.connected_components(G))
            if components:
                largest_cc = max(components, key=len)
                metrics['largest_component_size'] = len(largest_cc)
                
                if len(largest_cc) > 2:
                    largest_cc_subgraph = G.subgraph(largest_cc)
                    try:
                        metrics['average_clustering'] = nx.average_clustering(largest_cc_subgraph)
                    except:
                        metrics['average_clustering'] = 0
        
        return metrics
    
    def analyze_topic_network(self, topic_id: int) -> Optional[Dict]:
        """Analyze network for a single topic."""
        start_time = time.time()
        
        try:
            df = self.load_topic_data(topic_id)
            if df is None:
                return None
            
            self.logger.info(f"Analyzing topic {topic_id}: {len(df)} collaboration papers")
            
            if 'year_quarter' not in df.columns:
                self.logger.warning(f"Topic {topic_id}: No temporal information available")
                return None
            
            quarters = sorted(df['year_quarter'].unique())
            
            # Build quarterly networks
            quarterly_networks = {}
            quarterly_metrics = {}
            
            for quarter in quarters:
                quarter_papers = df[df['year_quarter'] == quarter]
                if len(quarter_papers) > 0:
                    quarter_network = self.build_collaboration_network(quarter_papers)
                    quarterly_networks[quarter] = quarter_network
                    
                    quarter_metrics = self.calculate_network_metrics(quarter_network)
                    quarter_metrics['quarter'] = quarter
                    quarter_metrics['num_papers'] = len(quarter_papers)
                    quarterly_metrics[quarter] = quarter_metrics
            
            # Overall network
            overall_network = self.build_collaboration_network(df)
            overall_metrics = self.calculate_network_metrics(overall_network)
            
            # Author analysis
            author_data = []
            for author, data in overall_network.nodes(data=True):
                author_papers = data.get('papers', 0)
                collaborators = len(list(overall_network.neighbors(author)))
                
                author_info = {
                    'author': author,
                    'papers': author_papers,
                    'collaborators': collaborators,
                    'productivity_score': author_papers / len(quarters) if quarters else 0
                }
                author_data.append(author_info)
            
            # Sort by papers count
            author_data.sort(key=lambda x: x['papers'], reverse=True)
            
            # Collaboration analysis
            collaboration_data = []
            for u, v, data in overall_network.edges(data=True):
                weight = data.get('weight', 1)
                collab_info = {
                    'author1': u,
                    'author2': v,
                    'weight': weight
                }
                collaboration_data.append(collab_info)
            
            # Sort by weight
            collaboration_data.sort(key=lambda x: x['weight'], reverse=True)
            
            # Overall statistics
            overall_stats = {
                'topic_id': topic_id,
                'total_papers': len(df),
                'collab_papers': len(df),
                'unique_authors': overall_network.number_of_nodes(),
                'collaboration_edges': overall_network.number_of_edges(),
                'network_density': overall_metrics.get('density', 0),
                'largest_component_size': overall_metrics.get('largest_component_size', 0),
                'num_components': overall_metrics.get('num_components', 0),
                'quarters_active': len(quarters),
                'first_quarter': quarters[0] if quarters else '',
                'last_quarter': quarters[-1] if quarters else '',
            }
            
            # Most productive author
            if author_data:
                most_productive = author_data[0]
                overall_stats['most_productive_author'] = most_productive['author']
                overall_stats['max_author_papers'] = most_productive['papers']
            
            # Peak quarter
            if quarterly_metrics:
                peak_quarter = max(quarterly_metrics.items(), key=lambda x: x[1].get('num_papers', 0))
                overall_stats['peak_quarter'] = peak_quarter[0]
                overall_stats['peak_quarter_papers'] = peak_quarter[1].get('num_papers', 0)
            
            # Strongest collaboration
            if collaboration_data:
                strongest_collab = collaboration_data[0]
                overall_stats['strongest_collab_weight'] = strongest_collab['weight']
            
            analysis_time = time.time() - start_time
            self.logger.info(f"Topic {topic_id} analyzed in {analysis_time:.2f}s")
            
            return {
                'quarterly_networks': quarterly_networks,
                'quarterly_metrics': quarterly_metrics,
                'overall_stats': overall_stats,
                'author_data': author_data,
                'collaboration_data': collaboration_data,
                'analysis_time': analysis_time
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing topic {topic_id}: {e}")
            return None
    
    def run_batch_analysis(self, 
                      topic_list: List[int] = None,
                      start_topic: int = 0,
                      end_topic: int = 1937,
                      resume_from_checkpoint: bool = False):
    
        """Run batch analysis."""
    
        # Determine topics to analyze
        if topic_list is not None:
            topics_to_analyze = topic_list
        else:
            topics_to_analyze = list(range(start_topic, end_topic + 1))
    
        # Filter completed topics if resuming
        if resume_from_checkpoint:
            completed = set(self.progress['completed_topics'])
            topics_to_analyze = [t for t in topics_to_analyze if t not in completed]
            self.logger.info(f"Resuming: {len(completed)} topics already completed.")
        else:
            # Clear the failed topics list if not resuming
            self.progress['failed_topics'] = []
    
        # Initialize progress
        if not self.progress['start_time']:
            self.progress['start_time'] = datetime.now().isoformat()
        self.progress['total_topics_planned'] = len(topics_to_analyze)
    
        self.logger.info(f"Starting batch analysis for {len(topics_to_analyze)} topics")
    
        batch_start_time = time.time()
        successful_analyses = 0
        failed_in_this_run = []  # Track failures in this specific run
    
        for i, topic_id in enumerate(topics_to_analyze):
            try:
                self.logger.info(f"Processing topic {topic_id} ({i+1}/{len(topics_to_analyze)})")
            
                results = self.analyze_topic_network(topic_id)
            
                if results is not None:
                    self.storage_manager.store_topic_results(
                        topic_id,
                        results['quarterly_networks'],
                        results['quarterly_metrics'],
                        results['overall_stats'],
                        results['author_data'],
                        results['collaboration_data']
                    )
                
                    # Remove from failed list if it was there before
                    if topic_id in self.progress['failed_topics']:
                        self.progress['failed_topics'].remove(topic_id)
                
                    # Add to completed if not already there
                    if topic_id not in self.progress['completed_topics']:
                        self.progress['completed_topics'].append(topic_id)
                
                    self.progress['last_completed'] = topic_id
                    successful_analyses += 1
                
                    self.logger.info(f"Topic {topic_id}: Completed successfully")
                
                else:
                    failed_in_this_run.append(topic_id)
                    if topic_id not in self.progress['failed_topics']:
                        self.progress['failed_topics'].append(topic_id)
                    self.logger.warning(f"Topic {topic_id}: Analysis failed")
            
                # Save checkpoint every 10 topics
                if i % 10 == 0:
                    self._save_checkpoint()
            
                # Progress update every 25 topics
                if i % 25 == 0 and i > 0:
                    elapsed = time.time() - batch_start_time
                    avg_time = elapsed / (i + 1)
                    remaining = len(topics_to_analyze) - (i + 1)
                    est_remaining = remaining * avg_time
                
                    self.logger.info(f"Progress: {i+1}/{len(topics_to_analyze)}")
                    self.logger.info(f"Avg time/topic: {avg_time:.2f}s")
                    self.logger.info(f"Est. remaining: {est_remaining/60:.1f} min")
            
            except Exception as e:
                self.logger.error(f"Error processing topic {topic_id}: {e}")
                failed_in_this_run.append(topic_id)
                if topic_id not in self.progress['failed_topics']:
                    self.progress['failed_topics'].append(topic_id)
                continue
    
        # Final save
        self._save_checkpoint()
    
        total_time = time.time() - batch_start_time
        self.logger.info(f"Batch analysis completed!")
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Successful: {successful_analyses}")
        self.logger.info(f"Failed: {len(failed_in_this_run)}")
    
        # Export summary
        try:
            export_file = self.storage_manager.export_summary()
            self.logger.info(f"Summary exported to: {export_file}")
        except Exception as e:
            self.logger.error(f"Export error: {e}")
    
        return {
            'completed': successful_analyses,
            'failed': len(failed_in_this_run),  # Return failures from this run only
            'total_time': total_time,
            'failed_topics': failed_in_this_run  # Return failures from this run only
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch collaboration network analysis"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=".",
        help="Path to data directory"
    )
    parser.add_argument(
        "--storage-dir", 
        type=str,
        default="results/network_analysis",
        help="Storage directory"
    )
    parser.add_argument(
        "--start-topic",
        type=int,
        default=0,
        help="Starting topic ID"
    )
    parser.add_argument(
        "--end-topic",
        type=int,
        default=1937,
        help="Ending topic ID"
    )
    parser.add_argument(
        "--topic-list",
        type=str,
        default=None,
        help="Comma-separated topic list"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run with sample data"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse topic list
    topic_list = None
    if args.topic_list:
        topic_list = [int(x.strip()) for x in args.topic_list.split(',')]
    
    # Test run override  
    if args.test_run:
        topic_list = [0]  # Just analyze topic 0 for testing
        print("TEST RUN: Analyzing only topic 0")
    
    # Initialize analyzer
    analyzer = BatchNetworkAnalyzer(args.data_path, args.storage_dir)
    
    # Validate dataset first
    print("Validating dataset...")
    if not analyzer.validate_dataset():
        print("Dataset validation failed. Please check the logs for details.")
        return
    
    print("Dataset validation successful!")
    
    # Run analysis
    results = analyzer.run_batch_analysis(
        topic_list=topic_list,
        start_topic=args.start_topic,
        end_topic=args.end_topic,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    print("\n" + "="*50)
    print("BATCH ANALYSIS SUMMARY")
    print("="*50)
    print(f"Successfully completed: {results['completed']} topics")
    print(f"Failed: {results['failed']} topics")
    print(f"Total time: {results['total_time']/60:.1f} minutes")
    
    if results['failed_topics']:
        print(f"Failed topics: {results['failed_topics']}")


def validate_dataset_standalone():
    """Standalone function to just validate the dataset."""
    analyzer = BatchNetworkAnalyzer(".", "results/network_analysis")
    success = analyzer.validate_dataset()
    
    if success:
        print("✅ Dataset validation successful!")
    else:
        print("❌ Dataset validation failed!")
    
    return success


if __name__ == "__main__":
    main()
