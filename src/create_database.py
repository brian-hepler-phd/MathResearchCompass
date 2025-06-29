#!/usr/bin/env python3
"""
Complete Database Migration Script for Math Research Compass
===========================================================

This script converts your CSV files into an optimized SQLite database for faster loading.

What it does:
1. Creates a SQLite database with proper schema and indexes
2. Migrates common_topics.csv (1,938 topics)
3. Migrates compact_docs_with_topics.csv (121,391 papers) with author processing
4. Migrates JSON keyword data
5. Migrates collaboration analysis results
6. Optimizes database for fast queries

Run with: python create_database.py
"""

import sqlite3
import pandas as pd
import json
import ast
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MathResearchDatabaseMigrator:
    """
    Migrates Math Research Compass data from CSV files to optimized SQLite database.
    """
    
    def __init__(self, db_path="data/dashboard.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Data file paths (adjust these if your files are in different locations)
        self.topics_csv = Path("results/topics/common_topics.csv")
        self.papers_csv = Path("data/cleaned/compact_docs_with_topics.csv")
        self.keywords_json = Path("results/topics/topic_keywords_20250509_221839.json")
        self.category_dist_json = Path("results/topics/topic_category_distribution.json")
        self.top_authors_json = Path("results/topics/top_authors_by_topic.json")
        
        # Collaboration analysis files (will be set by find_latest_collaboration_files)
        self.collaboration_summaries_csv = None
        self.cross_topic_analysis_json = None
        
        logger.info(f"Database will be created at: {self.db_path}")
    
    def find_latest_collaboration_files(self):
        """Find the most recent collaboration analysis files."""
        logger.info("🔍 Finding latest collaboration analysis files...")
        
        # Find latest topic summaries CSV
        summaries_pattern = Path("results/collaboration_analysis/topic_summaries_*.csv")
        summaries_files = list(summaries_pattern.parent.glob(summaries_pattern.name))
        if summaries_files:
            latest_summaries = max(summaries_files, key=lambda x: x.stat().st_mtime)
            self.collaboration_summaries_csv = latest_summaries
            logger.info(f"  Found summaries: {latest_summaries}")
        else:
            logger.warning("  No collaboration summaries CSV found")
            self.collaboration_summaries_csv = None
        
        # Find latest cross-topic analysis JSON
        analysis_pattern = Path("results/collaboration_analysis/cross_topic_analysis_*.json")
        analysis_files = list(analysis_pattern.parent.glob(analysis_pattern.name))
        if analysis_files:
            latest_analysis = max(analysis_files, key=lambda x: x.stat().st_mtime)
            self.cross_topic_analysis_json = latest_analysis
            logger.info(f"  Found analysis: {latest_analysis}")
        else:
            logger.warning("  No cross-topic analysis JSON found")
            self.cross_topic_analysis_json = None
    
    def validate_source_files(self):
        """Check that all required source files exist."""
        logger.info("🔍 Validating source files...")
        
        # Find latest collaboration files first
        self.find_latest_collaboration_files()
        
        required_files = [
            (self.topics_csv, "Topics CSV"),
            (self.papers_csv, "Papers CSV")
        ]
        
        optional_files = [
            (self.keywords_json, "Keywords JSON"),
            (self.category_dist_json, "Category Distribution JSON"),
            (self.top_authors_json, "Top Authors JSON"),
            (self.collaboration_summaries_csv, "Collaboration Summaries CSV"),
            (self.cross_topic_analysis_json, "Cross-Topic Analysis JSON")
        ]
        
        missing_required = []
        for file_path, description in required_files:
            if file_path and file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {description}: {file_path} ({size_mb:.1f} MB)")
            else:
                missing_required.append((file_path, description))
                logger.error(f"  ❌ {description}: {file_path} NOT FOUND")
        
        for file_path, description in optional_files:
            if file_path and file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {description}: {file_path} ({size_mb:.1f} MB)")
            else:
                logger.warning(f"  ⚠️  {description}: {file_path} NOT FOUND (optional)")
        
        if missing_required:
            logger.error("❌ Missing required files. Please check your file paths.")
            return False
        
        logger.info("✅ Source file validation complete!")
        return True
    
    def create_database_schema(self):
        """Create the database tables and indexes."""
        logger.info("🏗️  Creating database schema...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Table 1: Topics (from common_topics.csv)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                topic_id INTEGER PRIMARY KEY,
                count INTEGER NOT NULL,
                descriptive_label TEXT NOT NULL,
                primary_category TEXT NOT NULL,
                percent_of_corpus REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table 2: Papers (from compact_docs_with_topics.csv)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                topic_id INTEGER,
                date DATE,
                url TEXT,
                primary_category TEXT,
                authors_formatted TEXT,
                abstract_snippet TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)
        
        # Table 3: Topic Keywords (from JSON)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_keywords (
                topic_id INTEGER,
                keyword TEXT,
                weight REAL,
                rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (topic_id, rank),
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)
        
        # Table 4: Topic Category Distribution (from JSON)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_category_distribution (
                topic_id INTEGER,
                category TEXT,
                percentage REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (topic_id, category),
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)
        
        # Table 5: Top Authors by Topic (from JSON)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_top_authors (
                topic_id INTEGER,
                author_name TEXT,
                paper_count INTEGER,
                rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (topic_id, rank),
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)

        # Table 6: Topic Collaboration Metrics (from CSV)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_collaboration_metrics (
                topic_id INTEGER PRIMARY KEY,
                total_papers INTEGER,
                collaboration_papers INTEGER,
                collaboration_rate REAL,
                num_authors INTEGER,
                num_collaborations INTEGER,
                network_density REAL,
                num_components INTEGER,
                largest_component_fraction REAL,
                repeat_collaboration_rate REAL,
                power_law_fit BOOLEAN,
                power_law_exponent REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)

        # Table 7: Cross-Topic Author Profiles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS author_cross_topic_profiles(
                author_name TEXT PRIMARY KEY,
                primary_topic INTEGER,
                primary_topic_papers INTEGER,
                total_papers INTEGER,
                num_topics INTEGER,
                cross_topic_collaborations INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )            
        """)

        # Table 8: Team Size Distributions 
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_team_size_distributions (
                topic_id INTEGER,
                team_size INTEGER,
                paper_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (topic_id, team_size),
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)        
            )
        """)

                # Table 9: Advanced Network Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_advanced_metrics (
                topic_id INTEGER PRIMARY KEY,
                team_diversity REAL,
                degree_centralization REAL,
                betweenness_centralization REAL,
                closeness_centralization REAL,
                eigenvector_centralization REAL,
                max_k_core INTEGER,
                core_size_ratio REAL,
                coreness REAL,
                modularity REAL,
                num_communities INTEGER,
                avg_community_size REAL,
                community_size_gini REAL,
                largest_community_ratio REAL,
                random_robustness REAL,
                targeted_robustness REAL,
                robustness_ratio REAL,
                avg_newcomer_rate REAL,
                preferential_attachment_strength REAL,
                is_small_world BOOLEAN,
                small_world_coefficient REAL,
                avg_path_length REAL,
                clustering_vs_random REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)

        # Table 10: Degree Distribution Analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_degree_analysis (
                topic_id INTEGER PRIMARY KEY,
                min_degree INTEGER,
                max_degree INTEGER,
                mean_degree REAL,
                median_degree REAL,
                std_degree REAL,
                power_law_fit BOOLEAN,
                power_law_exponent REAL,
                power_law_r_squared REAL,
                power_law_p_value REAL,
                power_law_good_fit BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)

        # Table 11: Community Sizes (top 10 per topic)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_community_sizes (
                topic_id INTEGER,
                rank INTEGER,
                community_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (topic_id, rank),
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)

        # Table 12: Detailed Collaboration Patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_collaboration_patterns (
                topic_id INTEGER PRIMARY KEY,
                total_unique_pairs INTEGER,
                repeat_collaborations INTEGER,
                repeat_collaboration_rate REAL,
                max_pair_collaborations INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics (topic_id)
            )
        """)

        
        
        # Create indexes for fast queries
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_topics_category ON topics(primary_category)",
            "CREATE INDEX IF NOT EXISTS idx_topics_count ON topics(count DESC)",
            "CREATE INDEX IF NOT EXISTS idx_papers_topic ON papers(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(primary_category)",
            "CREATE INDEX IF NOT EXISTS idx_keywords_topic ON topic_keywords(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_authors_topic ON topic_top_authors(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_collab_metrics_topic ON topic_collaboration_metrics(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_cross_topic_author ON author_cross_topic_profiles(author_name)",
            "CREATE INDEX IF NOT EXISTS idx_team_size_topic ON topic_team_size_distributions(topic_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        # Create views for common dashboard queries
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS dashboard_summary AS
            SELECT 
                primary_category,
                COUNT(*) as topic_count,
                SUM(count) as total_papers,
                AVG(count) as avg_papers_per_topic,
                MAX(count) as max_papers_in_topic
            FROM topics 
            WHERE primary_category LIKE 'math.%'
            GROUP BY primary_category
            ORDER BY total_papers DESC
        """)
        
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS top_topics_by_category AS
            SELECT 
                topic_id,
                descriptive_label,
                count,
                primary_category,
                percent_of_corpus,
                ROW_NUMBER() OVER (PARTITION BY primary_category ORDER BY count DESC) as rank_in_category
            FROM topics
            WHERE primary_category LIKE 'math.%'
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database schema created successfully!")
    
    def migrate_topics(self):
        """Migrate data from common_topics.csv to topics table."""
        logger.info("📊 Migrating topics data...")
        
        try:
            # Read the CSV file
            topics_df = pd.read_csv(self.topics_csv)
            logger.info(f"  Loaded {len(topics_df)} topics from CSV")
            
            # Rename columns to match database schema
            topics_df = topics_df.rename(columns={
                'topic': 'topic_id'
            })
            
            # Ensure we have all required columns
            required_columns = ['topic_id', 'count', 'descriptive_label', 'primary_category']
            missing_columns = [col for col in required_columns if col not in topics_df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Add percent_of_corpus if not present
            if 'percent_of_corpus' not in topics_df.columns:
                total_papers = topics_df['count'].sum()
                topics_df['percent_of_corpus'] = (topics_df['count'] / total_papers) * 100
                logger.info("  Added calculated percent_of_corpus column")
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            topics_df.to_sql('topics', conn, if_exists='replace', index=False)
            conn.close()
            
            logger.info(f"✅ Successfully migrated {len(topics_df)} topics")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating topics: {e}")
            return False
    
    def format_authors_optimized(self, authors_str):
        """
        Optimized author formatting - converts nested list format to readable string.
        Input: '[["Smith", "John", ""], ["Doe", "Jane", ""]]'
        Output: "John Smith, Jane Doe"
        """
        if pd.isna(authors_str) or not authors_str or authors_str == '[]':
            return "Unknown Authors"
        
        try:
            # Handle string representation of list
            if isinstance(authors_str, str):
                authors_list = ast.literal_eval(authors_str)
            else:
                authors_list = authors_str
            
            formatted_names = []
            
            for author_parts in authors_list:
                if isinstance(author_parts, list) and len(author_parts) >= 1:
                    # Handle different formats
                    if len(author_parts) >= 2:
                        last_name = str(author_parts[0]).strip()
                        first_name = str(author_parts[1]).strip()
                        
                        if first_name and last_name:
                            formatted_names.append(f"{first_name} {last_name}")
                        elif last_name:
                            formatted_names.append(last_name)
                    elif len(author_parts) == 1:
                        name = str(author_parts[0]).strip()
                        if name:
                            formatted_names.append(name)
            
            result = ", ".join(formatted_names) if formatted_names else "Unknown Authors"
            return result
            
        except (ValueError, SyntaxError, TypeError) as e:
            # If parsing fails, try to extract something useful from the string
            if isinstance(authors_str, str) and len(authors_str) > 10:
                # Try to extract names from malformed string
                import re
                names = re.findall(r'"([^"]+)"', authors_str)
                if names:
                    return ", ".join(names[:3])  # Limit to first 3 names
            
            return "Unknown Authors"
    
    def migrate_papers(self):
        """Migrate data from compact_docs_with_topics.csv to papers table."""
        logger.info("📄 Migrating papers data (this may take several minutes)...")
        
        try:
            # Check if file exists and get size
            if not self.papers_csv.exists():
                logger.error(f"Papers CSV not found: {self.papers_csv}")
                return False
            
            file_size_mb = self.papers_csv.stat().st_size / (1024 * 1024)
            logger.info(f"  Processing file: {file_size_mb:.1f} MB")
            
            # Process in chunks to manage memory
            chunk_size = 5000
            total_processed = 0
            total_chunks = 0
            
            conn = sqlite3.connect(self.db_path)
            
            # Process CSV in chunks
            for chunk_df in pd.read_csv(self.papers_csv, chunksize=chunk_size):
                total_chunks += 1
                logger.info(f"  Processing chunk {total_chunks} ({len(chunk_df)} rows)...")
                
                # Pre-process authors (expensive operation done once)
                if 'authors' in chunk_df.columns:
                    chunk_df['authors_formatted'] = chunk_df['authors'].apply(
                        self.format_authors_optimized
                    )
                else:
                    chunk_df['authors_formatted'] = "Unknown Authors"
                
                # Create abstract snippets to save space
                if 'abstract' in chunk_df.columns:
                    chunk_df['abstract_snippet'] = chunk_df['abstract'].astype(str).str[:200]
                else:
                    chunk_df['abstract_snippet'] = ""
                
                # Select and rename columns for database
                column_mapping = {
                    'topic': 'topic_id',
                    # Other columns keep same names
                }
                
                # Select only needed columns
                db_columns = ['id', 'title', 'topic_id', 'date', 'url', 
                             'primary_category', 'authors_formatted', 'abstract_snippet']
                
                # Rename columns
                chunk_processed = chunk_df.rename(columns=column_mapping)
                
                # Select only columns that exist and are needed
                available_columns = [col for col in db_columns if col in chunk_processed.columns]
                chunk_final = chunk_processed[available_columns].copy()
                
                # Handle missing columns
                for col in db_columns:
                    if col not in chunk_final.columns:
                        if col == 'topic_id':
                            chunk_final[col] = 0  # Default topic
                        else:
                            chunk_final[col] = ""  # Default empty string
                
                # Insert into database
                chunk_final.to_sql('papers', conn, if_exists='append', index=False)
                
                total_processed += len(chunk_df)
                
                # Progress update every 10 chunks
                if total_chunks % 10 == 0:
                    logger.info(f"  Progress: {total_processed:,} papers processed...")
            
            conn.close()
            
            logger.info(f"✅ Successfully migrated {total_processed:,} papers in {total_chunks} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating papers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def migrate_keywords(self):
        """Migrate keyword data from JSON file."""
        logger.info("🔤 Migrating keywords data...")
        
        if not self.keywords_json.exists():
            logger.warning(f"  Keywords file not found: {self.keywords_json}")
            return True  # Not critical
        
        try:
            with open(self.keywords_json, 'r') as f:
                keywords_data = json.load(f)
            
            logger.info(f"  Loaded keywords for {len(keywords_data)} topics")
            
            # Convert to database format
            keywords_rows = []
            for topic_id_str, keyword_list in keywords_data.items():
                topic_id = int(topic_id_str)
                
                for rank, (keyword, weight) in enumerate(keyword_list, 1):
                    keywords_rows.append({
                        'topic_id': topic_id,
                        'keyword': keyword,
                        'weight': float(weight),
                        'rank': rank
                    })
                    
                    # Limit to top 20 keywords per topic
                    if rank >= 20:
                        break
            
            # Insert into database
            if keywords_rows:
                keywords_df = pd.DataFrame(keywords_rows)
                conn = sqlite3.connect(self.db_path)
                keywords_df.to_sql('topic_keywords', conn, if_exists='replace', index=False)
                conn.close()
                
                logger.info(f"✅ Successfully migrated {len(keywords_rows)} keyword entries")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating keywords: {e}")
            return False
    
    def migrate_category_distribution(self):
        """Migrate category distribution data from JSON file."""
        logger.info("📈 Migrating category distribution data...")
        
        if not self.category_dist_json.exists():
            logger.warning(f"  Category distribution file not found: {self.category_dist_json}")
            return True  # Not critical
        
        try:
            with open(self.category_dist_json, 'r') as f:
                category_data = json.load(f)
            
            logger.info(f"  Loaded category distributions for {len(category_data)} topics")
            
            # Convert to database format
            distribution_rows = []
            for topic_id_str, category_dict in category_data.items():
                topic_id = int(topic_id_str)
                
                for category, percentage in category_dict.items():
                    distribution_rows.append({
                        'topic_id': topic_id,
                        'category': category,
                        'percentage': float(percentage)
                    })
            
            # Insert into database
            if distribution_rows:
                dist_df = pd.DataFrame(distribution_rows)
                conn = sqlite3.connect(self.db_path)
                dist_df.to_sql('topic_category_distribution', conn, if_exists='replace', index=False)
                conn.close()
                
                logger.info(f"✅ Successfully migrated {len(distribution_rows)} category distribution entries")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating category distribution: {e}")
            return False
    
    def migrate_top_authors(self):
        """Migrate top authors data from JSON file."""
        logger.info("👥 Migrating top authors data...")
        
        if not self.top_authors_json.exists():
            logger.warning(f"  Top authors file not found: {self.top_authors_json}")
            return True  # Not critical
        
        try:
            with open(self.top_authors_json, 'r') as f:
                authors_data = json.load(f)
            
            # Handle nested structure: {"top_authors_by_topic": {topic_id: {"authors": [...]}}}
            if "top_authors_by_topic" in authors_data:
                authors_by_topic = authors_data["top_authors_by_topic"]
            else:
                authors_by_topic = authors_data
            
            logger.info(f"  Loaded top authors for {len(authors_by_topic)} topics")
            
            # Convert to database format
            authors_rows = []
            for topic_id_str, topic_data in authors_by_topic.items():
                topic_id = int(topic_id_str)
                
                # Handle different JSON structures
                if isinstance(topic_data, dict) and "authors" in topic_data:
                    author_list = topic_data["authors"]
                elif isinstance(topic_data, list):
                    author_list = topic_data
                else:
                    continue
                
                for rank, author_info in enumerate(author_list, 1):
                    if isinstance(author_info, dict):
                        author_name = author_info.get('name', 'Unknown')
                        paper_count = author_info.get('count', 0)
                    else:
                        # Handle simpler format
                        author_name = str(author_info)
                        paper_count = 1
                    
                    authors_rows.append({
                        'topic_id': topic_id,
                        'author_name': author_name,
                        'paper_count': paper_count,
                        'rank': rank
                    })
                    
                    # Limit to top 10 authors per topic
                    if rank >= 10:
                        break
            
            # Insert into database
            if authors_rows:
                authors_df = pd.DataFrame(authors_rows)
                conn = sqlite3.connect(self.db_path)
                authors_df.to_sql('topic_top_authors', conn, if_exists='replace', index=False)
                conn.close()
                
                logger.info(f"✅ Successfully migrated {len(authors_rows)} top author entries")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating top authors: {e}")
            return False
    
    def migrate_collaboration_metrics(self):
        """Migrate collaboration metrics from CSV file."""
        logger.info("🤝 Migrating collaboration metrics data...")
        
        if not self.collaboration_summaries_csv or not self.collaboration_summaries_csv.exists():
            logger.warning(f"  Collaboration summaries file not found")
            return True  # Not critical
        
        try:
            # Read the CSV file
            collab_df = pd.read_csv(self.collaboration_summaries_csv)
            logger.info(f"  Loaded collaboration metrics for {len(collab_df)} topics")
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            collab_df.to_sql('topic_collaboration_metrics', conn, if_exists='replace', index=False)
            conn.close()
            
            logger.info(f"✅ Successfully migrated {len(collab_df)} collaboration metric entries")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating collaboration metrics: {e}")
            return False

    def migrate_cross_topic_authors(self):
        """Migrate cross-topic author data from JSON file."""
        logger.info("🌐 Migrating cross-topic author data...")
        
        if not self.cross_topic_analysis_json or not self.cross_topic_analysis_json.exists():
            logger.warning(f"  Cross-topic analysis file not found")
            return True  # Not critical
        
        try:
            with open(self.cross_topic_analysis_json, 'r') as f:
                cross_topic_data = json.load(f)
            
            # Extract author primary topics data
            author_profiles = cross_topic_data.get('author_primary_topics', {})
            cross_topic_analysis = cross_topic_data.get('cross_topic_analysis', {})
            cross_topic_counts = cross_topic_analysis.get('cross_topic_collaboration_counts', {})
            
            logger.info(f"  Loaded profiles for {len(author_profiles)} authors")
            
            # Convert to database format
            author_rows = []
            for author_name, profile in author_profiles.items():
                cross_topic_collabs = cross_topic_counts.get(author_name, 0)
                
                author_rows.append({
                    'author_name': author_name,
                    'primary_topic': profile.get('primary_topic'),
                    'primary_topic_papers': profile.get('primary_topic_papers'),
                    'total_papers': profile.get('total_papers'),
                    'num_topics': profile.get('num_topics'),
                    'cross_topic_collaborations': cross_topic_collabs
                })
            
            # Insert into database
            if author_rows:
                authors_df = pd.DataFrame(author_rows)
                conn = sqlite3.connect(self.db_path)
                authors_df.to_sql('author_cross_topic_profiles', conn, if_exists='replace', index=False)
                conn.close()
                
                logger.info(f"✅ Successfully migrated {len(author_rows)} author profile entries")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating cross-topic authors: {e}")
            return False

    def migrate_team_size_distributions(self):
        """Migrate team size distribution data from topic analysis."""
        logger.info("👨‍👩‍👧‍👦 Migrating team size distributions...")
        
        # We need to extract this from the detailed topic analysis file
        topic_analysis_pattern = Path("results/collaboration_analysis/topic_analysis_*.json")
        analysis_files = list(topic_analysis_pattern.parent.glob(topic_analysis_pattern.name))
        
        if not analysis_files:
            logger.warning("  No topic analysis JSON found for team size data")
            return True  # Not critical
        
        try:
            # Use the latest file
            latest_analysis = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_analysis, 'r') as f:
                topic_data = json.load(f)
            
            logger.info(f"  Processing team size data from {len(topic_data)} topics")
            
            # Convert to database format
            team_size_rows = []
            for topic_id, analysis in topic_data.items():
                team_sizes = analysis.get('team_size_distribution', {})
                
                for team_size, paper_count in team_sizes.items():
                    team_size_rows.append({
                        'topic_id': int(topic_id),
                        'team_size': int(team_size),
                        'paper_count': paper_count
                    })
            
            # Insert into database
            if team_size_rows:
                team_df = pd.DataFrame(team_size_rows)
                conn = sqlite3.connect(self.db_path)
                team_df.to_sql('topic_team_size_distributions', conn, if_exists='replace', index=False)
                conn.close()
                
                logger.info(f"✅ Successfully migrated {len(team_size_rows)} team size entries")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating team size distributions: {e}")
            return False
    
    def optimize_database(self):
        """Optimize database for performance."""
        logger.info("⚡ Optimizing database for performance...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update database statistics for query optimization
        cursor.execute("ANALYZE")
        
        # Set performance settings
        performance_settings = [
            "PRAGMA journal_mode = WAL",        # Write-Ahead Logging for better performance
            "PRAGMA synchronous = NORMAL",      # Balance between speed and safety
            "PRAGMA cache_size = 10000",        # 10MB cache
            "PRAGMA temp_store = memory",       # Store temporary data in memory
            "PRAGMA mmap_size = 268435456",     # Memory-mapped I/O (256MB)
        ]
        
        for setting in performance_settings:
            cursor.execute(setting)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database optimization complete!")
    
    def validate_migration(self):
        """Validate that migration was successful."""
        logger.info("🔍 Validating migration results...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Test queries
        validation_tests = [
            ("Total topics", "SELECT COUNT(*) FROM topics"),
            ("Total papers", "SELECT COUNT(*) FROM papers"),
            ("Math categories", "SELECT COUNT(DISTINCT primary_category) FROM topics WHERE primary_category LIKE 'math.%'"),
            ("Topics with keywords", "SELECT COUNT(DISTINCT topic_id) FROM topic_keywords"),
            ("Topics with collaboration metrics", "SELECT COUNT(*) FROM topic_collaboration_metrics"),
            ("Cross-topic authors", "SELECT COUNT(*) FROM author_cross_topic_profiles"),
            ("Team size distributions", "SELECT COUNT(*) FROM topic_team_size_distributions"),
            ("Largest topic", "SELECT descriptive_label, count FROM topics ORDER BY count DESC LIMIT 1"),
            ("Sample paper", "SELECT title FROM papers WHERE authors_formatted != 'Unknown Authors' LIMIT 1"),
        ]
        
        results = {}
        for test_name, query in validation_tests:
            try:
                result = cursor.execute(query).fetchone()
                if result:
                    results[test_name] = result[0] if len(result) == 1 else result
                    logger.info(f"  ✅ {test_name}: {results[test_name]}")
                else:
                    logger.warning(f"  ⚠️  {test_name}: No results")
            except Exception as e:
                logger.error(f"  ❌ {test_name}: Error - {e}")
        
        # Check database size
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
        logger.info(f"  📊 Database size: {db_size_mb:.1f} MB")
        
        conn.close()
        
        # Summary
        logger.info("\n📋 Migration Summary:")
        for test_name, result in results.items():
            logger.info(f"  {test_name}: {result}")
        
        logger.info("✅ Validation complete!")
        return results
    
    def run_complete_migration(self):
        """Run the complete migration process."""
        start_time = datetime.now()
        logger.info("🚀 Starting complete database migration...")
        logger.info(f"Start time: {start_time}")
        
        # Step 1: Validate source files
        if not self.validate_source_files():
            logger.error("❌ Migration aborted due to missing files")
            return False
        
        # Step 2: Create database schema
        self.create_database_schema()
        
        # Step 3: Migrate core data
        success_steps = []
        
        if self.migrate_topics():
            success_steps.append("Topics")
        
        if self.migrate_papers():
            success_steps.append("Papers")
        
        # Step 4: Migrate supporting data (non-critical)
        if self.migrate_keywords():
            success_steps.append("Keywords")
        
        if self.migrate_category_distribution():
            success_steps.append("Category Distribution")
        
        if self.migrate_top_authors():
            success_steps.append("Top Authors")

        # Step 5: Migrate collaboration data (NEW)
        if self.migrate_collaboration_metrics():
            success_steps.append("Collaboration Metrics")

        if self.migrate_cross_topic_authors():
            success_steps.append("Cross-Topic Authors")

        if self.migrate_team_size_distributions():
            success_steps.append("Team Size Distributions")
        
        # Step 6: Optimize database
        self.optimize_database()
        
        # Step 7: Validate results
        validation_results = self.validate_migration()
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n🎉 Migration completed!")
        logger.info(f"Duration: {duration}")
        logger.info(f"Database created: {self.db_path}")
        logger.info(f"Successful steps: {', '.join(success_steps)}")
        
        if 'Total topics' in validation_results and 'Total papers' in validation_results:
            logger.info(f"✅ Ready to use: {validation_results['Total topics']} topics, {validation_results['Total papers']} papers")
            return True
        else:
            logger.error("❌ Migration may have failed - check validation results")
            return False


def main():
    """Main function to run the migration."""
    print("=" * 60)
    print("Math Research Compass - Database Migration")
    print("=" * 60)
    
    # Create migrator instance
    migrator = MathResearchDatabaseMigrator()
    
    # Check if database already exists
    if migrator.db_path.exists():
        response = input(f"\nDatabase already exists at {migrator.db_path}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return
        
        # Backup existing database
        backup_path = migrator.db_path.with_suffix('.db.backup')
        migrator.db_path.rename(backup_path)
        logger.info(f"Existing database backed up to {backup_path}")
    
    # Run migration
    success = migrator.run_complete_migration()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ MIGRATION SUCCESSFUL!")
        print("=" * 60)
        print(f"Your optimized database is ready at: {migrator.db_path}")
        print("\nNext steps:")
        print("1. Update your Shiny app to use the new OptimizedDataManager")
        print("2. Test the performance improvements")
        print("3. Deploy to production with the optimized database")
        print("\nThe old CSV files are still available as backup.")
    else:
        print("\n" + "=" * 60)
        print("❌ MIGRATION FAILED")
        print("=" * 60)
        print("Please check the error messages above and ensure all CSV files are present.")
        sys.exit(1)


if __name__ == "__main__":
    main()