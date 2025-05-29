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
5. Optimizes database for fast queries

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
        
        logger.info(f"Database will be created at: {self.db_path}")
    
    def validate_source_files(self):
        """Check that all required source files exist."""
        logger.info("🔍 Validating source files...")
        
        required_files = [
            (self.topics_csv, "Topics CSV"),
            (self.papers_csv, "Papers CSV")
        ]
        
        optional_files = [
            (self.keywords_json, "Keywords JSON"),
            (self.category_dist_json, "Category Distribution JSON"),
            (self.top_authors_json, "Top Authors JSON")
        ]
        
        missing_required = []
        for file_path, description in required_files:
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {description}: {file_path} ({size_mb:.1f} MB)")
            else:
                missing_required.append((file_path, description))
                logger.error(f"  ❌ {description}: {file_path} NOT FOUND")
        
        for file_path, description in optional_files:
            if file_path.exists():
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
        
        # Create indexes for fast queries
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_topics_category ON topics(primary_category)",
            "CREATE INDEX IF NOT EXISTS idx_topics_count ON topics(count DESC)",
            "CREATE INDEX IF NOT EXISTS idx_papers_topic ON papers(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(primary_category)",
            "CREATE INDEX IF NOT EXISTS idx_keywords_topic ON topic_keywords(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_authors_topic ON topic_top_authors(topic_id)"
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
        
        # Step 5: Optimize database
        self.optimize_database()
        
        # Step 6: Validate results
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
        print("3. Deploy to Railway with the optimized database")
        print("\nThe old CSV files are still available as backup.")
    else:
        print("\n" + "=" * 60)
        print("❌ MIGRATION FAILED")
        print("=" * 60)
        print("Please check the error messages above and ensure all CSV files are present.")
        sys.exit(1)


if __name__ == "__main__":
    main()
