#!/usr/bin/env python3
"""
Simple Data Structure Validator
------------------------------
Quick script to check your dataset structure without running the full analysis.
"""

import pandas as pd
from pathlib import Path
import ast

def validate_data_structure():
    """Validate the structure of your dataset."""
    print("🔍 DATASET VALIDATION")
    print("=" * 40)
    
    # Check if file exists
    data_file = Path("data/cleaned/author_topic_networks.csv")
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        print("   Please check the file path.")
        return False
    
    print(f"✅ Found data file: {data_file}")
    
    try:
        # Load just a few rows to inspect structure
        df_sample = pd.read_csv(data_file, nrows=5)
        print(f"✅ File loads successfully")
        print(f"   Columns found: {list(df_sample.columns)}")
        print(f"   Sample shape: {df_sample.shape}")
        
        # Check required columns
        required_cols = ['topic', 'authors_parsed']
        missing_cols = [col for col in required_cols if col not in df_sample.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print("✅ Required columns present")
        
        # Show sample data
        print("\n📋 SAMPLE DATA:")
        print("-" * 20)
        for i, row in df_sample.iterrows():
            print(f"Row {i+1}:")
            print(f"  Topic: {row.get('topic', 'N/A')}")
            print(f"  Authors: {str(row.get('authors_parsed', 'N/A'))[:100]}...")
            if 'year' in row:
                print(f"  Year: {row.get('year', 'N/A')}")
            if 'quarter' in row:
                print(f"  Quarter: {row.get('quarter', 'N/A')}")
            print()
        
        # Check full dataset stats
        print("📊 DATASET STATISTICS:")
        print("-" * 25)
        
        # Read just the topic column for stats
        df_topics = pd.read_csv(data_file, usecols=['topic'])
        total_papers = len(df_topics)
        unique_topics = sorted(df_topics['topic'].unique())
        
        print(f"Total papers: {total_papers:,}")
        print(f"Unique topics: {len(unique_topics)}")
        print(f"Topic range: {min(unique_topics)} to {max(unique_topics)}")
        
        # Show topic distribution for first 10 topics
        topic_counts = df_topics['topic'].value_counts().sort_index()
        print(f"\nSample topic distribution:")
        for topic in unique_topics[:10]:
            count = topic_counts.get(topic, 0)
            print(f"  Topic {topic}: {count:,} papers")
        
        if len(unique_topics) > 10:
            print(f"  ... and {len(unique_topics) - 10} more topics")
        
        # Test author parsing on a few examples
        print(f"\n🧪 AUTHOR PARSING TEST:")
        print("-" * 25)
        
        def parse_authors_test(authors_str):
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
            except:
                return []
        
        # Test on sample data
        for i, row in df_sample.iterrows():
            authors_raw = row.get('authors_parsed', '')
            authors_parsed = parse_authors_test(authors_raw)
            collab_status = "✅ Collaboration" if len(authors_parsed) > 1 else "❌ Single author"
            
            print(f"Row {i+1}: {len(authors_parsed)} authors - {collab_status}")
            if authors_parsed:
                authors_display = ", ".join(authors_parsed[:3])  # Show first 3
                if len(authors_parsed) > 3:
                    authors_display += f", ... (+{len(authors_parsed)-3} more)"
                print(f"  Authors: {authors_display}")
        
        print(f"\n✅ VALIDATION COMPLETE")
        print(f"📁 Your dataset appears to be properly formatted!")
        print(f"🚀 Ready for network analysis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def check_specific_topic(topic_id: int = 0):
    """Check data for a specific topic."""
    print(f"\n🎯 CHECKING TOPIC {topic_id}")
    print("=" * 30)
    
    try:
        data_file = Path("data/cleaned/author_topic_networks.csv")
        
        # Read data for specific topic
        df = pd.read_csv(data_file)
        topic_df = df[df['topic'] == topic_id]
        
        if topic_df.empty:
            print(f"❌ No papers found for topic {topic_id}")
            return False
        
        print(f"✅ Found {len(topic_df)} papers for topic {topic_id}")
        
        # Parse authors and check for collaborations
        def parse_authors(authors_str):
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
            except:
                return []
        
        topic_df = topic_df.copy()
        topic_df['authors_list'] = topic_df['authors_parsed'].apply(parse_authors)
        
        # Filter collaboration papers
        collab_papers = topic_df[topic_df['authors_list'].apply(lambda x: len(x) > 1)]
        
        print(f"📊 Topic {topic_id} Analysis:")
        print(f"   Total papers: {len(topic_df)}")
        print(f"   Collaboration papers: {len(collab_papers)}")
        print(f"   Single-author papers: {len(topic_df) - len(collab_papers)}")
        
        if len(collab_papers) > 0:
            print(f"   ✅ Ready for network analysis!")
            
            # Show a few collaboration examples
            print(f"\n📝 Sample collaborations:")
            for i, (_, paper) in enumerate(collab_papers.head(3).iterrows()):
                authors = paper['authors_list']
                print(f"   {i+1}. {len(authors)} authors: {', '.join(authors)}")
        else:
            print(f"   ❌ No collaboration papers found for topic {topic_id}")
        
        return len(collab_papers) > 0
        
    except Exception as e:
        print(f"❌ Error checking topic {topic_id}: {e}")
        return False

if __name__ == "__main__":
    # Run validation
    success = validate_data_structure()
    
    if success:
        # Check specific topic 0
        check_specific_topic(0)
        
        print(f"\n🎉 READY TO PROCEED!")
        print(f"You can now run:")
        print(f"   python combined_network_analysis.py --test-run")
    else:
        print(f"\n❌ Please fix dataset issues before proceeding.")
