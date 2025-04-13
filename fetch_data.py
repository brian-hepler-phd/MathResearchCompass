import arxiv
import pandas as pd
import time
from datetime import datetime, date # Import date too
from tqdm import tqdm

# --- Parameters ---
MATH_CATEGORIES = [
    'math.AG', 'math.AT', 'math.RT', 'math.SG'
]

# Define dates as actual date objects for easier comparison
# Use datetime.strptime to parse the string dates
START_DATE_OBJ = datetime.strptime("20190101", "%Y%m%d").date()
END_DATE_OBJ = datetime.strptime("20231231", "%Y%m%d").date()

MAX_RESULTS_PER_QUERY = 1000 # page_size for the client
SLEEP_TIME_SECONDS = 3
OUTPUT_FILENAME = 'arxiv_math_papers_raw_v2.csv' # Use a new filename

# -----------------
"""
The Query Loop
"""
#------------------

all_papers_data = []
client = arxiv.Client(
    page_size=MAX_RESULTS_PER_QUERY,
    delay_seconds=SLEEP_TIME_SECONDS,
    num_retries=5
)

print(f"Fetching papers published between {START_DATE_OBJ} and {END_DATE_OBJ}...")

# Using tqdm for progress bar over categories
for category in tqdm(MATH_CATEGORIES, desc="Processing Categories"):
    print(f"\nFetching category: {category} (will filter by date locally)")
    # --- MODIFIED: Simplified Query ---
    # Query ONLY by category. We will filter dates later.
    search_query = f'cat:{category}'
    # ---------------------------------

    try:
        # Use arxiv.Search - tell it to try and get ALL results for the category
        # The client will still paginate based on page_size
        search = arxiv.Search(
            query=search_query,
            max_results=float('inf'), # Try to get all results matching the category query
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Ascending # Or Descending if you want newest first
        )

        results_generator = client.results(search)
        category_papers_found = 0
        category_papers_kept = 0

        # Iterate through ALL results for the category returned by the generator
        for result in tqdm(results_generator, desc=f"Scanning {category}"):
            category_papers_found += 1
            # --- ADDED: Date Filtering ---
            # Compare result's published date (ignoring time part) with our range
            # Make sure result.published is a datetime object
            paper_date = result.published.date()
            if START_DATE_OBJ <= paper_date <= END_DATE_OBJ:
                # --- Keep Primary Category Filter ---
                if result.primary_category == category:
                    paper_data = {
                        'arxiv_id': result.entry_id.split('/')[-1],
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'abstract': result.summary.replace("\n", " "),
                        'categories': result.categories,
                        'primary_category': result.primary_category,
                        # Use result.published - seems more appropriate than updated for "when appeared"
                        'published_date': result.published.strftime('%Y-%m-%d'),
                        'updated_date': result.updated.strftime('%Y-%m-%d'),
                        'pdf_url': result.pdf_url
                    }
                    all_papers_data.append(paper_data)
                    category_papers_kept += 1
            # Optional: If fetching in ascending order, maybe break early if paper_date > END_DATE_OBJ?
            # elif paper_date > END_DATE_OBJ and search.sort_order == arxiv.SortOrder.Ascending:
            #    print(f"Reached paper date {paper_date}, stopping early for {category}")
            #    break # Optimization if sorting ascendingly

        print(f"Scanned {category_papers_found} total papers for {category}. Kept {category_papers_kept} primary papers within date range.")

    except Exception as e:
        # This might still happen for other reasons, keep the handler
        print(f"!!! Error processing category {category}: {e}")


print("\nFetching and filtering complete.")

# -----------------
"""
Storing the Data
"""
#------------------

if all_papers_data:
    print(f"Saving {len(all_papers_data)} papers to {OUTPUT_FILENAME}...")
    df = pd.DataFrame(all_papers_data)
    df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
    print("Data saved successfully.")
else:
    print("No papers found matching the criteria.")