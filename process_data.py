import pandas as pd
import re
import ast # For safely evaluating string representations of lists

# --- Configuration ---
INPUT_FILENAME = "arxiv_math_papers_raw_v2.csv" 
OUTPUT_FILENAME = "arxiv_math_papers_cleaned.csv"
COLUMNS_TO_DROP_NA = ['arxiv_id', 'title', 'published_date', 'authors', 'primary_category'] # Define essential columns

# --- Load Data ---
print(f"Loading data from {INPUT_FILENAME}...")
try:
    df = pd.read_csv(INPUT_FILENAME)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILENAME}' not found. Please ensure the fetch script ran correctly.")
    exit()

# --- Initial Inspection ---
print("\n--- Initial DataFrame Info ---")
print(df.info())
print("\n--- First 5 Rows (Raw) ---")
print(df.head())
print(f"\nInitial number of rows: {len(df)}")

# --- Data Type Conversion & Cleaning ---

# 1. Convert Date Columns
print("\nConverting date columns...")
# Use the correct column name 'published_date'
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
# Also convert 'updated_date' if it exists and you need it
if 'updated_date' in df.columns:
    df['updated_date'] = pd.to_datetime(df['updated_date'], errors='coerce')

# 2. Drop Duplicate Papers (based on arxiv_id)
print("Dropping duplicate entries based on arxiv_id...")
initial_rows = len(df)
df.drop_duplicates(subset=['arxiv_id'], keep='first', inplace=True)
print(f"Removed {initial_rows - len(df)} duplicate rows.")

# 3. Handle Missing Values in Essential Columns
print(f"Dropping rows with missing values in: {COLUMNS_TO_DROP_NA}...")
initial_rows = len(df)
# Apply dropna *after* date conversion, as 'coerce' might create NaT (which count as NA)
df.dropna(subset=COLUMNS_TO_DROP_NA, inplace=True)
print(f"Removed {initial_rows - len(df)} rows with missing essential data.")
# Optional: Fill missing abstracts with an empty string if you want to keep those rows
# df['abstract'].fillna('', inplace=True)

# 4. Clean Text Columns (Title and Abstract)
print("Cleaning text in 'title' and 'abstract' columns...")
def clean_text(text):
    if not isinstance(text, str):
        return "" # Return empty string if data is not text (e.g., float NaN)
    # Optional: Remove HTML tags if needed
    # try:
    #     text = BeautifulSoup(text, "html.parser").get_text()
    # except Exception as e:
    #     print(f"HTML parsing error for text: {text[:50]}... - {e}") # Log error
    #     pass # Continue even if parsing fails for some text
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    return text.lower()

# Use the correct column name 'abstract'
df['title'] = df['title'].apply(clean_text)
df['abstract'] = df['abstract'].apply(clean_text) # Use 'abstract' here

# 5. Parse List-like String Columns (Authors and Categories)
print("Parsing 'authors' and 'categories' columns...")
def parse_list_string(list_str):
    try:
        # Check if it's already a list (might happen if loading from formats other than CSV)
        if isinstance(list_str, list):
            return list_str
        # Check for non-string types (like NaN/float) that can occur after drops/fills
        if not isinstance(list_str, str):
            return [] # Return empty list for non-string inputs
        # Safely evaluate the string representation
        parsed_list = ast.literal_eval(list_str)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            return [] # Return empty list if parsing didn't result in a list
    except (ValueError, SyntaxError, TypeError):
        # Handle cases where the string is not a valid list representation
        # print(f"Warning: Could not parse list string: {list_str}") # Uncomment for debugging
        return [] # Return empty list on error

df['authors'] = df['authors'].apply(parse_list_string)
df['categories'] = df['categories'].apply(parse_list_string)

# --- Final Inspection ---
print("\n--- Cleaned DataFrame Info ---")
print(df.info())
print("\n--- First 5 Rows (Cleaned) ---")
print(df.head())
print(f"\nFinal number of rows: {len(df)}")
print("\nExample Authors entry (parsed):", df['authors'].iloc[0] if not df.empty else "N/A")
print("Example Categories entry (parsed):", df['categories'].iloc[0] if not df.empty else "N/A")


# --- Save Cleaned Data ---
print(f"\nSaving cleaned data to {OUTPUT_FILENAME}...")
df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
print("Data cleaning complete.")