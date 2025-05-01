# Math Research Compass: arXiv Trends Dashboard 🧭

**An interactive data science dashboard exploring research trends, emerging topics, and collaboration structures across mathematical subfields on arXiv.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Dashboard Screenshot](results/images/dashboard_preview.png)

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Current Status](#current-status)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Source](#data-source)
- [Analysis Methodology](#analysis-methodology)
- [Results & Visualizations](#results--visualizations)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

## Project Overview
Math Research Compass is a data science project designed to provide insights into the recent landscape of mathematical research published on arXiv. It fetches preprint metadata, analyzes publication trends across different subfields (currently: Algebraic Geometry, Algebraic Topology, Representation Theory, Symplectic Geometry), identifies emerging research topics using Natural Language Processing (NLP), and explores potential collaboration patterns. The ultimate goal is to present these findings in an interactive web dashboard.

## Motivation
Navigating the vast amount of research on arXiv can be challenging, especially for students or researchers entering a new field. This project aims to:
1.  Provide data-driven insights into the activity and focus areas within specific Mathematics subfields on arXiv.
2.  Identify potentially "hot" or interdisciplinary research topics based on abstract/title analysis.
3.  Explore collaboration dynamics by looking at co-authorship patterns.
4.  Track how research topics emerge, evolve, and decline over time.

## Features
*   **Data Pipeline:** Fetches and preprocesses metadata for arXiv preprints within specified math categories and date ranges.
*   **Trend Analysis:** Calculates and visualizes:
    *   Yearly/Monthly publication counts per category.
    *   Absolute and relative growth rates for different subfields.
* **Topic Modeling:** Identifies key research themes using:
  * BERTopic for semantic clustering of abstracts/titles, using Sentence-Transformers, UMAP, and HDBSCAN.
  * Hierarchical topic representation to show relationships between research areas.
  * Input texts are lemmatized using the `spaCy` NLP pipeline to improve topic quality and reduce lexical fragmentation.
*   **Temporal Topic Analysis:** Tracks how topics evolve over time, showing growth and decline patterns.
*   **Emerging Topics Detection:** Automatically identifies growing and declining research areas.
*   **Interdisciplinarity Analysis:**
    *   Identifies frequently co-listed category pairs.
    *   Analyzes keywords specific to papers cross-listed between fields.
*   **Collaboration Analysis:** Identifies top authors and frequent co-author pairs within fields using NetworkX.
*   **Interactive Dashboard:** Presents analyses through interactive charts and visualizations using Streamlit and Plotly.

## Technology Stack
*   **Core:** Python 3.x, Pandas, NumPy
*   **Data Acquisition:** `arxiv` (Python library for arXiv API)
*   **Data Processing:** Pandas
*   **NLP:** NLTK (tokenization, stopwords), spaCy (lemmatization), BERTopic, Sentence-Transformers
*   **Clustering:** UMAP (dimensionality reduction), HDBSCAN (clustering)
*   **Visualization:** Plotly Express, Matplotlib, Seaborn, WordCloud
*   **Dashboarding:** Streamlit
*   **Network Analysis:** NetworkX
*   **Version Control:** Git, GitHub
*   **Environment:** `venv` (Python virtual environments)

## Current Status
**In Progress**
*   ✅ Data acquisition pipeline implemented (`fetch_data.py`).
*   ✅ Data cleaning and preprocessing script functional (`process_data.py`).
*   ✅ Initial growth trend analysis implemented.
*   ✅ BERTopic modeling implemented, including hierarchical topic analysis.
*   ✅ Topic temporal tracking and emerging topics identification.
*   ✅ Streamlit dashboard with multiple views (overview, topic explorer, temporal trends).
*   ✅ Collaboration network analysis and visualization.
*   ⏳ Fine-tuning topic representations with LLM-based labels.
*   ⏳ Deployment of the Streamlit application to cloud services.

## Setup and Installation

### Prerequisites
- Python 3.8+ installed
- Virtual environment tool (venv, conda, etc.)
- Git (for cloning the repository)

### Step 1: Clone the repository
```bash
git clone https://github.com/brian-hepler-phd/MathResearchCompass.git
cd MathResearchCompass
```

### Step 2: Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download required models and data
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

### Option 1: Run the complete pipeline
This script will fetch data, run analysis, and launch the dashboard:

```bash
python run_analysis_pipeline.py
```

### Option 2: Run individual components

#### 1. Fetch Data
Modify parameters in `fetch_data.py` (categories, dates) if desired, then run:
```bash
python fetch_data.py
```
This will generate the raw CSV file (e.g., `arxiv_math_papers_raw.csv`).

#### 2. Process Data
Run the cleaning script:
```bash
python process_data.py
```
This uses the raw CSV and outputs a cleaned version (e.g., `arxiv_math_papers_cleaned.csv`).

#### 3. Run Topic Analysis
Execute the topic analysis script to generate visualizations and trend data:
```bash
python topic_trends_analyzer.py --subjects math.AG math.AT --years 5
```

#### 4. Launch the Dashboard
```bash
streamlit run app.py
```
This will start the Streamlit server and open the dashboard in your web browser.

## Project Structure

```text
MathResearchCompass/
├── venv/                           # Virtual environment directory
├── compass/                        # Package with core utilities
│   ├── __init__.py                 # Package initialization
│   ├── io.py                       # Data loading and saving functions
│   ├── metrics.py                  # Growth trend metrics
│   ├── topic.py                    # Topic modeling utilities
│   └── network.py                  # Collaboration network analysis
├── data/                           # Directory for data files
│   ├── raw/                        # Raw data from fetch script
│   └── cleaned/                    # Cleaned data from processing
├── results/                        # Analysis results
│   ├── plots/                      # Generated visualizations
│   ├── topics/                     # Topic modeling results
│   └── images/                     # Static images for documentation
├── scripts/                        # Analysis and utility scripts
│   ├── fetch_data.py               # Script to fetch data from arXiv
│   ├── process_data.py             # Data cleaning/preprocessing script
│   └── analyze_data.py             # Data analysis script
├── notebooks/                      # Jupyter notebooks
│   ├── TF-IDF_playground.ipynb     # TF-IDF analysis notebook
│   └── BERT_Playground.ipynb       # BERTopic analysis notebook
├── topic_trends_analyzer.py        # Advanced topic trends analysis
├── app.py                          # Streamlit dashboard
├── run_analysis_pipeline.py        # End-to-end pipeline runner
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Data Source
*   **Source:** [arXiv API](https://arxiv.org/help/api/index)
*   **Access Method:** Python `arxiv` library
*   **Scope:** Metadata (title, authors, abstract, categories, dates) for preprints primarily categorized under [ 'math.AG', 'math.AT', 'math.RT', 'math.SG'] between 2019-01-01 and 2025-04-30.

## Analysis Methodology

### Topic Modeling Approach
Our topic modeling approach uses BERTopic, which combines:
1. **Sentence Embeddings**: Using transformer-based models (e.g., all-MiniLM-L6-v2) to create high-quality document representations.
2. **Dimensionality Reduction**: UMAP transforms the high-dimensional embeddings into a space suitable for clustering.
3. **Clustering**: HDBSCAN identifies document clusters that represent coherent topics.
4. **Topic Representation**: Keywords are extracted using class-based TF-IDF (c-TF-IDF) for interpretable topic representation.
5. **Hierarchical Topic Structure**: Topics are organized in a hierarchical structure to show relationships between research areas.

### Temporal Trends Analysis
- Documents are grouped by time periods (month, quarter, or year)
- Topic frequencies are calculated for each time period
- Growth rates and trends are identified using linear regression on topic frequency curves
- Topics are classified as "emerging" or "declining" based on normalized slope values

### Collaboration Network Analysis
- Co-authorship networks are constructed using NetworkX
- Author relationships are weighted by the number of co-authored papers
- Network metrics (degree, centrality) identify key collaborative researchers
- Community detection algorithms identify research groups

## Results & Visualizations

### Topic Hierarchy Visualization
The dashboard provides an interactive visualization of how research topics are related hierarchically:

![Topic Hierarchy Example](results/images/hierarchical_clustering_AG.png)

### Temporal Trends
Visualizations show how topic frequencies evolve over time:

![Topic Trends Example](results/images/publ_volume_chart.png)

### Emerging Topics
The system automatically identifies research areas with rapid growth:

![Emerging Topics Example](results/images/emerging_topics.png)

## Future Work
*   [ ] Enhance topic labeling using LLMs for more descriptive topic names
*   [ ] Expand analysis to include more mathematical subjects
*   [ ] Incorporate citation data for impact analysis
*   [ ] Add citation network analysis to complement collaboration networks
*   [ ] Deploy as a public web service for broader accessibility
*   [ ] Add predictive modeling to forecast future research trends
*   [ ] Improve topic stability across time periods
*   [ ] Develop dynamic topic models to better capture evolving research interests

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. 

## Contact
Brian Hepler - [GitHub Profile](https://github.com/brian-hepler-phd) - [LinkedIn Profile](https://www.linkedin.com/in/brian-hepler-phd/)
