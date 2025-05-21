from shiny import App, ui, render, reactive
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import json
import ast  # For safely evaluating string representations of lists
import base64
import io
from wordcloud import WordCloud
import tempfile
import os

# --- Load the data ---

df = pd.read_csv('results/topics/common_topics.csv')

compact_docs_df = pd.read_csv('data/cleaned/compact_docs_with_topics.csv')

# --- PRE-PROCESSING compact_docs_df ---
def format_authors_for_preprocessing(authors_str):
    """
    Formats the author string from the CSV into a readable string.
    Example input: "[['Bhatt', 'Bhargav', ''], ['Blickle', 'Manuel', '']]"
    Example output: "Bhargav Bhatt, Manuel Blickle"
    """
    if pd.isna(authors_str) or not authors_str:
        return "N/A Authors"
    try:
        authors_list_of_lists = ast.literal_eval(authors_str)
        formatted_names = []
        for author_parts in authors_list_of_lists:
            if isinstance(author_parts, list) and len(author_parts) >= 1: # Check for at least one part
                # Prioritize first name then last name, handle missing parts
                first_name = author_parts[1].strip() if len(author_parts) > 1 and author_parts[1] else ""
                last_name = author_parts[0].strip() if author_parts[0] else ""
                
                if first_name and last_name:
                    formatted_names.append(f"{first_name} {last_name}")
                elif last_name: # Only last name
                    formatted_names.append(last_name)
                elif first_name: # Only first name
                    formatted_names.append(first_name)
                # else: could add a case for malformed author_parts if needed
        return ", ".join(formatted_names) if formatted_names else "N/A Authors"
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing authors string '{authors_str}': {e}")
        # Fallback for unparseable strings
        if isinstance(authors_str, str) and not authors_str.startswith('['):
            return authors_str # If it's already a plain string, return it
        return "Error Parsing Authors"

if 'authors' in compact_docs_df.columns:
    print("Preprocessing 'authors' column in compact_docs_df...")
    compact_docs_df['authors_formatted_display'] = compact_docs_df['authors'].apply(format_authors_for_preprocessing)
    print("Finished preprocessing 'authors' column.")
else:
    print("Warning: 'authors' column not found in compact_docs_df. 'authors_formatted_display' will be N/A.")
    compact_docs_df['authors_formatted_display'] = "N/A Authors"

# Convert 'date' column to datetime objects during initial load for efficiency
if 'date' in compact_docs_df.columns:
    print("Preprocessing 'date' column in compact_docs_df...")
    compact_docs_df['date'] = pd.to_datetime(compact_docs_df['date'], errors='coerce')
    print("Finished preprocessing 'date' column.")
else:
    print("Warning: 'date' column not found in compact_docs_df.")

# Create a mapping dictionary for math categories to their descriptions
math_category_labels = {
    "math.AC": "math.AC - Commutative Algebra",
    "math.AG": "math.AG - Algebraic Geometry",
    "math.AP": "math.AP - Analysis of PDEs",
    "math.AT": "math.AT - Algebraic Topology",
    "math.CA": "math.CA - Classical Analysis and ODEs",
    "math.CO": "math.CO - Combinatorics",
    "math.CT": "math.CT - Category Theory",
    "math.CV": "math.CV - Complex Variables",
    "math.DG": "math.DG - Differential Geometry",
    "math.DS": "math.DS - Dynamical Systems",
    "math.FA": "math.FA - Functional Analysis",
    "math.GM": "math.GM - General Mathematics",
    "math.GN": "math.GN - General Topology",
    "math.GR": "math.GR - Group Theory",
    "math.GT": "math.GT - Geometric Topology",
    "math.HO": "math.HO - History and Overview",
    "math.IT": "math.IT - Information Theory",
    "math.KT": "math.KT - K-Theory and Homology",
    "math.LO": "math.LO - Logic",
    "math.MG": "math.MG - Metric Geometry",
    "math.MP": "math.MP - Mathematical Physics",
    "math.NA": "math.NA - Numerical Analysis",
    "math.NT": "math.NT - Number Theory",
    "math.OA": "math.OA - Operator Algebras",
    "math.OC": "math.OC - Optimization and Control",
    "math.PR": "math.PR - Probability",
    "math.QA": "math.QA - Quantum Algebra",
    "math.RA": "math.RA - Rings and Algebras",
    "math.RT": "math.RT - Representation Theory",
    "math.SG": "math.SG - Symplectic Geometry",
    "math.ST": "math.ST - Statistics Theory"
}

# Load additional data for the Topic Explorer
try:
    # Load topic keywords data
    with open('results/topics/topic_keywords_20250509_221839.json', 'r') as f:
        topic_keywords = json.load(f)
        
    # Load category distribution data
    with open('results/topics/topic_category_distribution.json', 'r') as f:
        topic_category_dist = json.load(f)
        
    # Load top authors data
    with open('results/topics/top_authors_by_topic.json', 'r') as f:
        top_authors = json.load(f)
except FileNotFoundError as e:
    print(f"Warning: Some data files not found: {e}")
    # Initialize empty dictionaries if files not found
    if 'topic_keywords' not in locals():
        topic_keywords = {}
    if 'topic_category_dist' not in locals():
        topic_category_dist = {}
    if 'top_authors' not in locals():
        top_authors = {}

# Extract unique primary categories for the dropdown
if 'primary_category' in df.columns:
    # Get unique primary categories and sort them
    unique_primary_categories = sorted(df['primary_category'].unique())
    
    # Filter to math categories if desired
    math_categories = [cat for cat in unique_primary_categories if str(cat).startswith('math.')]
    
    if math_categories:
        unique_categories = math_categories
    else:
        unique_categories = unique_primary_categories

    # Create a list of display labels for the dropdown
    dropdown_choices = ["All Math Categories"]
    category_mapping = {}
    
    for category in unique_categories:
        if category in math_category_labels:
            # Use the friendly label from our mapping
            display_label = math_category_labels[category]
            dropdown_choices.append(display_label)
            category_mapping[display_label] = category
        else:
            # If category isn't in our mapping, use as-is
            dropdown_choices.append(category)
            category_mapping[category] = category

# Print categories for debugging
print(f"Found {len(unique_categories)} primary math categories")

# Create the Shiny app with a tab layout
app_ui = ui.page_navbar(

    ui.nav_panel(
        "Overview",
        ui.tags.head(
            ui.tags.style("""
                .value-box { text-align: center; }
                .value-box .value { font-size: 2rem; }
                .project-description { 
                    text-align: center; 
                    max-width: 800px; 
                    margin: 0 auto 30px auto; 
                    line-height: 1.6;
                    color: #555;
                }
            """)
        ),
        ui.panel_title(
            None,
            "Math Research Compass"
        ),
        ui.div(
            ui.h1("Math Research Compass", class_="text-center"),
            ui.h4("Created by Brian Hepler", class_="text-center"),
            style="margin-top: 5px; margin-bottom: 20px;"
        ),
        
        # Project description
        ui.div(
            ui.p("""
                Math Research Compass analyzes ArXiv preprints to identify trending research topics across mathematical subfields from the past 5 years. 
                This interactive dashboard visualizes topic modeling results from thousands of recent mathematics papers, 
                helping researchers and students discover emerging areas and popular research directions. 
                The analysis uses advanced natural language processing to cluster semantically related papers and identify coherent research themes.
                 
                Check out the Overview page for a high-level view of topics in each mathematics ArXiv category, or visit the Topic Explorer tab for a deeper-dive into each topic. 
            """),
            class_="project-description"
        ),

        # Links to GitHub and Personal Website
        ui.div(
            ui.p([
                "View the full documentation on ",
                ui.a("GitHub", href="https://github.com/brian-hepler-phd/MathResearchCompass", target="_blank"),
                " | Visit the creator's website at ",
                ui.a("bhepler.com", href="https://bhepler.com",target="_blank")
            ], style="text-align: center; margin-bottom: 20px;"),
        ),
        
        # Add a category dropdown at the top
        ui.row(
            ui.column(
                6,
                ui.input_select(
                    "category",
                    "Filter by Primary Math Category:",
                    choices=dropdown_choices,  # Use our new choices with descriptive labels
                    selected="All Math Categories",
                    width="100%"
                ),
                offset=3
            )
        ),
        
        # Add explanation of primary category
        ui.row(
            ui.column(
                8,
                ui.p("""
                    Topics are shown based on their primary category - the category that appears most frequently 
                    as the main category across all papers in that topic. 
                """, style="text-align: center; font-style: italic; color: #666;"),
                offset=2
            )
        ),
        
        ui.hr(),
        
        # Summary statistics cards
        ui.layout_columns(
            ui.value_box(
                title=ui.output_text("papers_title"),
                value=ui.output_text("total_papers"),
                showcase=ui.tags.i(class_="fa-solid fa-file-lines"),
                theme="primary",
            ),
            ui.value_box(
                title=ui.output_text("topics_title"),
                value=ui.output_text("total_topics"),
                showcase=ui.tags.i(class_="fa-solid fa-diagram-project"),
                theme="primary",
            ),
            col_widths=(6, 6),
        ),
        
        # Top topics bar chart
        ui.card(
            ui.card_header(ui.output_text("plot_title")),
            ui.output_ui("top_topics_plot"),
        ),
    ),
    
    # Topic Explorer tab
    ui.nav_panel(
        "Topic Explorer",
        ui.panel_title(
            ui.h1("Topic Explorer", class_="text-center"),
        ),
        
        # Category and Topic selection dropdowns
        ui.row(
            ui.column(
                4,
                ui.input_select(
                    "explorer_category",
                    "Filter by Primary Math Category:",
                    choices=dropdown_choices,
                    selected="All Math Categories",
                    width="100%"
                ),
                offset=2
            ),
            ui.column(
                4,
                ui.output_ui("topic_dropdown_container")
            )
        ),
        
        # Topic header (shown conditionally when a topic is selected)
        ui.output_ui("topic_header"),

        # Plots row
        ui.row(
            # left column - Top Authors
            ui.column(
                6,
                ui.card(
                    ui.card_header("Top Contributing Authors"),
                    ui.output_ui("explorer_top_authors_list")
                )
            ),
            # right column - Category distribution
            ui.column(
                6,
                ui.card(
                    ui.card_header("Where Can You Find This Topic?"),
                    ui.output_ui("explorer_category_dist_plot")
                )
            )

        ),
        
        # Summary card with details about the selected topic
        ui.output_ui("representative_articles")
    ),
    
    id="navbar",
    navbar_options=ui.navbar_options(position="static-top"),
    #title="Math Research Compass"
)

def server(input, output, session):
    # --- OVERVIEW PAGE FUNCTIONALITY ---
    # Reactive filtered dataframe based on primary category
    @reactive.Calc
    def filtered_data():
        selected_category = input.category()
        
        if selected_category == "All Math Categories":
            # For "All Math Categories", return all math-related topics
            return df[df['primary_category'].str.startswith('math.', na=False)]
        else:
            # Map the display label back to the actual category code
            actual_category = selected_category.split(" - ")[0] if " - " in selected_category else selected_category
            # Filter to topics with the selected primary category
            return df[df['primary_category'] == actual_category]
    
    # Dynamic titles
    @output
    @render.text
    def papers_title():
        if input.category() == "All Math Categories":
            return "Total Math Papers"
        else:
            # Extract just the full name part for display
            if " - " in input.category():
                category_name = input.category().split(" - ")[1]
                return f"Papers in {category_name}"
            else:
                return f"Papers in {input.category()}"
    
    @output
    @render.text
    def topics_title():
        if input.category() == "All Math Categories":
            return "Math Topics Discovered"
        else:
            # Extract just the full name part for display
            if " - " in input.category():
                category_name = input.category().split(" - ")[1]
                return f"Topics in {category_name}"
            else:
                return f"Topics in {input.category()}"
    
    @output
    @render.text
    def plot_title():
        if input.category() == "All Math Categories":
            return "Top Math Research Topics"
        else:
            # Extract just the full name part for display
            if " - " in input.category():
                category_name = input.category().split(" - ")[1]
                return f"Top Research Topics in {category_name}"
            else:
                return f"Top Research Topics in {input.category()}"
    
    # Calculate stats based on filtered data
    @reactive.Calc
    def stats():
        filtered_df = filtered_data()
        total_papers = filtered_df['count'].sum()
        total_topics = filtered_df['topic'].nunique()
        return {"papers": total_papers, "topics": total_topics}
    
    # Output the statistics
    @output
    @render.text
    def total_papers():
        return str(stats()["papers"])
    
    @output
    @render.text
    def total_topics():
        return str(stats()["topics"])
    
    @output
    @render.ui
    def top_topics_plot():
        # Get filtered data
        filtered_df = filtered_data()
        
        # Check if we have data to display
        if filtered_df.empty:
            return ui.p("No topics found for the selected category.")
        
        # Get top 10 topics by count (or all if fewer than 10)
        top_n = min(10, len(filtered_df))
        top_topics = filtered_df.sort_values('count', ascending=False).head(top_n)
        
        # Create the bar chart using descriptive_label column
        fig = px.bar(
            top_topics,
            y='descriptive_label',
            x='count',
            orientation='h',
            labels={'count': 'Number of Papers', 'descriptive_label': 'Topic'},
            color='count',
            color_continuous_scale="viridis",
            hover_data=['primary_category']  # Show primary category on hover
        )
        
        # Update layout for better appearance
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20),
            height=500
        )
        
        # Return a ui.tags.iframe with the Plotly figure
        return ui.tags.iframe(
            srcDoc=fig.to_html(include_plotlyjs='cdn'),
            style="width:100%; height:500px; border:none;"
        )

    # --- TOPIC EXPLORER PAGE FUNCTIONALITY ---
    # Filtered data for explorer page
    @reactive.Calc
    def filtered_explorer_data():
        selected_category = input.explorer_category()
        
        if selected_category == "All Math Categories":
            # Return all math-related topics
            return df[df['primary_category'].str.startswith('math.', na=False)]
        else:
            # Map the display label back to the actual category code
            actual_category = selected_category.split(" - ")[0] if " - " in selected_category else selected_category
            # Filter to topics with the selected primary category
            return df[df['primary_category'] == actual_category]
    
    # Get topic choices for the selected category
    @reactive.Calc
    def get_topic_choices():
        filtered_df = filtered_explorer_data()
        if filtered_df.empty:
            return []
            
        # Create a list of (topic_id, descriptive_label) tuples for the dropdown
        topic_choices = [(str(row.topic), f"Topic {row.topic}: {row.descriptive_label}") 
                          for _, row in filtered_df.iterrows()]
        # Sort by topic ID
        topic_choices.sort(key=lambda x: int(x[0]))
        return topic_choices
    
    # Dynamic topic dropdown
    @output
    @render.ui
    def topic_dropdown_container():
        topic_choices = get_topic_choices()
        
        if not topic_choices:
            return ui.p("No topics available for this category.")
        
        # Create a dropdown with the topic choices
        return ui.input_select(
            "selected_topic",
            "Select Topic:",
            choices=dict(topic_choices),
            width="100%"
        )
    
    # Get the selected topic data
    @reactive.Calc
    def selected_topic_data():
        if not hasattr(input, 'selected_topic') or not input.selected_topic():
            return None
            
        topic_id = int(input.selected_topic())
        topic_row = df[df['topic'] == topic_id]
        
        if topic_row.empty:
            return None
            
        return topic_row.iloc[0].to_dict()  # Convert to dictionary for easier access
    
    # Topic header with dynamic title
    @output
    @render.ui
    def topic_header():
        topic_data = selected_topic_data()
        
        if topic_data is None:
            return ui.div()  # Empty div if no topic selected
            
        return ui.div(
            ui.h2(f"Topic {topic_data['topic']}: {topic_data['descriptive_label']}", 
                 class_="text-center",
                 style="margin-top: 30px; margin-bottom: 20px;"
            ),
            ui.hr()
        )
    # Author List
    @output
    @render.ui
    def explorer_top_authors_list():
        topic_data = selected_topic_data()
        
        if topic_data is None:
            return ui.p("Please select a topic to see authors.")
            
        topic_id = str(int(topic_data['topic']))
        authors_data_by_topic_dict = top_authors.get("top_authors_by_topic")

        if authors_data_by_topic_dict and topic_id in authors_data_by_topic_dict:
            topic_specific_author_data = authors_data_by_topic_dict[topic_id]
            authors_list_data_from_json = topic_specific_author_data.get("authors") 
            
            if not authors_list_data_from_json:
                return ui.p(f"Author list for Topic {topic_id} is empty or missing.")

            author_items = []
            # Ensure there's at least one author for max_count
            max_count = authors_list_data_from_json[0]["count"] if authors_list_data_from_json else 1 
            
            for idx, author_info_json in enumerate(authors_list_data_from_json[:10]):
                original_name = author_info_json['name']
                # Attempt to parse "LastName, FirstName"
                name_parts = original_name.split(',', 1) # Split only on the first comma
                formatted_name_display = original_name # Fallback to original
                if len(name_parts) == 2:
                    last_name_json = name_parts[0].strip()
                    first_name_json = name_parts[1].strip()
                    if first_name_json and last_name_json:
                         formatted_name_display = f"{first_name_json} {last_name_json}"
                    elif last_name_json: # Only last name part found after comma (unlikely but handle)
                         formatted_name_display = last_name_json
                    # No need for elif first_name_json, as split expects a comma
                
                percentage = (author_info_json["count"] / max_count) * 100 if max_count > 0 else 0
                
                author_items.append(
                    ui.div(
                        ui.div(
                            f"{idx+1}. {formatted_name_display}", # Use newly formatted name
                            style="display: inline-block; width: 70%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; vertical-align: middle;"
                        ),
                        ui.div(
                            f"{author_info_json['count']} papers",
                            style="display: inline-block; width: 25%; text-align: right; vertical-align: middle;"
                        ),
                        ui.div( # Progress bar
                            style=f"background-color: #4CAF50; height: 5px; width: {percentage}%; margin-top: 3px;"
                        ),
                        style="margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px;"
                    )
                )
            
            if not author_items:
                 return ui.p("No authors to display for this topic.")

            return ui.div(
                ui.p("Authors with the most papers in this topic:", style="font-style: italic; margin-bottom: 10px;"),
                ui.div(*author_items, style="margin-top: 5px;"),
                style="padding: 0 10px;"
            )
        else:
            print(f"Data for 'top_authors_by_topic' or specific topic_id '{topic_id}' not found in JSON.")
            return ui.p(f"No author data available for Topic {topic_id}.")    

    # Category Distribution Plot
    @output
    @render.ui
    def explorer_category_dist_plot(): # Matches fixed ID in UI
        topic_data = selected_topic_data()
        
        if topic_data is None:
            return ui.p("Please select a topic to see category distribution.")
            
        topic_id = str(int(topic_data['topic']))
        
        if topic_category_dist and topic_id in topic_category_dist:
            cat_dist = topic_category_dist[topic_id]
            
            if not cat_dist: # Check if the distribution itself is empty
                return ui.p(f"Category distribution data for Topic {topic_id} is empty.")

            cat_df = pd.DataFrame(list(cat_dist.items()), columns=['category', 'percent'])
            
            cat_df['category_label'] = cat_df['category'].apply(
                lambda x: math_category_labels.get(x, x)
            )
            
            cat_df = cat_df.sort_values('percent', ascending=False).head(5)
            cat_df = cat_df.sort_values('percent', ascending=True)
            
            if cat_df.empty:
                return ui.p(f"No categories to plot for Topic {topic_id} after filtering.")

            fig = px.bar(
                cat_df,
                y='category_label',
                x='percent',
                orientation='h',
                labels={'percent': 'Percentage of Topic Papers', 'category_label': 'Category'},
                color='percent',
                color_continuous_scale="viridis"
            )
            
            fig.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=400
            )
            
            return ui.tags.iframe(
                srcDoc=fig.to_html(include_plotlyjs='cdn'),
                style="width:100%; height:400px; border:none;"
            )
        else:
            print(f"No category distribution data in JSON for topic_id: {topic_id}")
            return ui.p(f"No category distribution data available for Topic {topic_id}.")
        
    # Helper function for representative articles section    
    @reactive.Calc
    def get_representative_articles_data():
        topic_data = selected_topic_data()
    
        if topic_data is None:
            # print("get_representative_articles_data: topic_data is None.") # Less noisy now
            return None
        
        # compact_docs_df is already loaded and preprocessed globally
        if 'compact_docs_df' not in globals() or compact_docs_df.empty:
            # print("get_representative_articles_data: compact_docs_df is not loaded or is empty.")
            return None
    
        topic_id = int(topic_data['topic'])
        # print(f"get_representative_articles_data: Processing topic_id {topic_id}")
    
        if 'topic' not in compact_docs_df.columns:
            # print("get_representative_articles_data: 'topic' column missing in compact_docs_df.")
            return None
        
        # Use the globally preprocessed compact_docs_df
        topic_papers_df = compact_docs_df[compact_docs_df['topic'] == topic_id] 
        # No .copy() needed here if we are only reading from it for selection.
        # If you were to modify topic_papers_df later *within this function* and didn't want
        # to affect the global compact_docs_df (which we are not doing for this selection logic),
        # then .copy() would be important.

        if topic_papers_df.empty:
            # print(f"get_representative_articles_data: No papers for topic {topic_id}")
            return None
        
        # 'date' and 'authors_formatted_display' columns are already preprocessed.
    
        topic_author_data = None
        topic_id_str = str(topic_id) 
        authors_data_by_topic_dict = top_authors.get("top_authors_by_topic", {})
    
        if topic_id_str in authors_data_by_topic_dict:
            topic_author_data = authors_data_by_topic_dict[topic_id_str]
    
        selected_papers_list = [] 
        processed_paper_ids = set()

        if topic_author_data and 'authors' in topic_author_data and topic_author_data['authors']:
            top_author_names_from_json = [author_info['name'] for author_info in topic_author_data['authors'][:3]]
        
            for author_name_json in top_author_names_from_json:
                if len(selected_papers_list) >= 5:
                    break
                
                searchable_name_part = author_name_json.split(',')[0].strip() 

                # Search in the original 'authors' string for flexibility, 
                # or you could search in 'authors_formatted_display' if JSON names match that format.
                # For now, sticking to searching the raw 'authors' column from CSV.
                author_papers = topic_papers_df[
                    topic_papers_df['authors'].astype(str).str.contains(searchable_name_part, case=False, na=False)
                ]
            
                if not author_papers.empty:
                    # Date column is already pd.Timestamp or NaT
                    author_papers_sorted = author_papers.sort_values('date', ascending=False, na_position='last')
                    
                    for _, paper_row in author_papers_sorted.iterrows():
                        if paper_row['id'] not in processed_paper_ids:
                            selected_papers_list.append(paper_row.to_dict())
                            processed_paper_ids.add(paper_row['id'])
                            if len(selected_papers_list) >= 5:
                                break
                    if len(selected_papers_list) >= 5:
                        break
        
        if len(selected_papers_list) < 5:
            remaining_topic_papers = topic_papers_df[~topic_papers_df['id'].isin(processed_paper_ids)]
            if not remaining_topic_papers.empty:
                additional_papers_df = remaining_topic_papers.sort_values('date', ascending=False, na_position='last').head(5 - len(selected_papers_list))
                
                for _, paper_row in additional_papers_df.iterrows():
                     if paper_row['id'] not in processed_paper_ids: 
                        selected_papers_list.append(paper_row.to_dict())
                        # processed_paper_ids.add(paper_row['id']) # Not strictly vital for this fill-up part

        # print(f"get_representative_articles_data: Selected {len(selected_papers_list)} papers.")
        return selected_papers_list[:5]
    
    # Rendering the article cards for representative articles section
    @output
    @render.ui
    def representative_articles():
        articles_data_list = get_representative_articles_data() 
    
        if not articles_data_list:
            topic_data = selected_topic_data()
            topic_id_msg = f"for Topic {topic_data['topic']}" if topic_data else ""
            return ui.p(f"No representative articles available {topic_id_msg}.")
    
        article_cards = []
    
        for article_dict in articles_data_list:
            title = article_dict.get('title', 'N/A Title')
            # Use the pre-processed 'authors_formatted_display' column
            authors_to_display = article_dict.get('authors_formatted_display', 'N/A Authors') 
            article_date_obj = article_dict.get('date') # This should be a Timestamp or NaT
            url = article_dict.get('url', '#')

            date_str = "N/A Date"
            if pd.notna(article_date_obj) and isinstance(article_date_obj, pd.Timestamp):
                date_str = article_date_obj.strftime("%b %d, %Y")
            elif article_date_obj is not None: # If it's NaT but not None, or other types
                 date_str = str(article_date_obj).split(' ')[0] # Attempt to get just date part

            article_cards.append(
                ui.card(
                    ui.card_header(
                        ui.h5(title, style="margin: 0; font-weight: bold; white-space: normal; word-wrap: break-word;")
                    ),
                    ui.div(
                        ui.p(
                            ui.tags.i(class_="fa-solid fa-user-pen", style="margin-right: 5px;"),
                            authors_to_display, # Use the formatted version
                            style="margin-bottom: 8px; white-space: normal; word-wrap: break-word;"
                        ),
                        ui.p(
                            ui.tags.i(class_="fa-solid fa-calendar-day", style="margin-right: 5px;"),
                            date_str,
                            style="margin-bottom: 12px;"
                        ),
                        ui.tags.a(
                            ui.tags.div(
                                ui.tags.i(class_="fa-solid fa-external-link", style="margin-right: 5px;"),
                                "View on arXiv",
                                style="display: inline-block;"
                            ),
                            href=url,
                            target="_blank",
                            class_="btn btn-sm btn-outline-primary"
                        ),
                        style="padding: 0 15px 15px 15px;"
                    ),
                    style="margin-bottom: 15px;"
                )
            )
            
        if not article_cards:
            return ui.p("No representative articles to display.")

        return ui.div(
            ui.h4("Representative Articles", class_="text-center", style="margin-top: 25px; margin-bottom: 15px;"),
            ui.div(*article_cards),
            style="margin-top: 10px;"
        )

# Create and run the app
app = App(app_ui, server)
