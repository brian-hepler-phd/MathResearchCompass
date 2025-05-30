from shiny import App, ui, render, reactive
from optimized_data_manager import OptimizedDataManager
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

# --- Load the optimized data manager ---
print("🚀 Initializing Math Research Compass with optimized database...")
data_manager = OptimizedDataManager()
print("✅ Database connection established successfully!")

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

# Get available categories from the database
print("📋 Loading category choices...")
dropdown_choices = data_manager.get_category_choices()
print(f"✅ Found {len(dropdown_choices)} categories")

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
                ui.a("bhepler.com", href="https://bhepler.com", target="_blank")
            ], style="text-align: center; margin-bottom: 20px;"),
        ),
        
        # Add a category dropdown at the top
        ui.row(
            ui.column(
                6,
                ui.input_select(
                    "category",
                    "Filter by Primary Math Category:",
                    choices=dropdown_choices,
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
)

def server(input, output, session):
    # --- OVERVIEW PAGE FUNCTIONALITY ---
    # Reactive filtered dataframe based on primary category - now using optimized database
    @reactive.Calc
    def filtered_data():
        return data_manager.get_topics_by_category(input.category())
    
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
    
    # Calculate stats based on filtered data - now using optimized queries
    @reactive.Calc
    def stats():
        filtered_df = filtered_data()
        if filtered_df.empty:
            return {"papers": 0, "topics": 0}
        
        total_papers = filtered_df['count'].sum()
        total_topics = len(filtered_df)  # Use len() instead of nunique() since each row is a unique topic
        return {"papers": total_papers, "topics": total_topics}
    
    # Output the statistics
    @output
    @render.text
    def total_papers():
        return f"{stats()['papers']:,}"  # Add comma formatting for readability
    
    @output
    @render.text
    def total_topics():
        return f"{stats()['topics']:,}"  # Add comma formatting for readability
    
    @output
    @render.ui
    def top_topics_plot():
        # Get filtered data from optimized database
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
    # Filtered data for explorer page - now using optimized database
    @reactive.Calc
    def filtered_explorer_data():
        selected_category = input.explorer_category()
        return data_manager.get_topics_by_category(selected_category)
    
    # Get topic choices for the selected category
    @reactive.Calc
    def get_topic_choices():
        filtered_df = filtered_explorer_data()
        if filtered_df.empty:
            return []
            
        # Create a list of (topic_id, descriptive_label) tuples for the dropdown
        topic_choices = [
            (str(row['topic_id']), f"Topic {row['topic_id']}: {row['descriptive_label']}") 
            for _, row in filtered_df.iterrows()
        ]
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
    
    # Get the selected topic data - now using optimized database
    @reactive.Calc
    def selected_topic_data():
        if not hasattr(input, 'selected_topic') or not input.selected_topic():
            return None
            
        topic_id = int(input.selected_topic())
        return data_manager.get_topic_details(topic_id)
    
    # Topic header with dynamic title
    @output
    @render.ui
    def topic_header():
        topic_data = selected_topic_data()
        
        if topic_data is None:
            return ui.div()  # Empty div if no topic selected
            
        topic_info = topic_data['info']
        return ui.div(
            ui.h2(f"Topic {topic_info['topic_id']}: {topic_info['descriptive_label']}", 
                 class_="text-center",
                 style="margin-top: 30px; margin-bottom: 20px;"
            ),
            ui.hr()
        )
    
    # Author List - now using database data
    @output
    @render.ui
    def explorer_top_authors_list():
        topic_data = selected_topic_data()
        
        if topic_data is None:
            return ui.p("Please select a topic to see authors.")
        
        top_authors_list = topic_data.get('top_authors', [])
        
        if not top_authors_list:
            return ui.p("No author data available for this topic.")

        author_items = []
        max_count = top_authors_list[0]['paper_count'] if top_authors_list else 1
        
        for idx, author_info in enumerate(top_authors_list[:10]):
            author_name = author_info['author_name']
            paper_count = author_info['paper_count']
            
            # Format author name (handle "Last, First" format)
            formatted_name = author_name
            if ',' in author_name:
                name_parts = author_name.split(',', 1)
                if len(name_parts) == 2:
                    last_name = name_parts[0].strip()
                    first_name = name_parts[1].strip()
                    if first_name and last_name:
                        formatted_name = f"{first_name} {last_name}"
            
            percentage = (paper_count / max_count) * 100 if max_count > 0 else 0
            
            author_items.append(
                ui.div(
                    ui.div(
                        f"{idx+1}. {formatted_name}",
                        style="display: inline-block; width: 70%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; vertical-align: middle;"
                    ),
                    ui.div(
                        f"{paper_count} papers",
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

    # Category Distribution Plot - now using database data
    @output
    @render.ui
    def explorer_category_dist_plot():
        topic_data = selected_topic_data()
        
        if topic_data is None:
            return ui.p("Please select a topic to see category distribution.")
        
        category_distribution = topic_data.get('category_distribution', {})
        
        if not category_distribution:
            return ui.p("No category distribution data available for this topic.")

        # Convert to DataFrame for plotting
        cat_df = pd.DataFrame(
            list(category_distribution.items()), 
            columns=['category', 'percent']
        )
        
        # Add readable category labels
        cat_df['category_label'] = cat_df['category'].apply(
            lambda x: math_category_labels.get(x, x)
        )
        
        # Get top 5 categories and sort for display
        cat_df = cat_df.sort_values('percent', ascending=False).head(5)
        cat_df = cat_df.sort_values('percent', ascending=True)
        
        if cat_df.empty:
            return ui.p("No categories to plot for this topic.")

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
        
    # Representative articles - now using database data
    @output
    @render.ui
    def representative_articles():
        topic_data = selected_topic_data()
        
        if topic_data is None:
            return ui.p("Please select a topic to see representative articles.")
        
        papers_list = topic_data.get('papers', [])
        
        if not papers_list:
            return ui.p("No representative articles available for this topic.")

        article_cards = []
        
        for paper in papers_list:
            title = paper.get('title', 'N/A Title')
            authors = paper.get('authors_formatted', 'N/A Authors')
            date_str = paper.get('date', 'N/A Date')
            url = paper.get('url', '#')
            
            # Format date if it's a valid date
            if date_str and date_str != 'N/A Date':
                try:
                    date_obj = pd.to_datetime(date_str)
                    date_str = date_obj.strftime("%b %d, %Y")
                except:
                    pass  # Keep original date_str if parsing fails

            article_cards.append(
                ui.card(
                    ui.card_header(
                        ui.h5(title, style="margin: 0; font-weight: bold; white-space: normal; word-wrap: break-word;")
                    ),
                    ui.div(
                        ui.p(
                            ui.tags.i(class_="fa-solid fa-user-pen", style="margin-right: 5px;"),
                            authors,
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
print("🎉 Math Research Compass ready to launch!")
app = App(app_ui, server)

# Heroku deployment configuration
if __name__ == "__main__":
    
    # Heroku provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Important: bind to all interfaces for Heroku
    
    print(f"🚀 Starting Math Research Compass on {host}:{port}")
    print("📊 Database-powered for lightning-fast performance!")
    print("🌐 Ready for professional deployment on Heroku!")
    
    app.run(host=host, port=port)