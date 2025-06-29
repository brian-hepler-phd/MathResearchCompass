from shiny import App, ui, render, reactive
from src.optimized_data_manager import OptimizedDataManager
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
import networkx as nx
from pathlib import Path
import random

# --- Load the optimized data manager ---
print("Initializing Math Research Compass with optimized database...")
data_manager = OptimizedDataManager()
print("Database connection established successfully!")

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
print("Loading category choices...")
dropdown_choices = data_manager.get_category_choices()
print(f"Found {len(dropdown_choices)} categories")

def load_collaboration_network(topic_id):
    """Load collaboration network from pre-computed files if they exist."""
    try:
        # Look for network files in results/collaboration_analysis/network_graphs/
        network_dir = Path("results/collaboration_analysis/network_graphs")
        network_file = network_dir / f"topic_{topic_id}_network.json"
        
        if network_file.exists():
            with open(network_file, 'r') as f:
                network_data = json.load(f)
            return network_data
        else:
            return None
    except Exception as e:
        print(f"Error loading network for topic {topic_id}: {e}")
        return None

def create_collaboration_network_plot(network_data):
    """Create Plotly network visualization from network data."""
    if not network_data or 'nodes' not in network_data or 'edges' not in network_data:
        return None
    
    nodes = network_data['nodes']
    edges = network_data['edges']
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in edges:
        x0, y0 = edge['x0'], edge['y0']
        x1, y1 = edge['x1'], edge['y1']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = [node['x'] for node in nodes]
    node_y = [node['y'] for node in nodes]
    node_text = [node['name'] for node in nodes]
    node_size = [node.get('size', 5) for node in nodes]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=node_size,
            size=[max(5, min(20, s)) for s in node_size],  # Scale sizes between 5-20
            colorbar=dict(
                thickness=15,
                len=0.5,
                x=1.02,
                title="Papers"
            ),
            line=dict(width=0.5, color='DarkSlateGrey')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Collaboration Network',
                        title_font_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Node size = papers published<br>Hover for author names",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                            font=dict(color="grey", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                    ))
    
    return fig

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
                .collab-metric { 
                    background: #f8f9fa; 
                    border-left: 4px solid #007bff; 
                    padding: 10px; 
                    margin: 5px 0; 
                    border-radius: 3px; 
                }
                .network-container {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px 0;
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

        # Collaboration Analysis Section for Overview
        ui.output_ui("overview_collaboration_section"),
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

        # Collaboration Metrics Row
        ui.output_ui("collaboration_metrics_section"),

        # Existing plots row
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

        # Network Visualization Row
        ui.output_ui("collaboration_network_section"),
        
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

    # NEW: Overview Collaboration Section
    @output
    @render.ui
    def overview_collaboration_section():
        # Get collaboration insights for the selected category
        try:
            collaboration_insights = data_manager.get_collaboration_insights(input.category())
            
            if not collaboration_insights or not collaboration_insights.get('top_collaborative_topics'):
                return ui.div()
            
            return ui.div(
                ui.h3("Collaboration Network Analysis", class_="text-center", style="margin-top: 30px; margin-bottom: 20px;"),
                ui.row(
                    # Most Collaborative Topics
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Most Collaborative Topics"),
                            ui.output_ui("most_collaborative_topics_plot")
                        )
                    ),
                    # Cross-Topic Collaborators
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Top Cross-Topic Collaborators"),
                            ui.output_ui("cross_topic_collaborators_list")
                        )
                    )
                ),
                style="margin-top: 20px;"
            )
        except Exception as e:
            # If collaboration features aren't available, return empty div
            print(f"Collaboration features not available: {e}")
            return ui.div()
    
    @output
    @render.ui
    def most_collaborative_topics_plot():
        try:
            collaboration_insights = data_manager.get_collaboration_insights(input.category())
            
            top_collab_topics = collaboration_insights.get('top_collaborative_topics', [])
            
            if not top_collab_topics:
                return ui.p("No collaboration data available for this category.")
            
            # Prepare data for plotting
            collab_df = pd.DataFrame(top_collab_topics)
            
            # Limit to top 8 for readability
            collab_df = collab_df.head(8)
            
            # Create horizontal bar chart
            fig = px.bar(
                collab_df,
                y='descriptive_label',
                x='collaboration_rate',
                orientation='h',
                labels={'collaboration_rate': 'Collaboration Rate', 'descriptive_label': 'Topic'},
                color='collaboration_rate',
                color_continuous_scale="viridis",
                hover_data=['network_density']
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=400
            )
            
            # Format x-axis as percentage
            fig.update_xaxes(tickformat='.0%')
            
            return ui.tags.iframe(
                srcDoc=fig.to_html(include_plotlyjs='cdn'),
                style="width:100%; height:420px; border:none;"
            )
        except Exception as e:
            return ui.p("Collaboration data not available.")
        
    @output
    @render.ui
    def topic_comparison_section():
        """Show how the current category's topics compare to overall patterns."""
        try:
            # Get comparison insights from database
            category = input.category()
            comparison_data = data_manager.get_topic_comparison_insights(category)
        
            if not comparison_data:
                return ui.div()
        
            return ui.div(
                ui.h3("Topic Network Patterns", class_="text-center", style="margin-top: 30px;"),
                ui.row(
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Network Topology Distribution"),
                            ui.output_ui("network_topology_scatter")
                        )
                    ),
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Collaboration Patterns by Topic Size"),
                            ui.output_ui("collaboration_patterns_plot")
                        )
                    )
                )
            )
        except:
            return ui.div()

    @output
    @render.ui
    def network_topology_scatter():
        """Scatter plot showing centralization vs modularity colored by topic size."""
        comparison_data = data_manager.get_topic_comparison_insights(input.category())
    
        if not comparison_data:
            return ui.p("No comparison data available.")
    
        df = pd.DataFrame(comparison_data['topics'])
    
        fig = px.scatter(
            df,
            x='degree_centralization',
            y='modularity',
            size='total_papers',
            color='small_world',
            hover_data=['topic_id', 'descriptive_label'],
            labels={
                'degree_centralization': 'Network Centralization',
                'modularity': 'Community Modularity',
                'small_world': 'Small-World Network'
            },
            title="Network Structure Landscape"
        )
    
        # Add quadrant labels
        fig.add_annotation(x=0.05, y=0.9, text="Isolated<br>Communities", showarrow=False, opacity=0.5)
        fig.add_annotation(x=0.2, y=0.9, text="Hub-dominated<br>Communities", showarrow=False, opacity=0.5)
        fig.add_annotation(x=0.05, y=0.3, text="Integrated<br>Egalitarian", showarrow=False, opacity=0.5)
        fig.add_annotation(x=0.2, y=0.3, text="Integrated<br>Hierarchical", showarrow=False, opacity=0.5)
    
        return ui.tags.iframe(
            srcDoc=fig.to_html(include_plotlyjs='cdn'),
            style="width:100%; height:400px; border:none;"
    )       
    
    @output
    @render.ui  
    def cross_topic_collaborators_list():
        try:

            collaboration_insights = data_manager.get_collaboration_insights(input.category())
            

            cross_topic_collaborators = collaboration_insights.get('cross_topic_collaborators', [])
            
            if not cross_topic_collaborators:
                return ui.p("No cross-topic collaboration data available.")
            
            # Simple filtering: the data_manager.get_collaboration_insights already handles filtering
            # Just use the data as returned by the database query
            selected_category = input.category()
            
            collab_items = []
            max_collabs = cross_topic_collaborators[0]['cross_topic_collaborations'] if cross_topic_collaborators else 1
            
            for idx, author_info in enumerate(cross_topic_collaborators[:8]):
                author_name = author_info['author_name']
                cross_topic_count = author_info['cross_topic_collaborations']
                total_papers = author_info['total_papers']
                num_topics = author_info['num_topics']
                primary_topic = author_info.get('primary_topic', 'Unknown')
                
                # Format author name
                formatted_name = author_name
                if ',' in author_name and len(author_name.split(',')) == 2:
                    last, first = author_name.split(',', 1)
                    formatted_name = f"{first.strip()} {last.strip()}"
                
                percentage = (cross_topic_count / max_collabs) * 100 if max_collabs > 0 else 0
                
                # Get primary topic name for display
                primary_topic_name = f"Topic {primary_topic}"
                try:
                    topic_details = data_manager.get_topic_details(primary_topic)
                    if topic_details and topic_details.get('info'):
                        topic_label = topic_details['info'].get('descriptive_label', f"Topic {primary_topic}")
                        if selected_category == "All Math Categories":
                            primary_topic_name = f"Topic {primary_topic}: {topic_label[:50]}{'...' if len(topic_label) > 50 else ''}"
                        else:
                            primary_topic_name = f"{topic_label[:50]}{'...' if len(topic_label) > 50 else ''}"
                except:
                    pass
                
                collab_items.append(
                    ui.div(
                        ui.div(
                            f"{idx+1}. {formatted_name}",
                            style="display: inline-block; width: 60%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; vertical-align: middle; font-weight: bold;"
                        ),
                        ui.div(
                            f"{cross_topic_count} cross-topic collaborations",
                            style="display: inline-block; width: 35%; text-align: right; vertical-align: middle; font-size: 0.9em;"
                        ),
                        ui.div(
                            f"Primary area: {primary_topic_name}",
                            style="font-size: 0.75em; color: #666; margin-top: 2px;"
                        ),
                        ui.div(
                            f"{total_papers} total papers across {num_topics} topics",
                            style="font-size: 0.8em; color: #666; margin-top: 2px;"
                        ),
                        ui.div( # Progress bar
                            style=f"background-color: #17a2b8; height: 4px; width: {percentage}%; margin-top: 5px;"
                        ),
                        style="margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 8px;"
                    )
                )
            
            # Update the description based on category selection
            if selected_category == "All Math Categories":
                description = "Researchers collaborating across multiple mathematical topics:"
            else:
                category_name = selected_category.split(" - ")[1] if " - " in selected_category else selected_category
                description = f"Top cross-topic collaborators in {category_name}:"
            
            return ui.div(
                ui.p(description, 
                    style="font-style: italic; margin-bottom: 15px;"),
                ui.div(*collab_items, style="margin-top: 5px;"),
                style="padding: 0 10px;"
            )
        except Exception as e:
            return ui.p("Cross-topic collaboration data not available.")

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
    
    # NEW: Collaboration Metrics Section
    @output
    @render.ui
    def collaboration_metrics_section():
        topic_data = selected_topic_data()
    
        if topic_data is None:
            return ui.div()
    
        try:
            collaboration_metrics = topic_data.get('collaboration_metrics')
            enhanced_metrics = topic_data.get('enhanced_metrics')  # NEW: Get enhanced metrics
        
            if not collaboration_metrics:
                return ui.div()
        
            # Get team size distribution
            topic_id = topic_data['info']['topic_id']
            team_size_dist = data_manager.get_topic_team_size_distribution(topic_id)
        
            # NEW: Extract enhanced collaboration data
            advanced_metrics = enhanced_metrics.get('advanced', {}) if enhanced_metrics else {}
            degree_analysis = enhanced_metrics.get('degree_analysis', {}) if enhanced_metrics else {}
        
            # Create collaboration metrics cards with enhanced features
            return ui.div(
                ui.h4("Collaboration Network Analysis", class_="text-center", style="margin-top: 20px; margin-bottom: 15px;"),
            
                # Row 1: Basic collaboration metrics
                ui.row(
                    ui.column(
                        4,
                        ui.div(
                            ui.h5("Collaboration Rate", style="margin-bottom: 5px; color: #007bff;"),
                            ui.h3(f"{collaboration_metrics['collaboration_rate']:.1%}", style="margin: 0; font-weight: bold;"),
                            ui.tags.small("of papers have multiple authors", style="color: #666;"),
                            class_="collab-metric"
                        )
                    ),
                    ui.column(
                        4,
                        ui.div(
                            ui.h5("Repeat Collaborations", style="margin-bottom: 5px; color: #007bff;"),
                            ui.h3(f"{collaboration_metrics.get('repeat_collaboration_rate', 0):.1%}", style="margin: 0; font-weight: bold;"),
                            ui.tags.small("of partnerships are recurring", style="color: #666;"),
                            class_="collab-metric"
                        )
                    ),
                    ui.column(
                        4,
                        ui.div(
                            ui.h5("Research Communities", style="margin-bottom: 5px; color: #007bff;"),
                            ui.h3(f"{collaboration_metrics['num_components']}", style="margin: 0; font-weight: bold;"),
                            ui.tags.small("connected research groups", style="color: #666;"),
                            class_="collab-metric"
                        )
                    )
                ),
            
                # NEW: Row 2: Enhanced network topology metrics
                ui.row(
                    ui.column(
                        3,
                        ui.div(
                            ui.h5("Centralization", style="margin-bottom: 5px; color: #28a745;"),
                            ui.h3(f"{advanced_metrics.get('degree_centralization', 0):.3f}", style="margin: 0; font-weight: bold;"),
                            ui.tags.small("network hierarchy level", style="color: #666;"),
                            class_="collab-metric"
                        )
                    ),
                    ui.column(
                        3,
                        ui.div(
                            ui.h5("Modularity", style="margin-bottom: 5px; color: #28a745;"),
                            ui.h3(f"{advanced_metrics.get('modularity', 0):.3f}", style="margin: 0; font-weight: bold;"),
                            ui.tags.small("community strength", style="color: #666;"),
                            class_="collab-metric"
                        )
                    ),
                    ui.column(
                        3,
                        ui.div(
                            ui.h5("Small-World", style="margin-bottom: 5px; color: #28a745;"),
                            ui.h3("✓" if advanced_metrics.get('is_small_world', False) else "✗", 
                                  style="margin: 0; font-weight: bold; color: #28a745;" if advanced_metrics.get('is_small_world', False) else "margin: 0; font-weight: bold; color: #dc3545;"),
                            ui.tags.small("high clustering + short paths", style="color: #666;"),
                            class_="collab-metric"
                        )
                    ),
                    ui.column(
                        3,
                        ui.div(
                            ui.h5("Power-Law", style="margin-bottom: 5px; color: #28a745;"),
                            ui.h3("✓" if degree_analysis.get('power_law_good_fit', False) else "✗", 
                                  style="margin: 0; font-weight: bold; color: #28a745;" if degree_analysis.get('power_law_good_fit', False) else "margin: 0; font-weight: bold; color: #dc3545;"),
                            ui.tags.small("scale-free network", style="color: #666;"),
                            class_="collab-metric"
                        )
                    )
                ),
            
                # Add team size distribution and enhanced insights
                ui.output_ui("team_size_distribution_plot") if team_size_dist else ui.div(),
                ui.output_ui("enhanced_network_insights"),  # NEW: Add enhanced insights
                style="margin-bottom: 20px;"
            )
        except Exception as e:
            return ui.div()
    
    # NEW: Collaboration Network Visualization Section
    @output
    @render.ui
    def collaboration_network_section():
        topic_data = selected_topic_data()
        
        if topic_data is None:
            return ui.div()
        
        try:
            collaboration_metrics = topic_data.get('collaboration_metrics')
            
            if not collaboration_metrics or collaboration_metrics.get('num_authors', 0) < 3:
                return ui.div()
            
            topic_id = topic_data['info']['topic_id']
            top_authors_list = topic_data.get('top_authors', [])
            
            # Try to load pre-computed network, otherwise generate sample
            network_data = load_collaboration_network(topic_id)
            
            if network_data is None:
                # Generate sample network based on collaboration metrics with real author names
                network_data = generate_sample_network(topic_id, collaboration_metrics, top_authors_list)
            
            if network_data is None:
                return ui.div()
            
            # Create network visualization
            network_fig = create_collaboration_network_plot(network_data)
            
            if network_fig is None:
                return ui.div()
            
            return ui.div(
                ui.row(
                    ui.column(
                        12,
                        ui.card(
                            ui.card_header("Collaboration Network - Largest Connected Component"),
                            ui.div(
                                ui.tags.iframe(
                                    srcDoc=network_fig.to_html(include_plotlyjs='cdn'),
                                    style="width:100%; height:500px; border:none;"
                                ),
                                class_="network-container"
                            ),
                            ui.div(
                                ui.p([
                                    ui.strong("Network Insights: "),
                                    f"This visualization shows the largest connected component of researchers collaborating in this topic. ",
                                    f"Node size represents the number of papers published by each author. ",
                                    f"The network has {collaboration_metrics['num_authors']} total authors with ",
                                    f"{collaboration_metrics['network_density']:.3f} connectivity density."
                                ], style="margin-top: 10px; font-size: 0.9em; color: #666;")
                            )
                        )
                    )
                ),
                style="margin-bottom: 20px;"
            )
        except Exception as e:
            return ui.div()
        
    @output
    @render.ui
    def enhanced_network_insights():
        topic_data = selected_topic_data()
    
        if topic_data is None or 'enhanced_metrics' not in topic_data:
            return ui.div()
    
        enhanced = topic_data['enhanced_metrics']
        advanced = enhanced.get('advanced', {})
    
        # Determine network type based on metrics
        insights = []
    
        # Centralization insight
        centralization = advanced.get('degree_centralization', 0)
        if centralization > 0.15:
            insights.append("🎯 **Hub-dominated**: A few key researchers drive collaborations")
        elif centralization < 0.05:
            insights.append("🤝 **Egalitarian**: Collaborations are evenly distributed")
    
        # Modularity insight
        modularity = advanced.get('modularity', 0)
        if modularity > 0.8:
            insights.append("🏘️ **Highly modular**: Distinct research subcommunities exist")
        elif modularity < 0.4:
            insights.append("🌐 **Well-integrated**: Strong cross-community collaboration")
    
        # Robustness insight
        robustness_ratio = advanced.get('robustness_ratio', 1)
        if robustness_ratio < 0.1:
            insights.append("⚠️ **Fragile**: Network depends heavily on key individuals")
        elif robustness_ratio > 0.5:
            insights.append("💪 **Resilient**: Network can withstand researcher departures")
    
        # Core-periphery insight
        coreness = advanced.get('coreness', 0)
        if coreness > 0.5:
            insights.append("🎪 **Core-periphery**: Clear insider/outsider structure")
    
        if not insights:
            return ui.div()
    
        return ui.div(
            ui.h5("Network Characteristics", style="margin-top: 20px; margin-bottom: 10px;"),
            ui.div(
                *[ui.p(insight, style="margin: 5px 0; padding: 5px 10px; background: #f0f8ff; border-radius: 3px;") 
                  for insight in insights],
                style="font-size: 0.9em;"
            ),
            style="margin-top: 15px;"
        )
    
    @output
    @render.ui
    def research_communities_section():
        """Display detected research communities within the topic."""
        topic_data = selected_topic_data()
    
        if not topic_data:
            return ui.div()
    
        enhanced = topic_data.get('enhanced_metrics', {})
        communities = enhanced.get('communities', {})
    
        if not communities or communities.get('num_communities', 0) < 2:
            return ui.div()
    
        return ui.div(
            ui.h5(f"Research Subcommunities ({communities['num_communities']} detected)", 
                  style="margin-top: 20px;"),
            ui.div(
                ui.p(f"Largest community: {communities.get('largest_community_ratio', 0):.1%} of researchers"),
                ui.p(f"Community size inequality (Gini): {communities.get('community_size_gini', 0):.3f}"),
                ui.output_ui("community_size_distribution")
            ),
            style="margin-top: 15px; padding: 10px; background: #f9f9f9; border-radius: 5px;"
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
print("🎉 Math Research Compass ready to launch with collaboration network features!")
app = App(app_ui, server)

# Heroku deployment configuration
if __name__ == "__main__":
    
    # Heroku provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Important: bind to all interfaces for Heroku
    
    print(f"🚀 Starting Math Research Compass on {host}:{port}")
    print("📊 Database-powered for lightning-fast performance!")
    print("🤝 Now featuring collaboration network analysis!")
    print("🌐 Ready for professional deployment!")
    
    app.run(host=host, port=port)