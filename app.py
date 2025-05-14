from shiny import App, ui, render, reactive
import pandas as pd
import plotly.express as px
import ast  # For safely evaluating string representations of lists

# Load the data
df = pd.read_csv('results/topics/common_topics.csv')

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

# Create the Shiny app
app_ui = ui.page_fluid(
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
        ui.h1("Math Research Compass", class_="text-center"),
        ui.h3("Overview", class_="text-center"),
    ),
    ui.div(
        ui.h4("Created by Brian Hepler", class_="text-center"),
        style="margin-top: 5px; margin-bottom: 20px;"
    ),
    
    # Project description
    ui.div(
        ui.p("""
            Math Research Compass analyzes arXiv preprints to identify trending research topics across mathematical subfields. 
            This interactive dashboard visualizes topic modeling results from thousands of recent mathematics papers, 
            helping researchers and students discover emerging areas and popular research directions. 
            The analysis uses advanced natural language processing to cluster semantically related papers and identify coherent research themes.
        """),
        class_="project-description"
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
)

def server(input, output, session):
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

# Create and run the app
app = App(app_ui, server)