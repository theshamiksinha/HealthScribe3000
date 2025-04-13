# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from collections import Counter

def plot_perspective_distribution(data, output_path=None):
    """
    Plot the distribution of perspectives in the dataset
    
    Args:
        data: List of data instances with questions and answers
        output_path: Path to save the plot (if None, display the plot)
    """
    # Count perspectives
    perspective_counts = Counter()
    
    for instance in data:
        for answer in instance["answers"]:
            if "perspectives" in answer:
                for perspective in answer["perspectives"]:
                    perspective_counts[perspective] += 1
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Perspective': list(perspective_counts.keys()),
        'Count': list(perspective_counts.values())
    })
    
    # Sort by count descending
    df = df.sort_values('Count', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Perspective', y='Count', data=df)
    plt.title('Distribution of Perspectives in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved perspective distribution plot to {output_path}")
    else:
        plt.show()

def visualize_perspective_spans(question, answer, perspective_spans, output_path=None):
    """
    Visualize highlighted perspective spans in an answer
    
    Args:
        question: Question text
        answer: Answer text
        perspective_spans: Dictionary mapping perspective names to lists of spans
        output_path: Path to save the visualization (if None, return HTML content)
    """
    # Create HTML with highlighted spans
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Perspective Spans Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .question {{ font-weight: bold; margin-bottom: 10px; }}
            .answer {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            .highlight {{ padding: 2px 4px; border-radius: 3px; margin: 0 2px; }}
            .legend {{ margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            .legend-item {{ margin-bottom: 5px; }}
            .perspective-summary {{ margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
    """
    
    # Add perspective-specific colors
    perspective_colors = {
        "INFORMATION": "#8dd3c7",
        "CAUSE": "#ffffb3",
        "SUGGESTION": "#bebada",
        "QUESTION": "#fb8072",
        "EXPERIENCE": "#80b1d3"
    }
    
    for perspective, color in perspective_colors.items():
        html += f"""
            .{perspective.lower()} {{ background-color: {color}; }}
        """
    
    html += """
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Perspective Spans Visualization</h2>
            <div class="question">Q: """ + question + """</div>
            <div class="answer">
    """
    
    # Convert spans to character positions
    char_perspectives = {}
    
    for perspective, spans in perspective_spans.items():
        for span in spans:
            start_char = span["start_char"]
            end_char = span["end_char"]
            
            for i in range(start_char, end_char):
                if i not in char_perspectives:
                    char_perspectives[i] = []
                char_perspectives[i].append(perspective)
    
    # Build highlighted answer text
    highlighted_answer = ""
    current_perspectives = []
    
    for i, char in enumerate(answer):
        # Check if we need to close a span
        if current_perspectives and (i not in char_perspectives or set(char_perspectives[i]) != set(current_perspectives)):
            highlighted_answer += "</span>"
            current_perspectives = []
        
        # Check if we need to open a new span
        if i in char_perspectives and (not current_perspectives or set(char_perspectives[i]) != set(current_perspectives)):
            current_perspectives = char_perspectives[i]
            
            # Create CSS class for the span (using the first perspective if there are multiple)
            if current_perspectives:
                primary_perspective = current_perspectives[0]
                perspective_class = primary_perspective.lower()
                
                highlighted_answer += f'<span class="highlight {perspective_class}" title="{", ".join(current_perspectives)}">'
        
        # Add the character
        highlighted_answer += char
    
    # Close any open span
    if current_perspectives:
        highlighted_answer += "</span>"
    
    # Add highlighted answer to HTML
    html += highlighted_answer
    
    # Add legend
    html += """
            </div>
            <div class="legend">
                <h3>Legend</h3>
    """
    
    for perspective, color in perspective_colors.items():
        html += f"""
                <div class="legend-item">
                    <span class="highlight {perspective.lower()}">{perspective}</span>
                </div>
        """
    
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"Saved visualization to {output_path}")
        return output_path
    else:
        return html

def plot_summary_lengths(data, output_path=None):
    """
    Plot the distribution of summary lengths by perspective
    
    Args:
        data: List of data instances with generated summaries
        output_path: Path to save the plot (if None, display the plot)
    """
    # Collect summary lengths by perspective
    perspective_lengths = {}
    
    for instance in data:
        if "perspective_summaries" in instance:
            for perspective, summary in instance["perspective_summaries"].items():
                if perspective not in perspective_lengths:
                    perspective_lengths[perspective] = []
                
                # Count words
                word_count = len(summary.split())
                perspective_lengths[perspective].append(word_count)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    for perspective, lengths in perspective_lengths.items():
        sns.kdeplot(lengths, label=perspective)
    
    plt.title('Distribution of Summary Lengths by Perspective')
    plt.xlabel('Word Count')
    plt.ylabel('Density')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved summary length distribution plot to {output_path}")
    else:
        plt.show()

def create_perspective_heatmap(data, output_path=None):
    """
    Create a heatmap showing co-occurrence of perspectives
    
    Args:
        data: List of data instances with perspective labels
        output_path: Path to save the plot (if None, display the plot)
    """
    # Get all unique perspectives
    all_perspectives = set()
    for instance in data:
        for answer in instance["answers"]:
            if "perspectives" in answer:
                all_perspectives.update(answer["perspectives"])
    
    all_perspectives = sorted(list(all_perspectives))
    
    # Initialize co-occurrence matrix
    cooccurrence = np.zeros((len(all_perspectives), len(all_perspectives)))
    
    # Count co-occurrences
    for instance in data:
        for answer in instance["answers"]:
            if "perspectives" in answer:
                perspectives = answer["perspectives"]
                
                # Update co-occurrence counts
                for i, p1 in enumerate(all_perspectives):
                    for j, p2 in enumerate(all_perspectives):
                        if p1 in perspectives and p2 in perspectives:
                            cooccurrence[i, j] += 1
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt="d",
        xticklabels=all_perspectives,
        yticklabels=all_perspectives,
        cmap="YlGnBu"
    )
    plt.title('Perspective Co-occurrence Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved perspective co-occurrence heatmap to {output_path}")
    else:
        plt.show()