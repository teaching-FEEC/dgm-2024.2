import numpy as np
import pandas as pd
import plotly.express as px

def plotly_count_by_labels(x, y, class_names=None):
    # Convert y to class labels
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
        print("mayor")
    
    # Create a DataFrame from the arrays
    df_gen = pd.DataFrame({
        'Features': list(x),  # If you need features, otherwise this can be omitted
        'Class': y
    })

    # Count the number of samples per label
    label_counts = df_gen['Class'].value_counts().reset_index()
    label_counts.columns = ['Class', 'Count']

    # Replace class IDs with names if class_names is not None
    if class_names is not None:
        label_counts['Class'] = label_counts['Class'].map(lambda x: class_names[x])

    # Create the bar chart
    fig = px.bar(
        label_counts,
        x='Class',
        y='Count',
        title='Number of Samples by Generated Class',
        labels={'Class': 'Class', 'Count': 'Count'},
        color='Class',  # Color based on class
        text='Count',   # Add the count values on the bars
        template='plotly_white'  # Chart style
    )

    # Display the count values on the bars
    fig.update_traces(texttemplate='%{text}', textposition='outside')

    # Configure the layout
    fig.update_layout(
        xaxis_title='Class',
        yaxis_title='Count',
        showlegend=True  # Show legend to allow selection
    )

    
    return fig




def plotly_count_by_labels_compare(x_real, y_real, x_gen, y_gen, class_names=None):
    if y_real.ndim > 1:
        y_real = np.argmax(y_real, axis=1)
    if y_gen.ndim > 1:
        y_gen = np.argmax(y_gen, axis=1)

    df_real = pd.DataFrame({'Class': y_real})
    df_gen = pd.DataFrame({'Class': y_gen})

    label_counts_real = df_real['Class'].value_counts().reset_index()
    label_counts_real.columns = ['Class', 'Count']
    label_counts_real['Type'] = 'Real'

    label_counts_gen = df_gen['Class'].value_counts().reset_index()
    label_counts_gen.columns = ['Class', 'Count']
    label_counts_gen['Type'] = 'Generated'

    label_counts_combined = pd.concat([label_counts_real, label_counts_gen])

    if class_names is not None:
        label_counts_combined['Class'] = label_counts_combined['Class'].map(lambda x: class_names[x])

    fig = px.bar(
        label_counts_combined,
        x='Class',
        y='Count',
        color='Type',
        title='Number of Samples by Class (Real vs Generated)',
        labels={'Class': 'Class', 'Count': 'Count'},
        text='Count',
        template='plotly_white'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_title='Class', yaxis_title='Count', barmode='group', showlegend=True)
    return fig



