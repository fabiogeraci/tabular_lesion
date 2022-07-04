import pandas as pd
from helpers.dataset import DataSet
import plotly.express as px


def plot_class_balance(a_dataset: DataSet):
    """
    Plot class imbalance
    :param a_dataset:
    """
    target_series = pd.DataFrame(a_dataset.training_df[a_dataset.target_name].value_counts())
    target_series.reset_index(inplace=True)
    target_series = target_series.rename(columns={'index': 'Clinically_Sig'})
    target_series = target_series.rename(columns={a_dataset.target_name: 'Count'})

    fig = px.bar(target_series, x='Clinically_Sig', y='Count', color=('blue', 'red'), text='Count', title='Class Balance',
                 width=800, height=400)
    fig.update_layout(showlegend=False)
    fig.show(block=True)
