import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Dict, Any

def plot_trip_distribution(df: pd.DataFrame, column: str, title: str = None) -> None:
    """Plot distribution of a numerical column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=50)
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def plot_time_series(df: pd.DataFrame, date_column: str, value_column: str, 
                    agg_func: str = 'mean', title: str = None) -> None:
    """Plot time series data."""
    df_agg = df.groupby(pd.Grouper(key=date_column, freq='D'))[value_column].agg(agg_func)
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_agg.index, df_agg.values)
    plt.title(title or f'{agg_func.capitalize()} {value_column} over time')
    plt.xlabel('Date')
    plt.ylabel(f'{value_column} ({agg_func})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str,
                 title: str = None) -> None:
    """Create a heatmap using pivot table."""
    pivot_table = pd.pivot_table(df, values=value_col, 
                               index=y_col, columns=x_col, 
                               aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title(title or f'Heatmap of {value_col}')
    plt.tight_layout()
    plt.show()

def plot_map(df: pd.DataFrame, lat_col: str, lon_col: str, 
            color_col: str = None, size_col: str = None,
            mapbox_style: str = "carto-positron",
            height: int = 600, width: int = 800) -> None:
    """Create an interactive map using plotly."""
    fig = px.scatter_mapbox(df,
                           lat=lat_col,
                           lon=lon_col,
                           color=color_col,
                           size=size_col,
                           mapbox_style=mapbox_style,
                           height=height,
                           width=width)
    fig.show()
