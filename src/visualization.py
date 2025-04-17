# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Optional

# Configuration visuelle de base (peut être surchargée)
sns.set(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = [14, 7] # Taille par défaut

def plot_time_series_plotly(df: pd.DataFrame,
                           time_col: str,
                           value_col: str,
                           color_col: Optional[str] = None,
                           title: str = "Time Series Plot",
                           xaxis_title: str = "Time",
                           yaxis_title: str = "Value",
                           hover_name: Optional[str] = None):
    """Crée un graphique de série temporelle interactif avec Plotly."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting {value_col} vs {time_col}, colored by {color_col}")
    try:
        fig = px.line(
            df,
            x=time_col,
            y=value_col,
            color=color_col,
            title=title,
            labels={time_col: xaxis_title, value_col: yaxis_title, color_col: color_col},
            hover_name=hover_name
        )
        fig.update_layout(hovermode="x unified")
        fig.show()
    except Exception as e:
        logging.error(f"Error creating Plotly time series plot: {e}")


def plot_hourly_pattern(df: pd.DataFrame,
                        hour_col: str,
                        value_col: str,
                        agg_func: str = 'mean',
                        title: str = "Average Pattern by Hour of Day"):
    """Visualise le pattern horaire moyen."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting hourly pattern for {value_col} aggregated by {agg_func}")
    try:
        hourly_agg = df.groupby(hour_col)[value_col].agg(agg_func).reset_index()
        plt.figure(figsize=(14, 7))
        sns.barplot(x=hour_col, y=value_col, data=hourly_agg, palette='viridis')
        plt.title(title, fontsize=16)
        plt.xlabel('Hour of Day (0-23)', fontsize=14)
        plt.ylabel(f'{agg_func.capitalize()} {value_col}', fontsize=14)
        plt.xticks(range(0, 24, 2)) # Ticks toutes les 2 heures
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting hourly pattern: {e}")


def plot_daily_heatmap(df: pd.DataFrame,
                       day_col: str,
                       hour_col: str,
                       value_col: str,
                       agg_func: str = 'mean',
                       day_names: List[str] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], # Ou complets
                       title: str = "Average Pattern by Day and Hour"):
    """Crée une heatmap des valeurs moyennes par jour de la semaine et heure."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting daily heatmap for {value_col} aggregated by {agg_func}")
    try:
        # Assurer que day_col correspond aux indices de day_names (0-6)
        day_hour_agg = df.groupby([day_col, hour_col])[value_col].agg(agg_func).reset_index()

        # Ajuster day_col si nécessaire (ex: si BQ donne 1-7 et day_names est 0-6)
        if day_hour_agg[day_col].min() == 1 and day_hour_agg[day_col].max() == 7: # Assume BQ format 1(Sun)-7(Sat)
            logging.info(f"Detected day column '{day_col}' likely in BQ format (1-7). Mapping to Python format (0-6)...")
            # Convertir en Mon=0..Sun=6 pour correspondre à l'ordre python/day_names
            day_map = {2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 1:6} # BQ day -> Python day index (Mon=0..Sun=6)
            day_hour_agg[day_col] = day_hour_agg[day_col].map(day_map)
        elif day_hour_agg[day_col].max() > 6:
             logging.warning(f"Max value in '{day_col}' ({day_hour_agg[day_col].max()}) > 6. Heatmap labels might be misaligned if not 0-6.")
        elif day_hour_agg[day_col].min() < 0:
             logging.warning(f"Min value in '{day_col}' ({day_hour_agg[day_col].min()}) < 0. Heatmap labels might be misaligned if not 0-6.")

        # Vérifier si le mapping a fonctionné ou si les données étaient déjà 0-6
        if day_hour_agg[day_col].min() < 0 or day_hour_agg[day_col].max() > 6:
            logging.error(f"Day column '{day_col}' values are outside the expected 0-6 range after potential mapping. Cannot create heatmap correctly.")
            return

        day_hour_pivot = day_hour_agg.pivot(index=hour_col, columns=day_col, values=value_col)

        # Trier les colonnes (jours) selon l'ordre 0-6
        day_hour_pivot = day_hour_pivot.reindex(columns=range(len(day_names)), fill_value=pd.NA)

        # Appliquer les noms des jours
        if len(day_names) == day_hour_pivot.shape[1]:
             day_hour_pivot.columns = day_names
        else:
             logging.warning(f"Length of day_names ({len(day_names)}) does not match number of day columns ({day_hour_pivot.shape[1]}) in pivot table.")


        plt.figure(figsize=(15, 10))
        sns.heatmap(
            day_hour_pivot,
            cmap='viridis',
            annot=True,
            fmt='.1f',
            linewidths=0.5,
            annot_kws={"size": 8}, # Ajuster la taille des annotations
            na_color='lightgrey' # Couleur pour les valeurs manquantes
        )
        plt.title(title, fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Hour of Day', fontsize=14)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting daily heatmap: {e}")


def plot_top_zones(df: pd.DataFrame,
                   zone_col: str,
                   value_col: str,
                   agg_func: str = 'mean', # Ou 'sum'
                   top_n: int = 15,
                   title: str = "Top Zones by Average Value"):
    """Visualise les N zones principales par valeur agrégée."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting top {top_n} zones for {value_col} aggregated by {agg_func}")
    try:
        zone_agg = df.groupby(zone_col)[value_col].agg(agg_func).reset_index()
        zone_agg = zone_agg.sort_values(value_col, ascending=False).head(top_n)

        plt.figure(figsize=(16, 8))
        sns.barplot(
            x=zone_col,
            y=value_col,
            data=zone_agg,
            palette='viridis',
            # order=zone_agg[zone_col].astype(str) # Convertir en str pour éviter tri numérique si zone_col est numérique
            order=zone_agg[zone_col].tolist() # Utiliser tolist() pour garder l'ordre trié
        )
        plt.title(f'Top {top_n} Zones by {agg_func.capitalize()} {value_col}', fontsize=16)
        plt.xlabel('Zone', fontsize=14)
        plt.ylabel(f'{agg_func.capitalize()} {value_col}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting top zones: {e}")


def plot_prediction_vs_actual(df):
    """
    Plot predictions vs actuals for a sample series.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sample = df[df['pickup_community_area'] == df['pickup_community_area'].unique()[0]]
    ax.plot(sample['timestamp'], sample['target_actual'], label='Actual')
    ax.plot(sample['timestamp'], sample['prediction'], label='Prediction')
    ax.legend()
    ax.set_title('Prediction vs Actual')
    return fig