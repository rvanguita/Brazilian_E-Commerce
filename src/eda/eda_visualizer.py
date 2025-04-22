import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class EDAVisualizer:
    def __init__(self, dataframe, color='#c3e88d', figsize=(24, 12), title=''):
        """
        Initializes the DataVisualizer with a DataFrame and default plot settings.

        Args:
        - dataframe (pd.DataFrame): DataFrame containing the data.
        - color (str, optional): Default color for the plots.
        - figsize (tuple, optional): Default figure size.
        """
        self.dataframe = dataframe
        self.color = color
        self.figsize = figsize
        self.title = title


    def _hide_all_spines(self, ax, hide=True):
        if hide:
            for spine in ax.spines.values():
                spine.set_visible(False)  
    def _annotate_bar_label(self, ax, bar, label, offset=15, fontsize=10, va='bottom'):
        height = bar.get_height()
        ax.annotate(label,
                    (bar.get_x() + bar.get_width() / 2., height),
                    ha='center', 
                    va=va,
                    xytext=(0, offset), 
                    textcoords='offset points',
                    fontsize=fontsize)
    def _get_subplot_grid_shape(self, total_items):
        rows = (total_items + 2) // 3  
        cols = min(3, total_items)
        return rows, cols
    def _style_plot_axis(self, ax, hide_all_spines=False):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if hide_all_spines:
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        ax.grid(False)
    def _remove_extra_axes(self, axes, used_count):
        for idx in range(used_count, len(axes)):
            plt.delaxes(axes[idx])
    def _apply_default_plot_style(self, ax, title=''):
        ax.set_title(title or self.title, fontweight='bold', fontsize=13, pad=15, loc='center')
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='both', length=0)
        ax.yaxis.set_visible(False)
        self._hide_all_spines(ax)
    
    def get_category_distribution(self, category):
        counts = self.dataframe[category].value_counts().reset_index(name='count')
        counts.rename(columns={'index': category}, inplace=True)
        counts['percentage'] = counts['count'] / len(self.dataframe)* 100
        return counts

    
    def plot_categorical_distribution(self, category_col, palette=None, sort=True, show_pct=True, show_count=True):
        category_stats = self.get_category_distribution(category_col)
        
        if sort:
            category_stats[category_col] = pd.Categorical(
                category_stats[category_col],
                categories=category_stats.sort_values("percentage", ascending=False)[category_col],
            )
            
        plt.figure(figsize=self.figsize)
        
        axis = sns.barplot(
            data=category_stats,
            x=category_col,
            y="percentage",
            hue=category_col,
            palette=palette,
            order=category_stats[category_col],
            legend=False,
        )
        
        self._apply_default_plot_style(axis)
        
        for bar, (_, row) in zip(axis.patches, category_stats.iterrows()):
            if show_pct:
                self._annotate_bar_label(axis, bar, f'{row["percentage"]:.2f}%', offset=20, fontsize=10, va="top")
            if show_count:
                self._annotate_bar_label(axis, bar, f'({int(row["count"]):,})', offset=10, fontsize=9, va="top")
                
        plt.tight_layout()
    

    def plot_donut_chart(self, category_col, palette=None, show_pct=True, show_count=True, inner_radius=0.7, fontsize=10):
        category_stats = self.get_category_distribution(category_col)
        figure, axis = plt.subplots(figsize=self.figsize)
       
        wedges, _ = axis.pie(
            category_stats["percentage"],
            labels=category_stats[category_col],
            colors=palette,
            wedgeprops={"linewidth": 7, "edgecolor": "white"},
        )
        
        circle = plt.Circle((0, 0), inner_radius, fc="white")
        axis.add_artist(circle)
        axis.set_title(self.title, fontweight="bold", fontsize=13, pad=15, loc="center")
        
        for i, wedge in enumerate(wedges):
            
            angle = (wedge.theta1 + wedge.theta2) / 2
            x = np.cos(np.radians(angle)) * 0.5
            y = np.sin(np.radians(angle)) * 0.5
            
            label = "\n".join(
                filter(
                    None,
                    [
                        f"{category_stats['percentage'].iloc[i]:.1f}%" if show_pct else "",
                        f"({int(category_stats['count'].iloc[i]):,})" if show_count else "",
                    ],
                )
            )
            
            axis.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="black")
        axis.axis("equal")
        plt.tight_layout()
        plt.show()


    def plot_horizontal_feature_bars(self, feature_cols, group_col=None, palette=None):
        rows, cols = self._get_subplot_grid_shape(len(feature_cols))
        figure, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]

        for i, feature in enumerate(feature_cols):
            
            axis = axes[i]
            
            grouped = self.dataframe.groupby([feature, group_col]).size().reset_index(name="count")
            
            total = grouped["count"].sum()
            grouped["percentage"] = grouped["count"] / total * 100
            
            width = 0.8 if self.dataframe[feature].nunique() <= 5 else 0.6
            
            sns.barplot(
                data=grouped,
                x="count",
                y=feature,
                hue=group_col,
                palette=palette,
                ax=axis,
                width=width,
                orient="h",
            )
            
            self._apply_default_plot_style(axis, title=feature)
            
            axis.set_ylabel("")
            axis.xaxis.set_visible(False)
            axis.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
            
            for bar in axis.patches:
                
                val = bar.get_width()
                pct = (val / total) * 100
                
                if pct > 0:
                    axis.annotate(
                        f"{pct:.1f}%",
                        (val, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=11,
                        color="black",
                        fontweight="bold",
                    )
                    
        self._remove_extra_axes(axes, len(feature_cols))
        plt.tight_layout()


    def plot_feature_boxplots(self, feature_cols, group_col=None, palette=None):
        rows, cols = self._get_subplot_grid_shape(len(feature_cols))
        figure, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]

        for i, feature in enumerate(feature_cols):
            axis = axes[i]
            
            if group_col:
                sns.boxplot(
                    data=self.dataframe,
                    x=feature,
                    y=group_col,
                    hue=group_col,
                    palette=palette,
                    orient="h",
                    ax=axis,
                )
                axis.set_ylabel("")
                
            else:
                sns.boxplot(
                    data=self.dataframe,
                    x=feature,
                    color=self.color,
                    orient="h",
                    ax=axis,
                )
                axis.yaxis.set_visible(False)
                
            self._apply_default_plot_style(axis, title=feature)
            
        self._remove_extra_axes(axes, len(feature_cols)
                                )
        plt.tight_layout()


    def plot_feature_histograms(self, feature_cols, group_col=None, palette=None, kde=False):
        rows, cols = self._get_subplot_grid_shape(len(feature_cols))
        figure, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]

        for i, feature in enumerate(feature_cols):
            
            axis = axes[i]
            
            sns.histplot(
                data=self.dataframe,
                x=feature,
                hue=group_col,
                palette=palette,
                kde=kde,
                ax=axis,
                stat="proportion",
            )
            self._apply_default_plot_style(axis, title=feature)
            
        self._remove_extra_axes(axes, len(feature_cols))
        
        plt.tight_layout()


    def plot_scatter_plot(self, x_col, y_col, group_col, palette=None, title_fontsize=16, label_fontsize=14):
        plt.figure(figsize=self.figsize)
        
        sns.scatterplot(
            data=self.dataframe,
            x=x_col,
            y=y_col,
            hue=group_col,
            palette=palette,
            alpha=0.6,
        )
        
        plt.title(f"{x_col} vs {y_col}", fontsize=title_fontsize, weight="bold")
        
        plt.xlabel(x_col, fontsize=label_fontsize)
        plt.ylabel(y_col, fontsize=label_fontsize)
        
        axis = plt.gca()
        
        self._style_plot_axis(axis, hide_all_spines=True)
        
        legend = axis.get_legend()
        
        if legend:
            legend.set_title(group_col)
            legend.set_bbox_to_anchor((1.15, 0.8))
