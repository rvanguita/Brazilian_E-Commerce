import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np




class DataVisualizer:
    def __init__(self, df, color='#c3e88d', figsize=(24, 12), title=''):
        """
        Initializes the DataVisualizer with a DataFrame and default plot settings.

        Args:
        - df (pd.DataFrame): DataFrame containing the data.
        - color (str, optional): Default color for the plots.
        - figsize (tuple, optional): Default figure size.
        """
        self.df = df
        self.color = color
        self.figsize = figsize
        self.title = title



                
    def _hide_spines(self, ax, hide=True):
        """
        Hides all spines of the given axis if 'hide' is True.

        Args:
            ax (matplotlib.axes.Axes): The axis to modify.
            hide (bool): Whether to hide the axis spines.
        """
        if hide:
            for spine in ax.spines.values():
                spine.set_visible(False)  
    def _annotate_bar_chart(self, ax, bar, label, offset=15, fontsize=10, va='bottom'):
        """
        Annotates a bar in a bar chart with the given label.

        Args:
            ax (matplotlib.axes.Axes): The axis to annotate.
            bar (matplotlib.patches.Rectangle): The bar to annotate.
            label (str): Text to display.
            offset (int): Offset in points above the bar.
            fontsize (int): Font size of the annotation.
            vertical_alignment (str): Vertical alignment of the text.
        """
        height = bar.get_height()
        ax.annotate(label,
                    (bar.get_x() + bar.get_width() / 2., height),
                    ha='center', va=va,
                    xytext=(0, offset), textcoords='offset points',
                    fontsize=fontsize)
    
    
    def _calculate_subplot_grid(self, total_items):
        """
        Determines the optimal number of rows and columns for subplots 
        based on the number of items to be plotted.

        Args:
            total_items (int): Total number of items to plot.

        Returns:
            tuple: (rows, columns) for subplot layout.
        """
        num_rows = (total_items + 2) // 3
        num_cols = min(3, total_items)
        return num_rows, num_cols
    def _style_axis(self, ax, hide_all_spines=False):
        """
        Styles the given axis by removing the top and right spines, 
        optionally hiding all spines, and disabling the grid.

        Args:
            ax (matplotlib.axes.Axes): The axis to customize.
            hide_all_spines (bool): Whether to hide left and bottom spines as well.
        """
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if hide_all_spines:
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        ax.grid(False)
    def _remove_unused_axes(self, axes, used_count):
        """
        Removes extra subplot axes that are not needed for plotting.

        Args:
            axes (list): List of subplot axes.
            used_count (int): Number of axes actually used.
        """
        for idx in range(used_count, len(axes)):
            plt.delaxes(axes[idx])
    
    
    def get_category_distribution(self, category):
        """
        Computes count and percentage for a categorical variable in the given DataFrame.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            category (str): Column name to compute stats on.
            
        Returns:
            pd.DataFrame: DataFrame with columns [category, 'count', 'percentage'].
        """
        counts = self.df[category].value_counts().reset_index(name='count')
        counts.rename(columns={'index': category}, inplace=True)
        counts['percentage'] = counts['count'] / len(self.df)* 100
        return counts


    def plot_categorical_distribution(self, hue=None, palette=None, sort_by_percentage=True, 
                                      display_percentage=True, display_count=True):
        """
        Plots a bar chart displaying the percentage distribution of a categorical variable,
        sorted from highest to lowest, with optional percentage and count annotations on each bar.
        """
        
        # Compute frequency and percentage
        stats = self.get_category_distribution(hue)

        # Determine order
        if sort_by_percentage:
            sorted_categories = stats.sort_values(by='percentage', ascending=False)[hue].tolist()
        else:
            sorted_categories = stats[hue].tolist()

        stats[hue] = pd.Categorical(stats[hue], categories=sorted_categories, ordered=True)

        # Plotting
        plt.figure(figsize=self.figsize)
        ax = sns.barplot(
            data=stats,
            x=hue,
            y='percentage',
            hue=hue,
            palette=palette,
            order=sorted_categories
        )

        ax.set_title(self.title, fontweight='bold', fontsize=13, pad=15, loc='center')
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='both', length=0)
        ax.yaxis.set_visible(False)

        self._hide_spines(ax, hide=True)

        # Add annotations
        for bar, (_, row) in zip(ax.patches, stats.iterrows()):
            if display_percentage:
                self._annotate_bar_chart(ax, bar, f'{row["percentage"]:.2f}%', offset=25, fontsize=10, va='top')
            if display_count:
                self._annotate_bar_chart(ax, bar, f'({int(row["count"])})', offset=10, fontsize=9, va='top')

        plt.tight_layout()


    def plot_donut_chart(self, hue, palette=None, show_percentage=True,
                         show_count=True, inner_radius=0.7, label_fontsize=10):
        """
        Plots a donut chart showing the distribution of a categorical variable.

        Args:
            hue (str): Column name representing the categorical variable.
            palette (list or str, optional): Colors to use for each slice.
            show_percentage (bool): Whether to show percentage labels.
            show_count (bool): Whether to show count labels.
            inner_radius (float): Radius of the inner white circle for the donut effect.
            label_fontsize (int): Font size of the labels.
        """
        # Compute category stats
        stats = self.get_category_distribution(hue)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw pie chart
        wedges, _ = ax.pie(
            stats['percentage'],
            labels=stats[hue],
            colors=palette,
            wedgeprops={'linewidth': 7, 'edgecolor': 'white'}
        )

        # Draw center circle for donut effect
        center_circle = plt.Circle((0, 0), inner_radius, fc='white')
        ax.add_artist(center_circle)
        ax.set_title(self.title, fontweight='bold', fontsize=13, pad=15, loc='center')

        # Annotate each wedge
        for i, wedge in enumerate(wedges):
            angle = (wedge.theta1 + wedge.theta2) / 2
            x = np.cos(np.radians(angle)) * 0.5
            y = np.sin(np.radians(angle)) * 0.5

            label_parts = []
            if show_percentage:
                label_parts.append(f"{stats['percentage'].iloc[i]:.1f}%")
            if show_count:
                label_parts.append(f"({int(stats['count'].iloc[i])})")

            label = '\n'.join(label_parts)
            ax.text(x, y, label, ha='center', va='center', fontsize=label_fontsize, color='black')

        ax.axis('equal')
        plt.tight_layout()
        plt.show()



    def plot_barplot(self, features, hue=None, custom_palette=None):
        rows, cols = self._calculate_subplot_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            grouped = self.df.groupby([feature, hue]).size().reset_index(name='count')
            total_count = grouped['count'].sum()
            grouped['percentage'] = (grouped['count'] / total_count) * 100

            num_categories = self.df[feature].nunique()
            width = 0.8 if num_categories <= 5 else 0.6

            sns.barplot(data=grouped, x='count', y=feature, palette=custom_palette, hue=hue, ax=ax, width=width, orient='h')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            self._style_axis(ax)
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            ax.yaxis.set_visible(True)
            ax.xaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)
            
            for p in ax.patches:
                width = p.get_width()
                percentage = (width / total_count) * 100
                if percentage != 0:
                    ax.annotate(f'{percentage:.1f}%', 
                                (width, p.get_y() + p.get_height() / 2),
                                xytext=(5, 0), 
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=11, color='black', fontweight='bold')


        self._remove_unused_axes(axes, len(features))
        plt.tight_layout()


    def plot_boxplot(self, features, hue=None, custom_palette=None):
        rows, cols = self._calculate_subplot_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            if hue:
                sns.boxplot(data=self.df, x=feature, y=hue, hue=hue, orient='h', palette=custom_palette, ax=ax)
                ax.set_ylabel('')
            else:
                sns.boxplot(data=self.df, x=feature, color=self.color, orient='h', ax=ax)
                ax.yaxis.set_visible(False)
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            self._style_axis(ax)

            ax.set_xlabel('')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)

        self._remove_unused_axes(axes, len(features))
        plt.tight_layout()


    def plot_histplot(self, features, hue=None, custom_palette=None, kde=False):
        rows, cols = self._calculate_subplot_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            sns.histplot(data=self.df, x=feature, hue=hue, palette=custom_palette, kde=kde, ax=ax, stat='proportion')

            self._style_axis(ax)

            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            

            # ax.legend(loc='upper right')

            ax.set_xlabel('')
            ax.yaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)
            
        self._remove_unused_axes(axes, len(features))
        plt.tight_layout()



    def plot_custom_scatterplot(self, x, y, hue, custom_palette=None, title_fontsize=16, label_fontsize=14):
        """
        Plots a customized scatter plot with specified x and y axes, hue for color coding, and custom palette.
        """
        plt.figure(figsize=(self.figsize))
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue, palette=custom_palette, alpha=0.6)

        plt.title(f'{x} vs {y}', fontsize=title_fontsize, weight='bold')
        plt.xlabel(x, fontsize=label_fontsize)
        plt.ylabel(y, fontsize=label_fontsize)

        ax = plt.gca()
        self._style_axis(ax, hide_spines=True)

        legend = ax.get_legend()
        if legend:
            legend.set_title(hue)
            legend.set_bbox_to_anchor((1.15, 0.8))



        


            
            
            



class BinaryClassifiersAnalysis:
    def __init__(self):
        self.trained_models = {}

    def fit(self, models_dict, X_train, y_train, random_search=False, scoring='accuracy'):
        from sklearn.model_selection import RandomizedSearchCV

        for name, item in models_dict.items():
            print(f"Training model {name}")
            model = item['model']
            params = item['params']

            if random_search and params:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params,
                    scoring=scoring,
                    cv=5,
                    n_iter=10,
                    verbose=0,
                    n_jobs=-1,
                    random_state=42
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            self.trained_models[name] = best_model

    def evaluate_performance(self, X_train, y_train, X_test, y_test, cv=5):
        rows = []

        for name, model in self.trained_models.items():
            # Avaliação com K-fold cross-validation
            print(f"Avaliando modelo: {name}")
            start = time.time()
            y_pred_train = cross_val_predict(model, X_train, y_train, cv=cv, method='predict')
            y_proba_train = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
            end = time.time()
            
            self.cm_train = confusion_matrix(y_train, y_pred_train)

            rows.append({
                'model': name,
                'approach': f'Treino {cv} K-folds',
                'acc': accuracy_score(y_train, y_pred_train),
                'precision': precision_score(y_train, y_pred_train),
                'recall': recall_score(y_train, y_pred_train),
                'f1': f1_score(y_train, y_pred_train),
                'auc': roc_auc_score(y_train, y_proba_train),
                'total_time': round(end - start, 3)
            })

            # Avaliação com dados de teste
            start = time.time()
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)[:, 1]
            end = time.time()
            
            self.cm_test = confusion_matrix(y_test, y_pred_test)

            rows.append({
                'model': name,
                'approach': 'Teste',
                'acc': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test),
                'recall': recall_score(y_test, y_pred_test),
                'f1': f1_score(y_test, y_pred_test),
                'auc': roc_auc_score(y_test, y_proba_test),
                'total_time': round(end - start, 3)
            })

        df = pd.DataFrame(rows)
        return df
    
    
    def plot_confusion_matrix(self, classes, cmap='Blues'):
        """
        Gera e plota a matriz de confusão para o conjunto de treinamento e de teste.

        Parameters:
        X_train: Dados de entrada para o treinamento.
        y_train: Rótulos de saída para o treinamento.
        X_test: Dados de entrada para o teste.
        y_test: Rótulos de saída para o teste.
        classes: Lista com os nomes das classes (por exemplo, ['Negative', 'Positive']).
        """
        
        # Plotando as matrizes de confusão para todos os modelos treinados
        plt.figure(figsize=(12, 12))

        for i, (name, model) in enumerate(self.trained_models.items(), 1):
            # # Previsões para o conjunto de treinamento
            # y_train_pred = model.predict(X_train)
            # cm_train = confusion_matrix(y_train, y_train_pred)

            # # Previsões para o conjunto de teste
            # y_test_pred = model.predict(X_test)
            # cm_test = confusion_matrix(y_test, y_test_pred)

            disp_train = ConfusionMatrixDisplay(confusion_matrix=self.cm_train, display_labels=classes)
            disp_train.plot(cmap=cmap)
            plt.title(f'{name} - Training Set')

            disp_test = ConfusionMatrixDisplay(confusion_matrix=self.cm_test, display_labels=classes)
            disp_test.plot(cmap=cmap)
            plt.title(f'{name} - Test Set')

        # Exibindo o gráfico
        plt.tight_layout()
        plt.show()