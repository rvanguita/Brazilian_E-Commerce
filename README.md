# Sentiment Identification - NLP

![](assets/img/wallpaper.png)

## Introduction

The field of Natural Language Processing (NLP) is currently one of the most promising areas within the machine learning landscape. With the widespread adoption of large language models (LLMs) and the advancement of international discussions on the regulation of social media platforms, the task of sentiment analysis has gained prominence and become highly relevant. Having a model capable of classifying the sentiment of a given post can, for instance, prevent harmful content from being disseminated, thereby reducing reputational risks and potential damage to the image of individuals or organizations.

On the other hand, this technique can also be used to detect potential issues or dissatisfaction, serving as a way to assess the “emotional temperature” of customers regarding a brand or service. Within this context, this study will use review data from a Brazilian e-commerce platform, with the aim of performing text preprocessing and training machine learning models capable of identifying two possible sentiments expressed in the analyzed posts.

## Objective

The objective of this study is to train a machine learning model to solve a Natural Language Processing (NLP) problem, using data from a Brazilian e-commerce platform. Given the nature of the data (related to shopping experiences) the classification was defined into two main sentiments: positive and negative. This is justified by the fact that, in general, comments tend to express either praise or complaints, especially regarding aspects such as product delivery.

At the end of this study, a function was developed to identify the sentiment of a given sentence, returning not only the predictive classification, but also the confidence percentage associated with the prediction.

### Repository Structure

The `main.ipynb` notebook contains the core code responsible for executing the analyses conducted on the dataset. All visual assets used in this document are located in the `assets/img/` directory.

The `src/` directory houses the Python scripts developed throughout the analytical process. These scripts implement functions and classes designed to streamline future analyses by promoting code reusability, graphical standardization, and workflow organization. Each file follows a modular structure to minimize code redundancy and maintain the visual consistency adopted across all project visualizations.

The `data/` directory contains the `.csv` file with the dataset used in this project. Finally, the `requirements.txt` file provides a complete list of libraries and dependencies, enabling easy replication of the development environment by other users.



## [Data set](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The dataset used in this project was obtained from the Kaggle platform, where additional information is available, including detailed descriptions of each column and the corresponding data types.

We deliberately chose not to include these descriptions directly in this `README.md` to keep the content concise and avoid overloading the document. For further details regarding the dataset structure, we recommend referring to the original project page on Kaggle.

## Methodology and Results

The dataset used in this study contains a wide range of information that enables complex analyses of the Brazilian e-commerce market. However, as the primary focus of this work is to analyze customer reviews, all other data attributes were excluded at this stage. For this purpose, only the `olist_order_reviews_dataset.csv` file was utilized.

From the available data in this file, only two columns were selected: `review_comment_message` and `review_score`. The `review_score` column is a numerical variable ranging from 1 to 5. To simplify the sentiment analysis, a new categorical variable was created, where scores from 1 to 3 were labeled as negative sentiment, and scores 4 and 5 as positive sentiment.

This transformation facilitates a more straightforward analysis of comment polarity. The figure below illustrates the distribution of the two sentiment categories, with 62.12% classified as positive and 37.88% as negative. It is worth noting that, culturally, it is less common for customers to leave unsolicited positive feedback. Conversely, when an issue occurs with an order, the likelihood of posting a negative review increases significantly.

![](assets/img/1.png)


Following a preliminary analysis of the content in selected customer reviews, the **text preprocessing pipeline** described below was developed.

This pipeline consists of four main stages:

1. **Data cleaning**: Removal of line breaks, dates, monetary values, and other elements deemed irrelevant for textual analysis.
2. **Stopword removal**: Elimination of common Portuguese stopwords such as “de,” “para,” “com,” and similar terms that do not contribute meaningful semantic value to the context.
3. **RSLP stemming algorithm**: Application of the RSLP (Removedor de Sufixos da Língua Portuguesa) algorithm to reduce words to their root forms, facilitating lexical normalization.
4. **TF-IDF vectorization**: Conversion of the cleaned text into numerical vectors using the TF-IDF (Term Frequency–Inverse Document Frequency) method, which assigns weights to terms based on their local and global frequency.

This preprocessing pipeline standardizes the textual data, enabling its effective use as input for machine learning algorithms. It enhances the models' ability to detect meaningful patterns and extract relevant insights from textual content.


```python
text_pipeline = Pipeline([
    ('RegexCleanerTransformer', RegexCleanerTransformer(regex_cleaner)),
    ('StopwordRemover', StopwordRemover(pt_stopwords)),
    ('StemmerTransformer', StemmerTransformer(rslp_stemmer)),
    ('TextVectorizer', TextVectorizer(tfidf_vectorizer))
])
```

The model selected for this study was the LGBMClassifier, chosen for its strong performance when combined with vector representations generated through the TF-IDF method. The following are presented below: the confusion matrix, the ROC curve, and a table with all evaluation metrics. Model training was performed using cross-validation to ensure robustness and generalizability of results. Among the evaluation metrics, the model achieved a ROC AUC of 94.10% and an F1 Score of 88.63	%, highlighting its effectiveness in the sentiment classification task.


![](assets/img/2.png)



| Accuracy | Precision | Recall | F1 Score | ROC AUC | Matthews Corrcoef | Cohen Kappa | Log Loss |
|----------|-----------|--------|----------|---------|--------------------|-------------|----------|
| 88.62    | 88.62     | 88.62  | 88.63    | 94.10   | 0.75               | 0.75        | 29.18    |




With the model trained and the TF-IDF vectorizer fitted, a custom prediction function was developed. This function takes as input a review sentence, along with the trained model, the fitted TF-IDF transformer, and the text preprocessing function. As output, it generates an image displaying the prediction probability and the predicted sentiment (positive or negative).

Below are some example sentences along with the corresponding model predictions:

- **Sentence**: *"Péssimo produto! Não compro nessa loja, a entrega atrasou e custou muito dinheiro!"*  
  ![](assets/img/3.png)

- **Sentence**: *"Adorei e realmente cumpriu as expectativas. Comprei por um valor barato. Maravilhoso."*  
  ![](assets/img/4.png)

- **Sentence**: *"Não sei se gostei do produto. O custo foi barato, mas veio com defeito. Se der sorte, vale a pena."*  
  ![](assets/img/5.png)

The final sentence presents an **ambiguous or neutral tone**, reflecting uncertainty on the part of the customer. Nevertheless, the model was able to correctly interpret the overall context and classify the message as expressing **negative sentiment**, demonstrating sensitivity to the **linguistic nuances** present in the text.

Next, a word cloud is presented for each sentiment class (positive and negative), displaying the most frequently occurring terms. The word size corresponds to the frequency of each term in the analyzed comments, allowing for a quick visual identification of the most representative words in each sentiment category.

  ![](assets/img/6.png)
