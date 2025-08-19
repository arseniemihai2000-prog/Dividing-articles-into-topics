import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')

# Funcție pentru preprocesare
def preprocess_text(documents):
    news_df = pd.DataFrame({'document':documents})
    
    news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
    
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
    
    # Eliminare stop-words
    stop_words = set(stopwords.words('english'))
    tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    
    detokenized_doc = []
    for i in range(len(news_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)
    
    news_df['clean_doc'] = detokenized_doc
    
    return news_df['clean_doc'].values.tolist()

# Funcție de procesare
def process(documents, steps):
    progress_bar = tqdm(total=steps, desc="Processing")
    
    # Preprocesare text
    documents = preprocess_text(documents)
    progress_bar.update(1)
    
    # Aplicarea TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(documents)
    progress_bar.update(1)
    
    # Aplicarea SVD (LSI)
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    progress_bar.update(1)
    
    progress_bar.close()
    return X, X_reduced, svd, vectorizer

print("Fetching dataset...")
newsgroups = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'comp.graphics', 'sci.space'])
documents = newsgroups.data[:1000]  # Primele 1000 de documente
print("Dataset fetched.")

# Definirea numărului de pași
n_components = 5
steps = 2  # TF-IDF și SVD

# Procesare
X, X_reduced, svd, vectorizer = process(documents, steps)

# Afisarea rezultatelor
print("Matricea originală (TF-IDF):")
X_array = X.toarray()
print(X_array)
print("\nDimensiunea matricei originale (TF-IDF):")
print(X_array.shape)

print("\nMatricea redusa (LSI):")
print(X_reduced)
print("\nDimensiunea matricei reduse ((LSI)):")
print(X_reduced.shape)

# Topic-uri
terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    terms_in_topic = np.argsort(comp)[::-1]
    print(f"\nTopic {i}:")
    for term in terms_in_topic[:10]:
        print(terms[term])

# Plotare rezultatelor LSI
df = pd.DataFrame(X_reduced, columns=[f'Topic {i}' for i in range(n_components)])
df['Category'] = newsgroups.target[:1000]

plt.figure(figsize=(12, 8))
sns.scatterplot(x='Topic 0', y='Topic 1', hue='Category', palette='viridis', data=df)
plt.title('LSI Topic Plot')
plt.xlabel('Topic 0')
plt.ylabel('Topic 1')
plt.show()

# Plotare componentelor SVD
components_df = pd.DataFrame(svd.components_, columns=terms, index=[f'Topic {i}' for i in range(n_components)])
plt.figure(figsize=(16, 12))
sns.heatmap(components_df.T, cmap='viridis')
plt.title('SVD Components Heatmap')
plt.show()
