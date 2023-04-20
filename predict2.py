#get data
import pandas as pd

import streamlit as st

@st.cache_data(ttl=600)
def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url)

aita = load_data(st.secrets["public_gsheets_url"])


#build model
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(aita.clean_btext).toarray()
labels = aita.verdict


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
X_train, X_test, y_train, y_test = train_test_split(aita['clean_btext'], aita['verdict'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = linear_model.LogisticRegression(multi_class="multinomial", solver='newton-cg', C=0.1, penalty='l2', random_state=30).fit(X_train_tfidf, y_train)


#streamlit implimentation

st.title('Asshole Calculator')

sent = st.text_input("Type a summary of the situation.", "Type Here...")

result = clf.predict(count_vect.transform([sent]))

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

y_hat_nb = clf.predict(X_test_tfidf)

# get accuracy score
from sklearn import metrics
accuracy_score_nb = metrics.accuracy_score(y_test, y_hat_nb)


if result == [0.]:
    st.text('You ARE the asshole')
elif result ==  [1.]:
    st.text('You are NOT the asshole')
elif result == [2.]:
    st.text('Everyone Sucks')
else:
    st.text('No Assholes Here')

st.text('With', accuracy_score_nb*100, 'percent accuracy')