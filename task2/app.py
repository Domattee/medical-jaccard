import pandas as pd
import streamlit as st
from main import load_dataset, load_stopwords, build_k_shingles, jaccard, min_hash
#st.set_page_config(layout="wide")


@st.cache
def load_data():
    data_df = load_dataset()
    stopword_list = load_stopwords()
    return data_df, stopword_list


# TODO: Add some extra info on the two selected documents, allow selection with similar keywords.

st.title("Medical Jaccard")
st.write("This is a simple app to compute the jaccard similarities between records from "
         "[medical-nlp](https://github.com/socd06/medical-nlp). "
         "The true jaccard similarity and an estimate using MinHash are computed for pairs of records. "
         "The shingles that form the basis of the comparison can also be viewed directly.")

with st.spinner('Loading data...'):
    df, stopwords = load_data()

num_docs = len(df.index)


st.header("Input selection")
st.write("Select the two documents to be compared, the shingle size and whether or not stop words should be filtered.")


doc1 = st.number_input("Select the first document.", min_value=1, max_value=num_docs, value=2, key="doc1_choice")
doc2 = st.number_input("Select the second document.", min_value=1, max_value=num_docs, value=3, key="doc2_choice")

shingle_size = st.slider("Set the size of the shingles.", min_value=1, max_value=15, value=2,
                         help="Shingling is part of the process of building the sets on which the two documents will "
                              "be compared. A shingle size of 3 means that three consecutive words are concatenated "
                              "into a single 'shingle'. For example the sentence 'What a nice app!' has 2 three-word "
                              "shingles: 'What a nice' and 'a nice app'. The jaccard similarity is computed based on "
                              "the sets of shingles for each document.")

filterwords = st.checkbox("Tick this box to remove stop words before shingling.", value=True,
                          help="Stop words are common words that carry little information. This includes articles, "
                               "prepositions, etc., but also low content medical phrases. These words were selected "
                               "by the medical-nlp project.")

if filterwords:
    stopwordlist = stopwords
else:
    stopwordlist = None

# Note that shingles are not sorted, and are not listed in the order of appearance
shingles_a = build_k_shingles(df["transcription"][doc1-1], shingle_size, filterwords=stopwordlist)
shingles_b = build_k_shingles(df["transcription"][doc2-1], shingle_size, filterwords=stopwordlist)

st.header("Results")

jaccard_similarity = jaccard(shingles_a, shingles_b)
minhash_similarity = min_hash(shingles_a, shingles_b)
st.write("True jaccard similarity: {}".format(jaccard_similarity))
st.write("MinHash estimation of the jaccard similarity: {}".format(minhash_similarity))

with st.expander("See shingles for the documents"):
    c1, c2 = st.columns(2)
    with c1:
        st.write(pd.Series(sorted(list(shingles_a)), name="Document {}".format(doc1)))
    with c2:
        st.write(pd.Series(sorted(list(shingles_b)), name="Document {}".format(doc2)))
