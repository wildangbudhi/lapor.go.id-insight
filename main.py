import pandas as pd
import nltk
import string
import re
from nltk import word_tokenize
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from base64 import b64encode
from io import BytesIO

nltk.download('punkt')

from flask import Flask, send_from_directory, render_template, request

class Preprocessing:
    def __init__(self, text):
        self.text = text.lower()

    def all_preprocess(self):
        self.remove_digits()
        self.remove_punctuations()
        self.text =  word_tokenize(self.text)
        self.replace_slangword()
        self.stemming()
        return self.text

    def remove_digits(self):
        pattern = '[0-9]'
        self.text = re.sub(pattern, '', self.text)

    def remove_punctuations(self):
        pattern = '[{}]'.format(re.escape(string.punctuation))
        self.text = re.sub(pattern, '', self.text)

    def replace_slangword(self):
        self.text = ' '.join([dicts.get(i, i) for i in self.text])
    
    def stemming(self):
        self.text = stemmer.stem(self.text)

print( "Loading Model" )
slangword_df = pd.read_table('./Dataset/slangword.csv',sep=";")
dicts = dict(zip(slangword_df["kata"], slangword_df["kata_ganti"]))

columns = ["id_laporan", "instansi_laporan", "isi_laporan", "judul_laporan", "tgl_laporan"]
df_lapor = pd.read_csv('./Dataset/scraping_dataset.csv')
text = joblib.load('./Pickle/list_text_scraping-preprocessing.pkl')

tfidf_vectorizer = joblib.load('./Pickle/vectorizer_lapor250.pkl')
tfidf_matrix = joblib.load('./Pickle/tfidf-matrix_lapor250.pkl')

title = []
instansi = []
tgl_laporan = []
for x in range(len(df_lapor)):
    title.append(df_lapor.iloc[x, 3])
    instansi.append(df_lapor.iloc[x, 1])
    tgl_laporan.append(df_lapor.iloc[x, 4])

labels = joblib.load('./Pickle/labels_cluster50_feature250.pkl')
word_dataframe = pd.DataFrame(list(zip(title,instansi,labels,tgl_laporan)),columns=['judul_laporan','instansi','cluster', 'tgl_laporan'])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

print( "Model Loaded" )

def search_result( query: str ):
    
    # Input
    text_input = Preprocessing( query )
    text_input = [ text_input.all_preprocess() ]

    # Transform
    tfidf_input = tfidf_vectorizer.transform(text_input)

    # Cosine Similarity
    cos = cosine_similarity(tfidf_matrix, tfidf_input)
    df = pd.DataFrame(cos, columns=['cos'] )
    df = df.sort_values(by=['cos'], ascending=False)

    indexes = df.index.values

    # Get 500 DataFrames
    result = word_dataframe.loc[indexes]

    best_cluster = word_dataframe[df['cos']==df['cos'].max()].cluster.iloc[0]

    top_result = result[result['cluster']==best_cluster]
    top_result = top_result.head(500)

    # Get Instansi
    instansi_tuj = top_result['instansi'].mode().values

    # GroupBy Month
    result_visual = top_result.copy(deep=True)
    result_visual['month'] = result_visual['tgl_laporan'].str.split(" ").str[1]
    result_visual = result_visual.groupby('month')['judul_laporan'].count()

    # Sort Based Month
    cats = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei','Jun', 'Jul', 'Agu','Sep', 'Okt', 'Nov', 'Des']
    result_visual.index = pd.CategoricalIndex(result_visual.index, categories=cats, ordered=True)
    result_visual = result_visual.sort_index()

    plot = result_visual.plot(kind="bar")
    plt = plot.get_figure()

    pic_IObytes = BytesIO()
    plt.savefig( pic_IObytes,  format='png', bbox_inches='tight' )
    pic_IObytes.seek( 0 )
    pic_hash = b64encode( pic_IObytes.read() )

    res = {
        "instansi_tujuan" : instansi_tuj[0],
        "relevant_report" : top_result.to_dict('records'),
        "graphical_info" : pic_hash.decode('utf-8')
    }

    return res

app = Flask( __name__, static_folder="static", template_folder="views" )

@app.route( '/static/<path:path>' )
def static_route( path ):
    return send_from_directory('static', path )

@app.route( "/", methods=[ "GET", "POST" ] )
def index():
    return render_template("index.html")

@app.route( "/search", methods=[ "GET" ] )
def search():

    search_query = request.args.get('query')
    print( search_query )
    res = search_result( search_query )

    return render_template("result.html", **res)

app.run()