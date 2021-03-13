import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
import spacy
import nltk
from gensim.matutils import jensen_shannon
from gensim.models import LdaModel
import pyLDAvis.gensim
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
nltk.download('stopwords') 
nltk.download('corpus') 
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import pairwise_distances
import os, re, operator, warnings
warnings.filterwarnings('ignore')




#-----Start of Set Up-----#
st. set_page_config(layout="wide")
def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")

my_page = st.sidebar.radio('Contents',['Background','Data Information','Methodology','Result','Search Engine','Recommender Engine', 'Recommendations','The Team']) # creates sidebar #

st.markdown("""<style>.css-1aumxhk {background-color: #ebeae3;background-image: none;color: #ebeae3}</style>""", unsafe_allow_html=True) # changes background color of sidebar #

#-----End of Set Up-----#

#-----Start of Background-----#
if my_page == 'Background':
    st.title("Enhancing the User Experience of a Streaming Platform with Search and Recommender Engine using NLP")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    a1,a2,a3,a4 = st.beta_columns(4)
    a1.image('film1.png',use_column_width=True)
    a2.image('film2.png',use_column_width=True)
    a3.image('film3.png',use_column_width=True)
    a4.image('film4.png',use_column_width=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('During the lockdown brought by the COVID-19 pandemic, one of the industries that are heavily hit is the film and TV industry. Cinemas were not allowed to operate; film and TV show shoots were put to a halt. Despite these, the industry still managed to provide the public with a little bit of emotional comfort with their streaming platforms. One of these platforms is the Film Development Council of the Philippines’ “FDCP Channel."', unsafe_allow_html=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.subheader("FDCP and the FDCP Channel")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    b1,b2,b3 = st.beta_columns((1,4,1))
    b2.image('fdcpchannel1.png',use_column_width=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('The Film Development Council of the Philippines (FDCP) is the national film agency under the Office of the President of the Philippines responsible for film policies and programs to ensure the economic, cultural and educational development of the Philippine film industry.  Pursuant to the constitutional guarantee on freedom of expression, FDCP aims to promote and support the development and growth of the local film industry as a medium or the upliftment of aesthetic, cultural and social values, or the better understanding and appreciation of the Filipino identity.', unsafe_allow_html=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('FDCP Channel is an exclusive, FDCP-managed platform that lets audiences enjoy quality local and international content online. It also aims to give audiences access to the country’s most celebrated film festivals from the safe comfort of their homes, wherever they are in the country. It was launched October of last year (2020) in order to host the 4th Pista ng Pelikulang Pilipino.', unsafe_allow_html=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.subheader('Problem Statement')
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    c1,c2,c3 = st.beta_columns((1,4,1))
    c2.image('fdcpchannel2.png',use_column_width=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('Being a new streaming platform, FDCP Channel still needs some features that will enhance the user’s experience. Their web app currently does not have a “search” function; and when a  movie is selected from their catalog, it doesn’t have other films to recommend that may be related to the selected movie. The mentioned missing features are important to be addressed as FDCP’s movie catalog widens. These features can also help with FDCP’s aim of ‘promoting the local film industry’ and ‘appreciation of the Filipino identity’ as they will enable the users to explore films that may be related to their favorite films but are unknown to them.', unsafe_allow_html=True)
#-----End of Background-----#

#-----Start of Data Information-----#
elif my_page == 'Data Information':
    st.title("Data Information")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    b1,b2,b3 = st.beta_columns((1,5,1))
    b2.image('scrapingprocess.png',use_column_width=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('The dataset for this project was obtained from IMDbPY. IMDbPY is a Python package that can be used  to retrieve and manage the data of the IMDb movie database about movies, people, characters and companies. It can retrieve data from both the IMDb’s web server and a local copy of the whole database. Know more about IMDBPy through this link: https://imdbpy.github.io',unsafe_allow_html=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #

#-----End of Data Information-----#

#-----Start of Methodology-----#
elif my_page == 'Methodology':
    st.title("Methodology")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.subheader('Data Cleaning')
    b1,b2,b3 = st.beta_columns((1,5,1))
    b2.image('cleaningpipeline.png',use_column_width=True)
    st.subheader('Natural Language Processing')
    c1,c2,c3 = st.beta_columns((1,5,1))
    c2.image('nlppipeline.png',use_column_width=True)
    st.subheader('Search Engine')
    d1,d2,d3 = st.beta_columns((1,5,1))
    d2.image('searchpipeline.png',use_column_width=True)
    st.subheader('Recommender Engine')
    e1,e2,e3 = st.beta_columns((1,5,1))
    e2.image('recopipeline.png',use_column_width=True)
#-----End of Methodology-----#


#-----Start of Result-----#
elif my_page == 'Result':
    st.title("Result")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.subheader('Topic Modelling')
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('Close the sidebar for better experience.', unsafe_allow_html=True)
    HtmlFile = open("LDA.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=1600, width=1600)
    
    
#-----End of Result-----#


#-----Start of Search Engine-----#
elif my_page == 'Search Engine':
    
    df = pd.read_csv("new_topics.csv",encoding= 'unicode_escape')
    
    st.title("Search Engine")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    def remote_css(url):
        st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)
    local_css("style.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

    #
    search_input = st.text_input("Enter keyword/s", "")
    button_clicked = st.button("Go")
    
    df['Title'] = df['Title'].astype(str)
    df['Keywords'] = df['Keywords'].astype(str) 
    tfidf = TfidfVectorizer()
    tfidf_features = tfidf.fit_transform(df.Title)
    df = df.astype({'Dominant_Topic':int})
    df_topics = df.groupby(['Dominant_Topic', 'Keywords']).size().to_frame().reset_index()
    topics = df_topics[['Dominant_Topic','Keywords']]
    topics_dict = topics.set_index('Dominant_Topic').T.to_dict('list')
    keys_values = topics_dict.items()
    new_dict = {int(key):str(value) for key, value in keys_values}
    labels_map = new_dict

    X = df['Title'].astype(str)
    y = df['Dominant_Topic']
    
    random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify = y)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.25, stratify = y_train)
    tfidf = TfidfVectorizer(ngram_range=(2,2))
    X_train = tfidf.fit_transform(X_train)
    X_cv = tfidf.transform(X_cv)
    X_test = tfidf.transform(X_test)
    tfidf = TfidfVectorizer()
    data = tfidf.fit_transform(df.Title)
    y = df['Dominant_Topic'].values
    clf_final = SGDClassifier(alpha = 1e-7, loss = "log", class_weight="balanced", n_jobs=-1)
    clf_final.fit(data, y)
    
    def process_query(query):
        preprocessed_reviews = []
        sentance = re.sub("\S*\d\S*", "", query).strip()
        sentance = re.sub('[^A-Za-z]+', ' ', sentance)
        sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords.words('english'))
        preprocessed_reviews.append(sentance.strip())
        return preprocessed_reviews

    def tfidf_search(query):
        query = process_query(query)
        query_trans = tfidf.transform(query)
        pairwise_dist = pairwise_distances(data, query_trans)

        indices = np.argsort(pairwise_dist.flatten())[10:20]
        df_indices = list(df.index[indices])
        return df_indices

    def label(query):
        query = process_query(query)
        query = tfidf.transform(query)
        ans = clf_final.predict(query)
        return labels_map[ans[0]]


    def change_query(query):
        tag = label(query)
        return query + " " + tag
    
    def enter_queries(query) : 
        query = change_query(query)
        df_indices = tfidf_search(query)
        st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
        st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
        st.markdown('<div style="color: #ac722a;text-align: left;font-size: large;font-weight: bold;">Top results</div>',unsafe_allow_html=True)
        st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
        for i in (df_indices):
            title = df.Title.iloc[i]
            year = df.Year.iloc[i].astype(str)
            genre = df.Genre.iloc[i]
            plot = df.Plot.iloc[i]
            
            st.subheader('%s (%s)' % (title,year))
            st.markdown('Genre/s: %s' % genre)
            st.markdown('<div style="color: #000000;">---</div>',unsafe_allow_html=True)
            st.markdown('%s' % plot)
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    
    if button_clicked:
        enter_queries(search_input)

    
#-----End of Search Engine-----#

#-----Start of Recommender Engine-----#
elif my_page == 'Recommender Engine':
    
    df_plots = pd.read_csv("dataframe_plots.csv",encoding= 'unicode_escape')
    
    st.title("Recommender Engine")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    def remote_css(url):
        st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

    def display_movie(dataframe):
        df = dataframe
        for i in range(0, len(df)):
            title = df["Title"][i]
            year = df['Year'][i].astype(str)
            genre = df['Genre'][i]
            plot = df['Plot'][i]
            
            st.subheader('%s (%s)' % (title,year))
            st.markdown('Genre/s: %s' % genre)
            st.markdown('<div style="color: #000000;">---</div>',unsafe_allow_html=True)
            st.markdown('%s' % plot)
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    
    def related_movie(dataframe):
        df = dataframe
        for i in range(0, len(df)):
            title = df["Title"][i]
            year = df['Year'][i].astype(str)
            genre = df['Genre'][i]
            plot = df['Plot'][i]
            distance = df['dist'][i]
            
            st.subheader('%s (%s)' % (title,year))
            st.markdown('Distance Value: %.4f' % distance)
            st.markdown('Genre/s: %s' % genre)
            st.markdown('<div style="color: #000000;">---</div>',unsafe_allow_html=True)
            st.markdown('%s' % plot)
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    
    local_css("style.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

    #
    search_input = st.text_input("What's your favorite Filipino movie or TV show?", "")
    button_clicked = st.button("Go")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    
    if button_clicked:
        
        title2 = search_input.lower()
        
        np.seterr(divide='ignore', invalid='ignore')
        df_input = df_plots
        titles = df_input.Title
        title_list = df_input["Title"].str.lower().tolist()
        df_title = pd.DataFrame()
        # txt file storing values of the probability distribution value of its topic which has been assigned to every word in the dataset.
        # in english: probability that a movie is a particular topic
        with open("movie_ldabow.txt", "rb") as fp:
            lda_bow = pickle.load(fp)

        # check if title is in dataframe_plots
        if title2 in title_list:
            # index of title in list to search in lda_bow
            num = title_list.index(title2)
            # get Jensen–Shannon divergence for each title
            # Probabilty distribution value between two plots can be calculated using Jensen-Shannon Divergence.
            for j in range(len(lda_bow)):
                if j != num:
                    dst = jensen_shannon(lda_bow[num], lda_bow[j], 130)
                    df_individual = pd.DataFrame({"title": [titles[j]],
                                                "dist": [dst]})
                    df_title = df_title.append(df_individual, ignore_index=True)

            # sort by dist, the lower the dist, the closer the 2 movies are
            df_title = df_title.merge(df_input[['Title','Year','Genre','Plot']], left_on='title', right_on='Title')[['Title', 'dist', 'Year', 'Genre', 'Plot']].sort_values('dist')
            
            st.markdown('<div style="color: #ac722a;text-align: left;font-size: large;font-weight: bold;">If you liked...</div>',unsafe_allow_html=True)
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
            movie = df_plots[df_plots['Title'].str.lower() == search_input.lower()][['Title','Year','Genre','Plot']]
            movie = movie.reset_index()
            display_movie(movie)
            
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
            st.markdown('<div style="color: #ac722a;text-align: left;font-size: large;font-weight: bold;">You will also probably like...</div>',unsafe_allow_html=True)
            st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
            result = df_title[0:10]
            result = result.reset_index()
            related_movie(result)

        else:
            st.markdown("Not found. It might be that the film you're looking for is not in our database or you spelled it wrong.",unsafe_allow_html=True)
    
    
    
    
#-----End of Recommendation Engine-----#

#-----Start of Recommendation-----#
elif my_page == 'Recommendations':
    st.title("Recommendations")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('1. Add a ‘rate’ feature so that only the content that a user has rated highly will be used for recommending content to him/her.',unsafe_allow_html=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('2. Identify contents with top ratings and their corresponding topics in order to observe which topics are generally associated with highly rated content.',unsafe_allow_html=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('3. Perform hyperparameter tuning on the Text preprocessing and Topic Modelling to improve the quality of the topics generated.',unsafe_allow_html=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('4. Explore other sources of Movie data aside from IMDb.',unsafe_allow_html=True)
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
#-----End of Recommendation-----#

#-----Start of The Team-----#
elif my_page == 'The Team':
    st.title("The Team")
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    g1,g2,g3 = st.beta_columns(3)
    g1.image('gangster.jpg',use_column_width=True)
    g1.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Mikee Jazmines</div>',unsafe_allow_html=True)
    g1.markdown('<div style="text-align: left;font-style: italic;color: gray;">Mentor</div>',unsafe_allow_html=True)
    g1.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    link = '[LinkedIn](https://www.linkedin.com/in/mikee-jazmines-059b48bb/)'
    g1.markdown(link, unsafe_allow_html=True)
    link = '[GitHub](https://github.com/mikeejazmines)'
    g1.markdown(link, unsafe_allow_html=True)
    
    g2.image('drunkluvyou.jpg',use_column_width=True)
    g2.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Eric Magno</div>',unsafe_allow_html=True)
    g2.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g2.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    link = '[LinkedIn](https://www.linkedin.com/in/ericxmagno/)'
    g2.markdown(link, unsafe_allow_html=True)
    link = '[GitHub](https://github.com/ericxmagno)'
    g2.markdown(link, unsafe_allow_html=True)
    
    g3.image('foursisters.jpg',use_column_width=True)
    g3.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Fatima Santos</div>',unsafe_allow_html=True)
    g3.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g3.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    link = '[LinkedIn](https://www.linkedin.com/in/fatima-grace-santos/)'
    g3.markdown(link, unsafe_allow_html=True)
    link = '[GitHub](https://github.com/fatimasantos02)'
    g3.markdown(link, unsafe_allow_html=True)
    
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    
    
    g4,g5,g6 = st.beta_columns(3)
    g4.image('tadhana.jpg',use_column_width=True)
    g4.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Kaye Yao</div>',unsafe_allow_html=True)
    g4.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g4.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    link = '[LinkedIn](https://www.linkedin.com/in/kaye-janelle-yao/)'
    g4.markdown(link, unsafe_allow_html=True)
    link = '[GitHub](https://github.com/kayeyao)'
    g4.markdown(link, unsafe_allow_html=True)
    
    g5.image('batman.jpg',use_column_width=True)
    g5.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">King Calma</div>',unsafe_allow_html=True)
    g5.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g5.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    link = '[LinkedIn](https://www.linkedin.com/in/melchor-king-calma-699a4a10a/)'
    g5.markdown(link, unsafe_allow_html=True)
    link = '[GitHub](https://github.com/CLMKing)'
    g5.markdown(link, unsafe_allow_html=True)
    
    g6.image('temple.jpg',use_column_width=True)
    g6.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Phoemela Ballaran</div>',unsafe_allow_html=True)
    g6.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g6.markdown('<div style="color: #ebeae3;">.</div>',unsafe_allow_html=True) # space #
    link = '[LinkedIn](https://www.linkedin.com/in/phoemela-ballaran/)'
    g6.markdown(link, unsafe_allow_html=True)
    link = '[GitHub](https://github.com/phoemelaballaran)'
    g6.markdown(link, unsafe_allow_html=True)
#-----End of The Team-----#