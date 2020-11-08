
import pickle
import pandas as pd
from flask import Flask, render_template, request
from models import top_cosine_similarity, return_similar_titles, get_movie_id


df_movies = pd.read_pickle('pickle_objects/df_movies.pickle')

with open ('pickle_objects/movie_matrix.pickle', 'rb') as f:
    movie_matrix = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('form.html')

@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"This URL cannot be accessed directly. Go <a href=/> here</a> to submit the form"
    if request.method == 'POST':
        form_data = request.form
        movie_title = form_data['title'].lower().strip()
        movie_id = form_data['id'].strip()

        # In case the user enters the movie id and not the movie title
        if movie_title=='': 
            try:
                movie_id = int(movie_id)
            except(ValueError):
                error_message = 'Movie ID must be a number'
                return render_template('form.html',form_data = form_data, error_message = error_message)
            
            # Check the movie id entered
            try:
                details = df_movies.iloc[movie_id].values
            except(IndexError): 
                error_message = 'Movie id does not exist.'
                return render_template('form.html',form_data = form_data, error_message = error_message)

        
        else:
            details = get_movie_id(movie_title, data=df_movies) 
            if details==None:
                similar_titles = return_similar_titles(movie_title, df_movies['title'], threshold=0.60)
                similar_titles_df = df_movies.iloc[similar_titles.index]
                similar_titles_df['title'] = similar_titles_df['title'].map(lambda x: x.title())
                similar_titles_df.rename(columns=lambda x: x.title(), inplace=True)
                return render_template('similar_movies.html',dataframe = similar_titles_df.to_html(header="true", table_id="table"), form_data=form_data)


        id_ = details[0]
        top_n_similar = top_cosine_similarity( id_, movie_matrix)
        similar_movies_df = df_movies[df_movies['ID'].isin(top_n_similar)].reset_index().drop('index', axis=1)
        similar_movies_df['title'] = similar_movies_df['title'].map(lambda x: x.title())
        similar_movies_df.rename(columns=lambda x: x.title(), inplace=True)
            

        return render_template('rec_system.html',form_data = form_data, dataframe=similar_movies_df.to_html(header="true", table_id="table"), 
                details = details)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5105)

