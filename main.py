import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Function to create a MovieRecommender class
class MovieRecommender:
    def __init__(self, movie_data_path):
        self.movies_data = pd.read_csv(movie_data_path)
        self.movies_data['Genre'] = self.movies_data['Genre'].fillna('')  # Replace missing values with empty string
        self.tfid = TfidfVectorizer()
        self.vectors = self.tfid.fit_transform(self.movies_data['Genre'].values.astype('U'))  # Convert to Unicode
        self.index = pd.Series(data=self.movies_data.index,
                               index=self.movies_data['Title'].str.lower())  # Convert titles to lowercase

    def recommend_similar_movies(self, movie_title, n=3):
        movie_title_lower = movie_title.lower()  # Convert user input to lowercase
        if movie_title_lower in self.index:
            similarity_distance = linear_kernel(self.vectors[self.index[movie_title_lower]], self.vectors)
            similarity_scores = pd.Series(similarity_distance[0])  # Convert to Series
            similarity_scores.index = self.movies_data.index  # Set index to match with movie_info DataFrame
            similarity_scores = similarity_scores.sort_values(ascending=False)
            empty_list = []
            for recommendation_index in range(len(similarity_scores)):  # Adjusted loop range
                rec_index = similarity_scores.index[recommendation_index]
                if rec_index != self.index[movie_title_lower]:
                    result = {'Title': self.movies_data.loc[rec_index, 'Title'],
                              'Genre': self.movies_data.loc[rec_index, 'Genre'],
                              'Rating': self.movies_data.loc[rec_index, 'Rating']
                              }
                    empty_list.append(result)
                    if len(empty_list) == n:
                        break
            return empty_list
        else:
            return None


# Streamlit UI
st.set_page_config(layout="wide")
st.sidebar.title("Movie Recommendation Engine")

movie_recommender = MovieRecommender('Movie_list.csv')  # Provide the path to your movie list CSV file here
selected_movie_title = st.sidebar.selectbox("Select a Movie", movie_recommender.movies_data['Title'].values, placeholder="Select a Movie")
recommendation_count = st.sidebar.number_input("Enter the recommendation count", min_value=1, max_value=10, step=1)

if selected_movie_title and recommendation_count:
    recommendations = movie_recommender.recommend_similar_movies(
        selected_movie_title, n=recommendation_count)
    if recommendations:
        st.subheader(f"Recommendations for the movie '{selected_movie_title.upper()}'")
        st.markdown("---")
        for i, rec in enumerate(recommendations):
            st.write(f"{i+ 1}. {rec['Title']} (Rating: {rec['Rating']})")
            st.write(f"   - Genre: {rec['Genre']}")
            st.markdown("---")
    else:
        st.error("Movie not found in the database.")
