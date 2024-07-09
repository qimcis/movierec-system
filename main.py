import numpy as np
import pandas as pd 
import difflib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Data Collection and Pre-processing
#loading data from csv file to pandas dataframe
movies_data = pd.read_csv('C:/Users/1/Documents/ML Movie Rec/movies.csv')

#printing first 5 rows of the data frame
movies_data.head()

#number of rows and columuns in data frame 

movies_data.shape

#feature selection

selected_features = ['genres','keywords','tagline','cast','director']

#replacing the null (useless) values with null strings (using fillna)

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

#combining the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

#converting the text data to numerical data so you can easily find the cosine similarity value

vectorizer = TfidfVectorizer()

#transforming the data here and storing it in feature_vector

feature_vector = vectorizer.fit_transform(combined_features)

#COSINE SIMILARITY: get similarity scores using cosine similarity

#feeding feature_vector numerical values into cosine_similarity so that it can find which movies are similar
similarity = cosine_similarity(feature_vector)

#getting movie name from user
movie_name = input("Enter your favourite movie name : ")

#creating list with movie names given in the dataset tolist takes all the values and creates a list
list_of_all_titles = movies_data['title'].tolist()

#now find close match for movie given by user using difflib
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

#finding the index of the movie with title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

#getting list of similar movies based on index value. enumerate is used as a loop to find the similarity scores for all the movies in the list

similarity_score = list(enumerate(similarity[index_of_the_movie]))

#sorting the movies based on similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse=True)

#print the name of similar movies based on the index 

print('Movies suggested for you: \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]

    if (i<20):
        print(i, '.', title_from_index)
        i+=1

