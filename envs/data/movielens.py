import os
import pandas as pd

# Load Data
root = './ml-100k'
#movies = pd.read_csv(os.path.join(root, 'movies.csv'))
movies = pd.read_csv(os.path.join(root, "u.item"), sep='|', encoding='latin-1', header=None)
movies_col = ['movieId', 'name', 'date', 'unknown', 'link', 'unknown2', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies.columns = movies_col
movies = movies.drop(["name", "date", "unknown", "link", "unknown2"], axis=1)
movies['genres'] = movies.drop("movieId", axis=1).idxmax(1)
movies = movies[['movieId', 'genres']]
#ratings = pd.read_csv(os.path.join(root, 'ratings.csv'))
ratings = pd.read_csv(os.path.join(root, 'u.data'), sep='\t')
ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']

if __name__ == '__main__':

    #small_ratings = ratings.iloc[:500]
    #small_ratings.sort_values(['userId', 'timestamp'], ascending=[True, True])

    ratings = ratings.sort_values(['userId', 'timestamp'], ascending=[True, True])
    ratings = ratings[['userId', 'movieId', 'rating']]
    ratings["movie_mean"] = ratings[["movieId", "rating"]].groupby("movieId").transform("mean")
    movie_ratings = ratings[["movieId", "movie_mean"]].drop_duplicates("movieId")

    movies["simple_genres"] = movies["genres"].apply(lambda string: string.split('|')[0])
    disc_cat = "Drama"
    movies["disc"] = (movies["simple_genres"] == disc_cat).astype(int)
    movies_proc = pd.merge(movies, movie_ratings, how="left", on="movieId")
    movies_proc.to_csv(os.path.join(root, 'movies_proc.csv'))

    ratings_proc = pd.merge(ratings, movies_proc[["movieId", "disc"]], how="left", on="movieId")
    ratings_proc["rating_mean"] = ratings_proc[["userId", "rating"]].groupby("userId").transform("mean")
    ratings_proc["prop_disc"] = ratings_proc[["userId", "disc"]].groupby("userId").transform("mean")
    ratings_proc.to_csv(os.path.join(root, 'ratings_proc.csv'), index_label=False)

    small_rating_procs = ratings_proc.iloc[:500]
    small_rating_procs.to_csv(os.path.join(root, 'small_ratings_proc.csv'), index_label=False)

    end = True
