"""
Name: movie_recomendations.py
Date: 4/12/2020
Author: Justin de Sousa and Rodolfo Lopez
Description: This program implements the movie reccomendation system called collobrative filtering. We start with a large 
set of users and a large set of movies and each user has only rated a subset of these movies. This subset of ratings is thus
used to train the system. To measure effectiveness, the system is divided into two categories: one for the training set to
train the system and one for testing set to test the system. 
"""

import csv

from scipy.stats import pearsonr


class BadInputError(Exception):
    """Represents a user-defined exception for invalid movie or user ids."""

    pass


class Movie_Recommendations:
    """Represents a movie recommendation system using collobrative filtering."""

    def __init__(self, movie_filename, training_ratings_filename):
        """
        Constructor.
        Initializes the Movie_Recommendations object from
        the files containing movie names and training ratings.
        The following instance variables should be initialized:
        self.movie_dict - A dictionary that maps a movie id to
               a movie objects (objects the class Movie)
        self.user_dict - A dictionary that maps user id's to a
               a dictionary that maps a movie id to the rating
               that the user gave to the movie.
        """
        self.movie_dict = {}
        self.user_dict = {}

        # Reads in data from movies.csv, adds the movie ids into movie_dict which maps to its movie object.
        movie_file = open(movie_filename, "r", encoding="utf-8")
        csv_reader = csv.reader(movie_file, delimiter=",", quotechar='"')
        for line in csv_reader:
            if line[0] != "movieId":
                self.movie_dict[int(line[0])] = Movie(int(line[0]), line[1])
        movie_file.close()

        # Reads in data from training_ratings.csv, adds user ids to movie object's users list,
        # Adds the user ids into the nested dictionary user_dict, which has a value that maps
        # the movies the user watched to the ratings they gave to that particular movie.
        rating_placeholder = {}
        training_file = open(training_ratings_filename, "r", encoding="utf-8")
        csv_reader = csv.reader(training_file, delimiter=",")
        for line in csv_reader:
            if line[0] != "userId":
                self.movie_dict[int(line[1])].users.append(int(line[0]))
                if int(line[0]) in self.user_dict:

                    rating_placeholder = self.user_dict[int(line[0])]
                    rating_placeholder[int(line[1])] = float(line[2])
                    self.user_dict[int(line[0])] = rating_placeholder
                else:
                    self.user_dict[int(line[0])] = {int(line[1]): float(line[2])}
        training_file.close()

    def predict_rating(self, user_id, movie_id):
        """
        Returns the predicted rating that user_id will give to the
        movie whose id is movie_id.
        If user_id has already rated movie_id, return
        that rating.
        If either user_id or movie_id is not in the database,
        then BadInputError is raised.
        """
        # Checks to see if the user or movie ids are valid. If invalid, it raises a bad input exception.
        if user_id not in self.user_dict or movie_id not in self.movie_dict:
            raise BadInputError
        else:
            # If the user already rated the movie, return that rating which is located in the user nested dictionary.
            if user_id in self.movie_dict[movie_id].users:
                rating = self.user_dict[user_id]
                rating = rating[movie_id]
                return rating
            # If the user has not rated the movie, perform the prediction rating calculation.
            else:
                # Creates a list of all of the movies that the user rated
                user_rated_movies = list(self.user_dict[user_id])
                sim_x_rate_total = 0
                sim_total = 0
                # Reads in the rating for each movie from the user_dict and calculates the similarity using the movie dict to access
                # the movie object's get similarity method.
                for movie in user_rated_movies:
                    rating = self.user_dict[user_id]
                    rating = rating[movie]
                    similarity = self.movie_dict[movie].get_similarity(
                        movie_id, self.movie_dict, self.user_dict
                    )
                    # Now with the user rating and similarity calculation, multiply them. Save this multiplying total and the similarity total.
                    sim_x_rate_total += float(similarity) * float(rating)
                    sim_total += float(similarity)

                # If the sum of the similarities is zero, the default prediction rating is 2.5
                if sim_total == 0.0 or sim_total == 0:
                    predicted_rating = 2.5
                    return predicted_rating
                # Divide the rating and similarity multiplication total by the similarity total to get the predicted rated and return this value.
                else:
                    predicted_rating = sim_x_rate_total / sim_total
                    return predicted_rating

    def predict_ratings(self, test_ratings_filename):
        """
        Returns a list of tuples, one tuple for each rating in the
        test ratings file.
        The tuple should contain
        (user id, movie title, predicted rating, actual rating)
        """
        # Initializes tuple list
        tuple_list = []
        test_file = open(test_ratings_filename, "r", encoding="utf-8")
        csv_reader = csv.reader(test_file, delimiter=",")
        # Reads in the user id, movie id, and ratings from test_ratings.csv and appends data to the tuple list. The first element of the tuple is
        # the user_id, the second element is the movie title which is read from the movie object's title found in the movie_dict. The third element
        # is the prediction value that is calculated with the movie object's predict_rating method. The fourth element is the actual rating which is
        # found in the test_ratings.csv.
        for line in csv_reader:
            if line[0] != "userId":
                tuple_list.append(
                    (
                        int(line[0]),
                        self.movie_dict[int(line[1])].title,
                        self.predict_rating(int(line[0]), int(line[1])),
                        float(line[2]),
                    )
                )
        print(type(tuple_list[0]))
        return tuple_list

    def correlation(self, predicted_ratings, actual_ratings):
        """
        Returns the correlation between the values in the list predicted_ratings
        and the list actual_ratings.  The lengths of predicted_ratings and
        actual_ratings must be the same.
        """
        return pearsonr(predicted_ratings, actual_ratings)[0]


class Movie:
    """
    Represents a movie from the movie database.
    """

    def __init__(self, id, title):
        """
        Constructor.
        Initializes the following instances variables.  You
        must use exactly the same names for your instance
        variables.  (For testing purposes.)
        id: the id of the movie
        title: the title of the movie
        users: list of the id's of the users who have
            rated this movie.  Initially, this is
            an empty list, but will be filled in
            as the training ratings file is read.
        similarities: a dictionary where the key is the
            id of another movie, and the value is the similarity
            between the "self" movie and the movie with that id.
            This dictionary is initially empty.  It is filled
            in "on demand", as the file containing test ratings
            is read, and ratings predictions are made.
        """
        self.id = id
        self.title = str(title)
        self.users = []
        self.similarities = {}

    def __str__(self):
        """
        Returns string representation of the movie object.
        Handy for debugging.
        """
        return f"{self.id} {self.title}"

    def __repr__(self):
        """
        Returns string representation of the movie object.
        """
        return f"Movie ID- {self.id} Movie Title- {self.title} Users- {self.users} Similarities- {self.similarities}"

    def get_similarity(self, other_movie_id, movie_dict, user_dict):
        """
        Returns the similarity between the movie that
        called the method (self), and another movie whose
        id is other_movie_id.  (Uses movie_dict and user_dict)
        If the similarity has already been computed, return it.
        If not, compute the similarity (using the compute_similarity
        method), and store it in both
        the "self" movie object, and the other_movie_id movie object.
        Then return that computed similarity.
        If other_movie_id is not valid, raise BadInputError exception.
        """
        try:
            # Checks to see if the similarity has already been computed. If it has been, it returns the similarity computation that is
            # stored in the movie object's similarity dictionary which maps other movies the user has seen to the similarity that was
            # computed with respect to the movie object.
            if other_movie_id in movie_dict:
                if other_movie_id in self.similarities:
                    return self.similarities[other_movie_id]
                # If the similarity has not been calculated, it calls compute similiality, stores this calculation in the movie object's
                # similarity dictionary as well as stores this calculation in the other movie's object similarity dictionary for perhaps
                # later, and lastly returns the similarity calculation.
                else:
                    similarity = self.compute_similarity(
                        other_movie_id, movie_dict, user_dict
                    )
                    self.similarities[other_movie_id] = similarity
                    movie_dict[other_movie_id].similarities[self.id] = similarity
                    return similarity
            # If the other movie id is not valid, raises bad input exception.
            else:
                raise BadInputError
        except BadInputError:
            print("The other_movie_id is not valid.")

    def compute_similarity(self, other_movie_id, movie_dict, user_dict):
        """
        Computes and returns the similarity between the movie that
        called the method (self), and another movie whose
        id is other_movie_id.  (Uses movie_dict and user_dict)

        """
        # Initializes a list for the user ratings for the movie that called the method (self).
        init_movie_ratings = []
        # Initializes a list for the user ratings for the other movie.
        other_movie_ratings = []
        # Initialzes a list to hold all of the absolute difference calcuations for the user ratings.
        absolute_difference = []

        # Creates list of users who have seen the __init__ movie.
        init_movie_users = movie_dict[self.id].users
        # Creates list of users who have seen the other movie.
        other_movie_users = movie_dict[other_movie_id].users

        # Initializes a list for users who viewed both the __init__ movie and the other movie and appends to the list by iterating over
        # init_movie_users and checking if each user is also in the other_movie_users list.
        users_who_viewed_both = []
        for user in init_movie_users:
            if user in other_movie_users:
                users_who_viewed_both.append(user)

        # Returns similarity value of zero if no user viewed both movies.
        if len(users_who_viewed_both) == 0:
            similarity = 0.0
            return similarity
        else:
            # For all users who have seen both movies, it appends the user rating for the __init__ movie into the init_movie_ratings list.
            for user in users_who_viewed_both:
                init_rating = user_dict[user]
                init_rating = init_rating[self.id]
                init_movie_ratings.append(init_rating)
                # For all users who have seen both movies, it appends the user rating for the other movie into the other_movie_ratings list.
                other_rating = user_dict[user]
                other_rating = other_rating[other_movie_id]
                other_movie_ratings.append(other_rating)

            # Calculates the absolute difference between the ratings in the init_movie_ratings list from the ratings in the
            # other_movie_ratings list and appends each calculation into the absolute difference list.
            for i, init_rating in enumerate(init_movie_ratings):
                absolute_difference.append(abs(init_rating - other_movie_ratings[i]))

            # Sums the absolute diffenece calculations together and divides by the total number of calcualtions to find the average difference.
            absolute_difference_total = sum(absolute_difference)
            average_difference = absolute_difference_total / len(init_movie_ratings)

            # Linear Equation to calculate the weighted similarity value using the average diffence.
            similarity = 1 - average_difference / 4.5

            return float(similarity)


if __name__ == "__main__":
    # Create movie recommendations object.
    movie_recs = Movie_Recommendations("movies.csv", "training_ratings.csv")

    # Predict ratings for user/movie combinations
    rating_predictions = movie_recs.predict_ratings("test_ratings.csv")
    print("Rating predictions: ")
    for prediction in rating_predictions:
        print(prediction)
    predicted = [rating[2] for rating in rating_predictions]
    actual = [rating[3] for rating in rating_predictions]
    correlation = movie_recs.correlation(predicted, actual)
    print(f"Correlation: {correlation}")
