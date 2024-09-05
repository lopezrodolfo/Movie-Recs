# Movie Recommendation System

This program implements a collaborative filtering movie recommendation system. It uses a dataset of user ratings to predict how users might rate movies they haven't seen yet.

## Authors

- Rodolfo Lopez
- Justin de Sousa

## Date

March 12, 2020

## Features

- Predicts ratings for movies a user hasn't rated yet
- Computes similarity between movies based on user ratings
- Evaluates the accuracy of predictions using correlation

## Files

- `movie_recommendations.py`: Main implementation of the recommendation system
- `test.py`: Test script for the full dataset
- `test_dummy.py`: Test script for a small dummy dataset
- `movies.csv`: Dataset containing movie information
- `training_ratings.csv`: Dataset of user ratings for training the system
- `test_ratings.csv`: Dataset of user ratings for testing the system
- `dummy_*.csv`: Small datasets for initial testing

## Usage

1. Ensure you have Python 3.x installed
2. Install required packages:
   ```
   pip install scipy
   ```
3. Run the main program:
   ```
   python movie_recommendations.py
   ```
4. To run tests:
   ```
   python test.py
   python test_dummy.py
   ```

## How it works

1. The system loads movie and rating data from CSV files
2. It computes similarities between movies based on user ratings
3. To predict a rating, it uses a weighted average of the user's ratings for similar movies
4. The system's accuracy is evaluated by comparing predicted ratings to actual ratings in the test set

## Note

This is a basic implementation of collaborative filtering. For production use, consider using more advanced techniques and optimizing for larger datasets.
