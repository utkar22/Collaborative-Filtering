# Collaborative-Filtering
This project aims to evaluate and compare various collaborative filtering techniques for building recommender systems. The evaluations were performed on the popular MovieLens dataset, which contains user ratings for movies.

## Project Overview

The main objective of this project is to explore different collaborative filtering approaches and assess their effectiveness in making personalized movie recommendations. The project components include:

1. User-Based Recommender System
2. Item-Based Recommender System
3. Significant and Variance Weighting
4. Latent Factor Model
5. Nuclear Norm Minimization Algorithm
6. Schatten-p Norm Minimization

## Methodology

For each technique, the following steps were taken:

1. Data preprocessing: The MovieLens dataset was used, and any necessary data preprocessing steps, such as handling missing values or normalizing ratings, were performed.
2. Technique implementation: The specific collaborative filtering technique was implemented using appropriate algorithms or libraries.
3. Evaluation metrics: Metrics such as accuracy, precision, and recall were used to measure the performance of each technique.
4. Comparative analysis: The results from each technique were compared to assess their strengths, weaknesses, and suitability for recommender systems.


## Dataset

The evaluations in this project were performed on the MovieLens dataset. Specifically, the MovieLens 100K dataset was used. The dataset provides user ratings for movies and is commonly used in recommender system research and evaluation.

### Dataset Description

- **Dataset Name:** MovieLens 100K
- **Source:** [MovieLens](https://grouplens.org/datasets/movielens/100k/)
- **Description:** The MovieLens 100K dataset contains 100,000 ratings from 943 users on 1,682 movies. Each rating ranges from 1 to 5, where 1 represents the lowest rating and 5 represents the highest. Along with the ratings, the dataset includes additional information such as movie metadata (e.g., title, genres) and user demographic data.

### Data Preprocessing

Before performing the evaluations, the MovieLens 100K dataset underwent some preprocessing steps, which may include:

- Handling missing values: If there were any missing ratings or other missing data, appropriate techniques were applied to handle them (e.g., imputation or exclusion).
- Data normalization: The ratings were normalized to a common scale or standardized using techniques such as min-max scaling or z-score normalization.

## Code

The algorithms were implemented in Python due to the convenience of the `scikit` library. 
The project code is organized in the `src/` (source) folder, which contains the following Python files:

- `src/user_based.py`: Implementation of the user-based recommender system.
- `src/user_significance.py`: Implementation of the user-based recommender system with significance weighting.
- `src/user_variance.py`: Implementation of the user-based recommender system with variance weighting.
- `src/item_based.py`: Implementation of the item-based recommender system.
- `src/latent_factor_model.py`: Implementation of the latent factor model.
- `src/nuclear_norm_minimization.py`: Implementation of the nuclear norm minimization algorithm.
- `src/schatten_p_norm_minimization.py`: Implementation of the Schatten-p norm minimization algorithm.

Please refer to the individual code files for more details and specific implementations of the collaborative filtering techniques.

## Techniques

This section provides an overview of the different collaborative filtering techniques implemented in this project and discusses their advantages and disadvantages.

### User-Based Recommender System

The user-based recommender system predicts ratings for items based on the similarity between users. It identifies similar users to a target user and recommends items that those similar users have rated highly. 

**Pros:**
- Intuitive and easy to understand.
- Can provide accurate recommendations for users with similar tastes.
- Can handle the "cold-start" problem for new users.

**Cons:**
- Scalability issues when dealing with a large number of users.
- Inefficiency in dynamic or frequently changing datasets.
- Limited coverage for recommending items outside a user's neighborhood.

### User-Based Recommender System with Significance Weighting

The user-based recommender system with significance weighting assigns different weights to ratings based on their significance or importance. It takes into account factors such as the number of ratings a user has given or the average rating deviation.

**Pros:**
- Incorporates the significance or importance of ratings in the recommendation process.
- Can mitigate the impact of less reliable or sparse user data.
- Can improve the accuracy of recommendations by giving more weight to ratings with higher significance.

**Cons:**
- Determining appropriate weightings can be challenging and may require domain knowledge.
- Incorrectly assigning weights may lead to biased or inaccurate recommendations.
- Additional complexity in the implementation and maintenance of the weighting mechanism.

### User-Based Recommender System with Variance Weighting

The user-based recommender system with variance weighting considers the variability or spread of ratings when making recommendations. Ratings with lower variance may be given higher weights to improve recommendation accuracy.

**Pros:**
- Considers the reliability and consistency of user ratings.
- Can mitigate the impact of noisy or inconsistent user data.
- Can provide more accurate recommendations by giving higher weight to ratings with lower variance.

**Cons:**
- Estimating variance may require additional computational resources.
- Identifying appropriate weightings based on variance can be subjective.
- Inaccurate estimation of variance may lead to suboptimal recommendations.

### Item-Based Recommender System

The item-based recommender system recommends items to users based on the similarity between items. It identifies items that are similar to the ones a user has shown interest in and recommends those similar items.

**Pros:**
- Can handle a large number of users more efficiently than user-based approaches.
- More robust to changes in user behavior compared to user-based methods.
- Can provide recommendations for niche or less popular items.

**Cons:**
- The computation of item similarity can be expensive for large item sets.
- Cold-start problem for new items without sufficient ratings.
- Limited coverage for recommending items outside a user's item history.

### Latent Factor Model

The latent factor model represents user-item interactions using a lower-dimensional latent space. It aims to capture underlying factors or dimensions that influence user preferences and item attributes.

**Pros:**
- Can capture complex patterns and interactions in user-item data.
- Handles the sparsity problem effectively.
- Allows for personalized recommendations based on individual user preferences.

**Cons:**
- Can be computationally intensive, especially for large datasets.
- Model selection and tuning may require expertise.
- Cold-start problem for new users or items without sufficient data.

### Nuclear Norm Minimization Algorithm

The nuclear norm minimization algorithm is used for matrix completion problems. It aims to fill in missing entries in a partially observed matrix by minimizing the sum of the singular values (nuclear norm) of the matrix.

**Pros:**
- Guarantees a low-rank solution, capturing the underlying structure in the matrix.
- Effective for matrix completion tasks with missing entries.
- Provides robustness against noise and outliers.

**Cons:**
- Computationally expensive for large matrices.
- The quality of completion depends on the matrix's underlying low-rank structure.
- Sensitivity to noisy or inaccurate data.

### Schatten-p Norm Minimization Algorithm

The Schatten-p norm minimization algorithm generalizes the nuclear norm minimization by considering the sum of the pth powers of singular values. It offers more flexibility in capturing different structures in the matrix.

**Pros:**
- Allows for different types of low-rank structures to be captured.
- Can adapt to specific requirements or constraints of the problem.
- Provides a range of solutions from low-rank to sparse structures.

**Cons:**
- Choice of p may depend on the specific problem and requires careful consideration.
- Computationally demanding for large-scale problems.
- Sensitivity to the selection of the regularization parameter.

## License

This project is licensed under the [MIT License](LICENSE).

