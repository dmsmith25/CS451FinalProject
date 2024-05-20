# D3Boards.com Bias Evaluation

D3Boards.com Bias Evaluation consists of a bias evaluation on the favoritablility of teams in the NESCAC (Specifically Middlebury, Williams, and Amherst).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Data](#data)
- [Implementation](#implementation)
- [Results](#results)

## Prerequisites
These were the versions of the packages we used during this project:  
- Python 3.8.0
- numpy 1.24.3
- pandas 2.0.3
- beautifulsoup4 4.9.3
- gensim 4.3.2
- nltk 3.8.1
- vaderSentiment 3.3.2
- matplotlib 3.7.0  
*All other packages used in this project are from the Python standard library*

## Data

*IMPORTANT NOTE:*  
*Since the data scraping portion of this project has been conducted, the D3Boards.com website requires a user login to browse their website. Because of this, the scraping code we provide in this project will not be entirely reproduceable.*  

- Source: We scraped our data from the D3Boards.com website in the NESCAC reigon in Basketball. Specifically, the link we used for our searches was https://www.d3boards.com/index.php?topic=4491 with additional ".indexNumber" for what messages we were at on the blog.

- Features:
  - User: a string of the username for the account that made the corresponding blog post
  - Date: a string of the date and time the post was made
  - Message: a string of the body of the blog post
 
- Data Examples:
  - Total Number of Posts Scraped: 30,641
  - Number of Posts between 2012-2013 Season Through 2023-2024 Season (excluding 2020-2021 season due to covid): 18,870
  - Number of Posts between 2012-2013 Season Through 2023-2024 Season That Include Middlebury, Amherst, or Williams (sample included in our evaluation): 10,849
 
- Data File:
  - Our total data for this project is stored in nlp_project_data.csv and is included in our code  

## Implementation
- implementation.ipynb walks through the large steps of our bias evaluation while it uses our source.py file to carry out these proceses.
- We start by loading in our data using our source.py file and visualizing it
- Then we clean our data using the .cleanData() method in our source.py file to give us a dictionary of the different seasons posts where they mention Middlebury, Williams, or Amherst
- Next, we train our Word2Vec word embedding models for all of the different seasons
  - We use hyperparameters of vector_size=10, window=5, min_count=1, workers=4 that we optimized through trial and error with tweaking the params
  - We generate a random seed from 1 to 1000 in the seed param for the models
- Finally we conduct our evaluation:
  - We use our models of each year (10 per year with seeding) to make sentiment scores for each team
  - Sentiment scores are calculated using a formula involving cosine similarity as shown in our .getCosineSimScores() method in our source.py file within the BiasEval class
  - We then graph these results in two seperate graphs:
    - One graph shows all of the different sentiment scores for teams along with their team success
    - The next graph shows the average of the models for each years sentiment scores and standardized to the smallest sentiment score that year.
  - We conclude with a table showing the average sentiment scores of the teams and their average team success throughout the years  

## Results
- 

 

