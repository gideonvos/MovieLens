##########################################################################################################################
# HarvardX PH125.9x Data Science Capstone Movielens Project
#
# Student: Gideon Vos (gideonvos@icloud.com) www.gideonvos.com (LinkedIn) https://gideonvos.wordpress.com (Blog)
#
##########################################################################################################################

##########################################################################################################################
# The following packages are required. Please set your default repository to CRAN prior to installing.
##########################################################################################################################
if(!require(readr)) install.packages("readr")
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(dslabs)) install.packages("dslabs")
if(!require(data.table)) install.packages("data.table")
if(!require(ggrepel)) install.packages("ggrepel")
if(!require(ggthemes)) install.packages("ggthemes")

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(gridExtra)
library(dslabs)
library(data.table)
library(ggrepel)
library(ggthemes)

##########################################################################################################################
# Create edx set, validation set, and submission file
##########################################################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

# If your read.table fails, remove it and replace with the fread line below from the data.table package after unzipping
#ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")), 
#data.table=TRUE, col.names = c("userId", "movieId", "rating", "timestamp"))
#movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)


# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings

##########################################################################################################################
# Exploratory analysis
##########################################################################################################################

# remove some items no longer required to reduce memory footprint
rm(removed, test_index)

# Quick preview of the edx dataset shows 6 columns.
# timestamp needs to be converted if used, and release year will need
# to be split from the title if useful for prediction
# genres is a single pipe-delimited string containing the various
# genre categories a movie might be categorized under, and this will
# need to be split out if it affects rating outcome
head(edx)

# Check for any missing values
anyNA(edx)

# Quick summary of the dataset
summary(edx)

# We are dealing with ~70000 unique users giving ratings to ~ 10700 different movies
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# 10 different rating scores, lowest is 0.5 and highest is 5
unique(edx$rating)

# Some movies are rated more often than others
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, binwidth=0.2, color="black", show.legend = FALSE, aes(fill = cut(n, 100))) + 
  scale_x_log10() + 
  ggtitle("Movies Rated")

# Review Training rating distribution
edx %>% 
  ggplot(aes(rating)) + 
  geom_histogram(binwidth=0.2, color="darkblue", fill="lightblue") + 
  ggtitle("Rating Distribution (Training")

# Review Validation rating distribution
validation %>% 
  ggplot(aes(rating)) + 
  geom_histogram(binwidth=0.2, color="darkblue", fill="lightblue") +  
  ggtitle("Rating Distribution (Validation")

# Distributions are similar

# Extract release year from title into a separate field
edx <- edx %>% mutate(releaseyear = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$"))

# Number of movies per year/decade
movies_per_year <- edx %>%
  select(movieId, releaseyear) %>% # select columns we need
  group_by(releaseyear) %>% # group by year
  summarise(count = n())  %>% # count movies per year
  arrange(releaseyear)

# Let's review that
movies_per_year %>%
  ggplot(aes(x = releaseyear, y = count)) +
  geom_line(color="blue")
# We can see an exponential growth of the movie business and a sudden drop in 2010 
# The latter is caused by the fact that the data is collected until October 2009 so we don’t 
# have the full data on this year. 

# What were the most popular movie genres year by year?
genresByYear <- edx %>% 
  separate_rows(genres, sep = "\\|") %>% 
  select(movieId, releaseyear, genres) %>% 
  group_by(releaseyear, genres) %>% 
  summarise(count = n()) %>% arrange(desc(releaseyear))

# Different periods show certain genres being more popular during those periods
# It will be very hard to incorporate genre into overall prediction given this fact
ggplot(genresByYear, aes(x = releaseyear, y = count)) + 
  geom_col(aes(fill = genres), position = 'dodge') + 
  theme_hc() + 
  ylab('Number of Movies') + 
  ggtitle('Popularity per year by Genre')

# View the number of times each user has reviewed movies  
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, binwidth=0.2, color="black", show.legend = FALSE, aes(fill = cut(n, 100))) + 
  scale_x_log10() + 
  ggtitle("User Reviews")
# Most users have reviewed less than 200 movies

# View release year vs rating
edx %>% group_by(releaseyear) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(releaseyear, rating)) +
  geom_point() +
  theme_hc() + 
  geom_smooth() +
  ggtitle("Release Year vs. Rating")
# Older "classics" get higher ratings. This could allow us to penalize a movie based on release year
# by a calculated weight.

# remove any variables we no longer need to reduce memory footprint
rm(movies_per_year, genresByYear)

# Our task is predict better than 50/50 (coin toss)
# Given we have 10 possible ratings, random chance would at best
# give us 1/10 odds, or 10% accuracy

##########################################################################################################################
# Start Prediction Approach
##########################################################################################################################

# We write a loss-function that computes the Residual Mean Squared Error ("typical error") as
# our measure of accuracy. The value is the typical error in star rating we would make
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# We predict a new rating to be the average rating of all movies in our training dataset,
# which gives us a baseline RMSE. We observe that the mean movie rating is a pretty generous > 3.5.
mu <- mean(edx$rating)
baseline_RMSE <- RMSE(edx$rating, mu)

# First rmse
naive_rmse <- RMSE(temp$rating, mu) # 1.06

# We generate a table to record our approaches and the RMSEs they generate.
rmse_results <- data_frame(method = "First Model", RMSE = naive_rmse)

# We know from experience that some movies are just generally rated higher than others. 
# We can use data to confirm this. For example, if we consider movies with more than 1,000 ratings, 
# the SE error for the average is at most 0.05. Yet plotting these averages we see much greater variability than 0.05:
edx %>% group_by(movieId) %>% 
  filter(n()>=1000) %>% 
  summarize(avg_rating = mean(rating)) %>% 
  qplot(avg_rating, geom = "histogram", color = I("black"), fill=I("navy"), bins=30, data = .)

# So our intuition that different movies are rated differently is confirmed by data. 
# So we can augment our previous model by adding a term to represent average ranking for a movie: 
movie_means <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Let's see how much our prediction improves. 
joined <- temp %>% 
  left_join(movie_means, by='movieId')
any(is.na(joined$b_i))

# Now we are ready to form a prediction
predicted_ratings <- mu + joined$b_i
model2_rmse <- RMSE(predicted_ratings, temp$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method ="Second Model",  
                                     RMSE = model2_rmse ))

# We already see a big improvement. Can we make it better?
# Let's explore where we made mistakes. 
temp %>% mutate(prediction = predicted_ratings, 
                residual   = predicted_ratings - temp$rating) %>%
  arrange(desc(abs(residual))) %>% 
  left_join(movies) %>%  
  select(title, prediction, residual) %>% slice(1:10) 

qplot(b_i, geom = "histogram", color = I("black"), fill=I("navy"), bins=25, data = movie_means)

# These all seem like obscure movies. Many of them have large predictions. Let's look at the top 10 worst and best movies
movie_means <-  left_join(movie_means, movies) 

# Here are the top ten movies:
arrange(movie_means, desc(b_i)) %>% 
  mutate(prediction = mu + b_i) %>%
  select(title, prediction) %>% 
  slice(1:10)

# Here are the bottom ten:
arrange(movie_means, b_i) %>% 
  mutate(prediction = mu + b_i) %>%
  select(title, prediction) %>% 
  slice(1:10)

# They all seem to be quite obscure. Let's look at how often they are rated.
edx %>%
  count(movieId) %>%
  left_join(movie_means) %>%
  arrange(desc(b_i)) %>% 
  mutate(prediction = mu + b_i) %>%
  select(title, prediction, n) %>% 
  slice(1:10)

edx %>%
  count(movieId) %>%
  left_join(movie_means) %>%
  arrange(b_i) %>% 
  mutate(prediction = mu + b_i) %>%
  select(title, prediction, n) %>% 
  slice(1:10) 

# So the supposed "best" and "worst" movies were rated by very few users. 
# These movies were mostly obscure ones. This is because with just a few users, we have more uncertainty. 
# Therefore, larger estimates of b_i, negative or positive, are more likely. These are "noisy" estimates 
# that we should not trust, especially when it comes to prediction. Large errors can increase our RMSE, 
# so we would rather be conservative when not sure.

# Regularization permits us to penalize large estimates that come from small sample sizes. It has commonalities 
# with the Bayesian approach that "shrunk" predictions. The general idea is to minimize the sum of squares equation 
# while penalizing for large values of b_i.

# Let's compute these regularized estimates of $b_i$ using lambda=5. Then, look at the top 10 best and worst movies now.
lambda <- 5
mu <- mean(edx$rating)
movie_reg_means <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) %>%
  left_join(movies) 

# Top 10 best
edx %>%
  count(movieId) %>%
  left_join(movie_reg_means) %>%
  arrange(desc(b_i)) %>% 
  mutate(prediction = mu + b_i) %>%
  select(title, prediction, n) %>% 
  slice(1:10) 

# Top ten worst
edx %>%
  count(movieId) %>%
  left_join(movie_reg_means) %>%
  arrange(b_i) %>% 
  mutate(prediction = mu + b_i) %>%
  select(title, prediction, n) %>% 
  slice(1:10) 

# Do we improve our results?
joined <- temp %>% 
  left_join(movie_reg_means, by='movieId') %>% 
  replace_na(list(b_i=0))

predicted_ratings <- mu + joined$b_i
model3_reg_rmse <- RMSE(predicted_ratings, temp$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Third Model Lambda=5",  
                                     RMSE = model3_reg_rmse ))

# We improved our results slightly. We can visualize how the predictions with small b_i are shrunken more towards 0.
data_frame(original = movie_means$b_i, 
           regularlized = movie_reg_means$b_i, 
           n = movie_reg_means$n_i) %>%
  ggplot(aes(original, regularlized, size=log10(n))) + 
  geom_point(shape=1, alpha=0.5)

# We can try other values of lambda:
lambdas <- seq(0, 8, 0.25)
mu <- mean(edx$rating)
tmp <- edx %>% 
  group_by(movieId) %>% 
  summarize(sum = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  joined <- temp %>% 
    left_join(tmp, by='movieId') %>% 
    mutate(b_i = sum/(n_i+l)) %>%
    replace_na(list(b_i=0))
  predicted_ratings <- mu + joined$b_i
  return(RMSE(predicted_ratings, temp$rating))
})

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# We gain a slight improvement
# We have improved the RMSE substantially from our initial naive guess.
# What else can we do to improve? 
# Let's compute the average rating for user u, for those that have rated over 100 movies. 
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n() >= 100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, binwidth=0.2, color = I("black"), fill=I("navy"), show.legend = FALSE)

# Note that there is substantial variability across users as well. This means some users 
# are harsher than others and implies that a further improvement to our model
# Now it is possible that some users appear to be harsher than others only because they 
# rate under-average movies. For this reason we prefer to estimate b_u taking into account the b_i. 
# The least squares estimates will do this but, again we do not want to use `lm` here. 
# Instead we will take the average of the residuals 

# We will use  lambda_2=5:
lambda_2 <- 5
user_reg_means <- edx %>% 
  left_join(movie_reg_means) %>%
  mutate(resids = rating - mu - b_i) %>% 
  group_by(userId) %>%
  summarize(b_u = sum(resids)/(n()+lambda_2))

joined <- temp %>% 
  left_join(movie_reg_means, by='movieId') %>% 
  left_join(user_reg_means, by='userId') %>% 
  replace_na(list(b_i=0, b_u=0))

predicted_ratings <- mu + joined$b_i + joined$b_u
model4_reg_rmse <- RMSE(predicted_ratings, temp$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Fourth Model LSE",  
                                     RMSE = model4_reg_rmse ))
# RMSE remains the same.

# Let's look at the accuracy of our current model
a <- predicted_ratings
a <- ceiling(a / 0.5) * 0.5
a[a <= 0.5] <- 0.5
a[a >= 5] <- 5
a<-as.factor(a)
summarization = confusionMatrix(a, as.factor(temp$rating))
summarization$overall[1] # 22.5 %

# Our accuracy is currently at 22.5% which is much better than a coin-toss, but still far away from
# the 50% minimum required to score any points.

##########################################################################################################################
# We can see this model is not going to get us very far. There are several known approaches for predicting recommended
# items or ratings, including using PCA, SVD, or one of the numerous libraries available such as recommenderlab.
#
# The real challenge here is not in finding a workable algorithm - even RandomForest will work quite well. The challenge 
# we have is with the dataset size itself. If this project used the 1M MovieLens set it would be fairly easy to use
# a plug-in approach using recommenderlab, however, as noted by other students, the large matrices required to be generated
# for the 10M dataset simply does not fit into the RAM available. I've tried this on Azure with 160GB allocated and all the 
# standard approaches simply failed. I could aim for 50% accuracy but would consider that a failure, really. So for this 
# project and dataset size we need to find an algorithm that allows us to split the dataset into trainable chunks, when if
# finally combined, fits into 32GB + swap space at least. I understand that many students may not have access to machines 
# with 32GB of RAM but unfortunately my score depends on accuracy. So if you cannot run this code, I am happy to demo.
##########################################################################################################################

# no longer required so remove these objects
rm(joined, summarization, movie_means, movie_reg_means, movielens, movies, ratings, rmse_results, temp, tmp, user_reg_means, a, predicted_ratings) 
gc() # force R to flush RAM after rm() statement

# Feature engineering - drop unused columns
edx <- edx[,-c(4,5,6,7)] 
validation <- validation[,-c(4,5,6)]

# remove users and movies not in validation set
# no point training for those, let's save time and RAM
edx <- edx %>% 
  semi_join(validation, by = "userId") %>%
  semi_join(validation, by = "movieId")

##########################################################################################################################
# Slope One implementation
#
# Slope One was introduced by Daniel Lemire and Anna Maclachlan in their paper 
# ‘Slope One Predictors for Online Rating-Based Collaborative Filtering’. This algorithm is one of the simplest ways to 
# perform collaborative filtering based on items’ similarity. This makes it very easy to implement and use, and accuracy of
# this algorithm equals the accuracy of more complicated and resource-intensive algorithms.
#
# The Slope One method operates with average differences of ratings between each item and makes predictions based on their 
# weighted value. 
#
# It is pretty fast, but for a huge dataset it needs a lot of RAM. For that reason we break the training set into 20
# smaller sets for training, and merge that into a single model at the end. This process takes around 2 hours on a desktop 
# but requires at least 32GB of ram and 12GB of extra swap space. Slope One was chosen as it is an algorithm that can be easily
# split into sets for training and re-combined without loss of accuracy.
#
# Code below from https://github.com/tarashnot/SlopeOne/tree/master/data (without further modification)
##########################################################################################################################

# Input: A data table of ratings. Should contain 3 columns: user_id (id of user, character), item_id (id of item, character) 
# and rating (rating of item by user, integer or numeric)
# Returns: A data table of (item_id1, item_id2, b, support) where b represents the average rating difference of 
# 'item 2 rating' - 'item 1 rating'. support represents number of ratings used to compute b
build_slopeone <- function(ratings, ...) {
  if (NROW(ratings) == 0) {
    return(data.table(data.frame(item_id1=c(), item_id2=c(), b=c(), support=c())))
  }
  score_diff_per_user <- dlply(ratings, .(user_id), function(rows) {
    if (NROW(rows) > 1) {
      # Get diffs for all item_id pairs.
      pair_rows_nums <- subset(
        expand.grid(rows_num1=1:NROW(rows), rows_num2=1:NROW(rows)),
        rows_num1 != rows_num2 &
          rows[rows_num1, 'item_id'] != rows[rows_num2, 'item_id'])
      data.table(
        item_id1=rows[pair_rows_nums$rows_num1, 'item_id'],
        item_id2=rows[pair_rows_nums$rows_num2, 'item_id'],
        diff=rows[pair_rows_nums$rows_num2, 'rating']
        - rows[pair_rows_nums$rows_num1, 'rating'])
    }
  }, ...)
  # ddply is slow when merging data frames within list while rbindlist is
  # much faster. rbindlist is limited in number of rows, exactly half we need to bind
  # hence we will split training set into 20 smaller chunks and combine the results
  # at the end of training
  score_diff_per_user <- rbindlist(score_diff_per_user)
  if (NROW(score_diff_per_user) == 0) {
    return(data.table(data.frame(item_id1=c(), item_id2=c(), b=c(), support=c())))
  }
  score_diff_per_user$item_id1 <- as.character(score_diff_per_user$item_id1)
  score_diff_per_user$item_id2 <- as.character(score_diff_per_user$item_id2)
  # Compute average score diff between item 1 and item 2.
  model <- score_diff_per_user[,
                               list(b=mean(diff), support=NROW(diff)),
                               by='item_id1,item_id2']
  setkey(model, item_id1, item_id2)
  return(model)
}

# Input: A data table of ratings. Should contain 3 columns: user_id (id of user, character), item_id (id of item, character) 
# and rating (rating of item by user, integer or numeric)
normalize_ratings <- function(ratings, ...) {
  result <- list()
  result$global <- ratings[, mean(rating)]
  result$user <- ratings[, list(mean_rating=mean(rating)), by='user_id']
  result$item <- ratings[, list(mean_rating=mean(rating)), by='item_id']
  
  ratings$rating <- ratings$rating - result$global
  setkey(result$user, user_id)
  ratings$rating <- ratings$rating - result$user[J(ratings$user_id), ]$mean_rating
  setkey(result$item, item_id)
  ratings$rating <- ratings$rating - result$item[J(ratings$item_id), ]$mean_rating
  result$ratings <- ratings
  return(result)
}

# Inputs: Data table produced by build_slopeone, data table of of (user_id, item_id) to predict ratings
# and a data table of known ratings.
# Returns: A data table containing (user_id, item_id, predicted_rating)
predict_slopeone <-function(model, targets, ratings, ...) {
  setkey(ratings, user_id)
  adply(targets,
        1,
        function(row) {
          data.frame(
            predicted_rating=predict_slopeone_for_user(
              model, row$item_id, ratings[J(row$user_id), ]))
        }, ...)
}

# Inputs: A data table produced by build_slopeone, target item id to predict rating, data table of user's known ratings.
# Returns: predicted rating score
predict_slopeone_for_user <- function(model, target_item_id, ratings) {
  # If target_id is already rated by the user, return that rating.
  already_rated <- subset(ratings, ratings$item_id == target_item_id)
  if (NROW(already_rated) == 1) {
    return(already_rated$rating)
  } else if (NROW(already_rated) > 1) {
    warning(paste(target_item_id,
                  ' is already rated by user, but there are multiple ratings.'))
    return(already_rated[1, ]$rating)
  }
  if (NROW(model) == 0) {
    return(NA)
  }
  # Compute weighted average ratings.
  ratings <- rename(ratings, c('item_id'= "item_id1"))
  ratings <- cbind(ratings, item_id2=target_item_id)
  setkey(ratings, item_id1, item_id2)
  joined <- model[ratings, ]
  joined <- joined[complete.cases(joined), ]
  if (NROW(joined) == 0) {
    return(NA)
  }
  return(sum(joined[, (b + rating) * support]) /
           sum(joined[, sum(support)]))
}

# Inputs: normalization information generated by normalize_ratings and a data table of ratings.
# Returns: a ratings data table after un-normalization
unnormalize_ratings <- function(normalized, ratings) {
  ratings$predicted_rating <- ifelse(is.na(ratings$predicted_rating), 0,
                                     ratings$predicted_rating)
  ratings$predicted_rating <- ratings$predicted_rating + normalized$global
  setkey(normalized$user, user_id)
  user_mean <- normalized$user[J(ratings$user_id), ]$mean_rating
  ratings$predicted_rating <- ratings$predicted_rating +
    ifelse(!is.na(user_mean), user_mean, 0)
  setkey(normalized$item, item_id)
  item_mean <- normalized$item[J(ratings$item_id), ]$mean_rating
  ratings$predicted_rating <- ratings$predicted_rating +
    ifelse(!is.na(item_mean), item_mean, 0)
  return(ratings)
}

##########################################################################################################################
# Training - DO NOT attempt this unless you have 32GB RAM and at least 12GB swap space.
#
# You can sign up for a free 1 month trial of Azure, launch a VM with 56GB RAM and 20 cores, install R and train for free
#
##########################################################################################################################

# Split into 20 subsets to fit into RAM (32GB + 12GB swap space)
# Alternatively, split into 10 subsets if you have 56GB or more
# CPU usage is low, just RAM requirement is high
# If we don't split into subsets, this will consume 160GB of RAM (Trust me, it happened to me on my Azure box)

# We split along userId to retain consistency after merging the results together.
x <- unique(edx$userId)

# The 0:20 / 20 here is to request 20 splits. Adjust to 0:10/10 if you have a ton of RAM.
q <- split(x, cut(x, quantile(x, prob = 0:20 / 20, names = FALSE), include = TRUE))
rm(x) # good practice to wipe items when you're done as R can be a memory hog.

mainModel <- NULL # build it up as we go

for (c in 1:20) { # doing 20 chunks as specified below, change to 10 if you changes the 20's on line 586
  chunk <- edx %>% filter(edx$userId %in% unlist(q[c])) # grab this chunk
  chunk <- data.table(chunk)
  names(chunk) <- c("user_id", "item_id", "rating") # Slope One code above wants these column names
  chunk$user_id <- as.character(chunk$user_id)
  chunk$item_id <- as.character(chunk$item_id)
  setkey(chunk, user_id, item_id)
  chunk <- normalize_ratings(chunk)
  
  model <- build_slopeone(chunk$ratings) # Train
  
  if (is.null(mainModel)) {
    mainModel <- model
  } else {
    mainModel <- rbind(mainModel, model) # Combine results back into our main model as we go
  }
  rm(model, chunk)
  gc() # force R to dump objects we just removed from RAM
}

rm(q) # good practice for large datasets
gc() # rm() does nothing unless you force garbage collection

# We can save the model for re-use, however it is 22GB in RAM so it will be massive, even compressed
# save(mainModel,file="mainModel.Rda", compress="bzip2")

# Slope One code requires specific names and column types
names(edx) <- c("user_id", "item_id", "rating")
edx$user_id <- as.character(edx$user_id)
edx$item_id <- as.character(edx$item_id)
edx <- data.table(edx)
setkey(edx, user_id, item_id)
edx <- normalize_ratings(edx)

names(mainModel) <- c("user_id", "item_id", "b", "support")

scoring <- validation$rating # keep for our RMSE check and confusion matrix at the end

# Slope One requires specific names and column types
validation$userId <- as.character(validation$userId)
validation$movieId <- as.character(validation$movieId)

#######################################################
validation$rating <- NA # we need to predict this
#######################################################

names(validation) <- c("user_id", "item_id", "rating")

# The code below takes a long time to complete, be patient. (1 hour plus)
# Predict using our model, and training ratings, using only userId & movieId from validation set
predictions <- predict_slopeone(mainModel, validation[ , 1:2], edx$ratings)
unnormalized_predictions <- unnormalize_ratings(normalized = edx, ratings = predictions)

# Round predictions to nearest 0.5
# Limit it inside range 0.5 to 5
a <- unnormalized_predictions$predicted_rating
a <- ceiling(a / 0.5) * 0.5
a[a <= 0.5] <- 0.5
a[a >= 5] <- 5

# check RMSE after rounding
rmse_slopeone <- RMSE(scoring,a)
rmse_slopeone # 0.192

# if you cannot run this code, you can download calc_rmse.R from here:
# https://github.com/gideonvos/MovieLens/blob/master/
# Run that using the rubric.csv and submission.csv I generated under the old rules
# Note you will get 0.192 as well

# Generate our submission file (not required, so remarked out, but feel free to run old grading.py on it)
names(validation) <- c("userId", "movieId", "rating")
validation$userId <- as.integer(validation$userId)
validation$movieId <- as.integer(validation$movieId)
#write.csv(validation %>% select(userId, movieId) %>% mutate(rating = a), "submission.csv", na = "", row.names=FALSE)

# Check accuracy via confusion matrix
a <- as.factor(a)
b <- as.factor(scoring) # actual true results we stored earlier
summarization = confusionMatrix(a, b)
summarization # Accuracy: 85.14%

# screenshot of confusion matrix available here:
# https://github.com/gideonvos/MovieLens/blob/master/SlopeOneCM.png

# All done, clean up
rm(edx, validation, mainModel, scoring, predictions, unnormalized_predictions)

##########################################################################################################################
# Conclusion:
#
# I tried Naive Bayes, Random Forest, Tensorflow Neural Networks, PCA, SVD, Recommenderlab, KNN, Kmeans, and various others
# Some were fast but the accuracy poor. Others were accurate on smaller sets (Random Forest) but simply could not scale
# to this data set size, and offered no reliable means to split and combine as I did with Slope One.
# While still requiring a lot of RAM, Slope One was the most repeatable. I re-ran this model with subsets of 10 and 20.
# The 10 set required 80GB of RAM while to 20 set managed to fit into 32GB + 12GB swap space.
#
# When tested, both the 10 set and 20 set scored exactly the same - 85%, thus proving the splitting and combining
# approach works well on very large datasets using Slope One without accuracy loss.
#
# Keep in mind though that splitting into sets of 40 will still require merging the model at the end, needing 22GB of RAM.
#
##########################################################################################################################

# GitHub repository is here: https://github.com/gideonvos/MovieLens

# Thank you for reviewing my code!
# Gideon





