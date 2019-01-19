library(data.table)

score <- fread(text = gsub("::", "\t", readLines("/rubric.csv")), data.table=TRUE, col.names = c("userId", "movieId", "rating"))

pred <- fread(text = gsub("::", "\t", readLines("/submission.csv")), data.table=TRUE, col.names = c("userId", "movieId", "rating"))

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

rmse <- RMSE(score$rating, pred$rating)
rmse # 0.192





