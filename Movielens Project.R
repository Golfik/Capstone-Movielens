##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

########################################
#Creating additonal columns in movielens
########################################
#Creating year_of_release and weekday of rating column from title.
movielens <- mutate(movielens, year_of_release = as.numeric(str_sub(title, start=-5, end=-2)), rateday=wday(as_datetime(timestamp),week_start = 1))

########################################
#Test set creation
########################################

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

rm(dl, ratings, movies, test_index, temp, movielens, removed)

########################################################
#Modeling
############################

options(digits=5)
library(lubridate)

#Creating a test and training set for edx set
edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.5, list = FALSE)
edx_test <- edx %>% slice(edx_index)
edx_train <- edx %>% slice(-edx_index)

#Making sure that both edx_test and edx_train have same users and movies (day of the week are not unique)
edx_test <- edx_test %>% semi_join(edx_train, by = "movieId") %>% semi_join(edx_train, by = "userId") %>% semi_join(edx_train, by="year_of_release")

#Calculating average of all ratings, equal to mu
mu <- mean(edx_train$rating)

#Estimating movie effect
b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = mean(rating-mu))

#Estimating user effect (+ left joining with movie_ef to have b_i accessible)
b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=mean(rating-mu-b_i))

#Estimating release year effect (+ left joining with movie_ef and user_ef to have b_i and b_u accessible)
b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=mean(rating-mu-b_i-b_u))

#Estimating rating day of the week effect (+ left joining to have b_i, b_u and b_y accessible)
b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))

#Making predictions on average + movie effect + user effect + release year effect + rate day of the week effect
pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release")  %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
rmse <- sqrt(mean((pred-edx_test$rating)^2))
rmse

#Regularizing movie effect, with finding alpha that minimizes rmse. Previously narrowed it down from 0:10 range
alpha <- seq(3,4,0.05)
rmse_reg_a <- sapply(alpha, function(a){
  b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+a))
  b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=mean(rating-mu-b_i))
  b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=mean(rating-mu-b_i-b_u))
  b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
  pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release")  %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(alpha,rmse_reg_a)
alpha <- alpha[which.min(rmse_reg_a)]
alpha

b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=mean(rating-mu-b_i))
b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=mean(rating-mu-b_i-b_u))
b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
sqrt(mean((pred-edx_test$rating)^2))

#Regularizing user effect, with finding lambda that minimizes rmse. Previously narrowed it down from 0:10 range
lambda <- seq(4.5,5.5,0.1)
rmse_reg_l <- sapply(lambda, function(l){
  b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
  b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+l))
  b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=mean(rating-mu-b_i-b_u))
  b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
  pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release")  %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(lambda,rmse_reg_l)
lambda <- lambda[which.min(rmse_reg_l)]
lambda

b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=mean(rating-mu-b_i-b_u))
b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release")  %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
sqrt(mean((pred-edx_test$rating)^2))

#Regularizing year_of_release effect, with finding delta that minimizes rmse. Previously narrowed it down from 0:50 range
delta <- seq(28,29,0.1)
rmse_reg_d <- sapply(delta, function(d){
  b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
  b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
  b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=sum(rating-mu-b_i-b_u)/(n()+d))
  b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
  pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release")  %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(delta,rmse_reg_d)
delta <- delta[which.min(rmse_reg_d)]
delta

b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=sum(rating-mu-b_i-b_u)/(n()+delta))
b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release")  %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
sqrt(mean((pred-edx_test$rating)^2))

#Regularizing rate day of the week effect, with finding delta that minimizes rmse. Previously narrowed it down from 0:1,000,000 range
kappa <- seq(140000,155000,1000)
rmse_reg_k <- sapply(kappa, function(k){
  b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
  b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
  b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=sum(rating-mu-b_i-b_u)/(n()+delta))
  b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=sum(rating-mu-b_i-b_u-b_y)/(n()+k))
  pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release")  %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(kappa,rmse_reg_k)
kappa <- kappa[which.min(rmse_reg_k)]
kappa

b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
b_y <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=sum(rating-mu-b_i-b_u)/(n()+delta))
b_d <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=sum(rating-mu-b_i-b_u-b_y)/(n()+kappa))
pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release")  %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)
sqrt(mean((pred-edx_test$rating)^2))

#Final test of RMSE on test set
rmse <- sqrt(mean((pred-edx_test$rating)^2))
rmse

#Final test of RMSE on validation set
b_i <- edx %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
b_y <- edx %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(year_of_release) %>% summarise(b_y=sum(rating-mu-b_i-b_u)/(n()+delta))
b_d <- edx %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% group_by(rateday) %>% summarise(b_d=sum(rating-mu-b_i-b_u-b_y)/(n()+kappa))
pred <- validation %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_y, by="year_of_release") %>% left_join(b_d, by="rateday") %>% mutate(pred=mu+b_i+b_u+b_y+b_d) %>% pull(pred)

rmse_val <- sqrt(mean((pred-validation$rating)^2))
rmse_val
