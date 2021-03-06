#############################################
# Download and prepare the movielens dataset
#############################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
options(digits=5)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

############################################################################################
# Creating additonal columns in movielens (movie release year and weekday of the rating time)
############################################################################################

glimpse(movielens)

movielens <- mutate(movielens, year_of_release = as.numeric(str_sub(title, start=-5, end=-2)), 
                    rateday=wday(as_datetime(timestamp),week_start = 1))

#########################################################
# Create edx set, validation set (final hold-out test set)
#########################################################

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId, movieId, year_of_release and rateday in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId") %>% 
  semi_join(edx, by="year_of_release") %>% 
  semi_join(edx, by="rateday") %>%
  semi_join(edx, by="genres")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##################################################################################
# Creating train and test sets out of edX set, for sake of modelling aproach tests
##################################################################################

# Creating a test and training set for edx set
edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.5, list = FALSE)
edx_test <- edx %>% slice(edx_index)
edx_train <- edx %>% slice(-edx_index)

# Making sure that both edx_test and edx_train have same set of userId, movieId, year_of_release and rateday
edx_test <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>% 
  semi_join(edx_train, by = "userId") %>% 
  semi_join(edx_train, by="year_of_release") %>% 
  semi_join(edx_train, by="rateday") %>%
  semi_join(edx_train, by="genres")

######################################
# Model creation - basic naive model
######################################

# Calculating average of all ratings, equal to mu
mu <- mean(edx_train$rating)

# Estimating movie effect
b_i <- edx_train %>% 
  group_by(movieId) %>% 
  summarise(b_i = mean(rating-mu))

# Estimating user effect (+ left joining with movie_ef to have b_i accessible)
b_u <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarise(b_u=mean(rating-mu-b_i))

# Estimating release year effect (+ left joining with movie_ef and user_ef to have b_i and b_u accessible)
b_y <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  group_by(year_of_release) %>% 
  summarise(b_y=mean(rating-mu-b_i-b_u))

# Estimating rating day of the week effect (+ left joining to have b_i, b_u and b_y accessible)
b_d <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>% 
  group_by(rateday) %>% 
  summarise(b_d=mean(rating-mu-b_i-b_u-b_y))

# Estimating rating day of the genres (+ left joining to have b_i, b_u, b_y and b_d accessible)
b_g <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>% 
  left_join(b_d, by="rateday") %>%
  group_by(genres) %>% 
  summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))

# Creating a predictions function
predictions<- function(x,i,u,y,d,g){
  x %>% 
    left_join(i, by="movieId") %>% 
    left_join(u, by="userId") %>% 
    left_join(y, by="year_of_release") %>% 
    left_join(d, by="rateday") %>%
    left_join(g, by="genres") %>%
    mutate(pred=mu+b_i+b_u+b_y+b_d+b_g) %>%
    pull(pred)
}
  
# Making predictions on average + movie effect + user effect + release year effect + rate day of the week + genres effect)
pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)
rmse_naive <- sqrt(mean((pred-edx_test$rating)^2))
print(c("The RMSE for naive model is:", rmse_naive), quote=FALSE, digits = 5)

############################################
# Model creation - regularization parameters
############################################

# Regularizing movie effect, with finding alpha that minimizes rmse. Previously narrowed it down from 0:10 range for speed of calculations purposes.
alpha <- seq(3.4,4.2,0.05)

rmse_reg_a <- sapply(alpha, function(a){
  b_i <- edx_train %>% 
    group_by(movieId) %>% 
    summarise(b_i = sum(rating-mu)/(n()+a))
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarise(b_u=mean(rating-mu-b_i))
  b_y <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    group_by(year_of_release) %>% 
    summarise(b_y=mean(rating-mu-b_i-b_u))
  b_d <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>% 
    group_by(rateday) %>% 
    summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
  b_g <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>%
    left_join(b_d, by="rateday") %>%
    group_by(genres) %>% 
    summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))
  pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)
  sqrt(mean((pred-edx_test$rating)^2))
})

plot(alpha,rmse_reg_a, xlab ="Regularization parameter for movie effect (alpha)", ylab="RMSE")

alpha <- alpha[which.min(rmse_reg_a)]

print(c("The alpha parameter to get the smallest RMSE is",alpha), quote = FALSE)

b_i <- edx_train %>% 
  group_by(movieId) %>% 
  summarise(b_i = sum(rating-mu)/(n()+alpha))

b_u <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarise(b_u=mean(rating-mu-b_i))

b_y <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  group_by(year_of_release) %>% 
  summarise(b_y=mean(rating-mu-b_i-b_u))

b_d <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>% 
  group_by(rateday) %>% 
  summarise(b_d=mean(rating-mu-b_i-b_u-b_y))

b_g <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>%
  left_join(b_d, by="rateday") %>%
  group_by(genres) %>% 
  summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))

pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)

rmse_moviereg <- sqrt(mean((pred-edx_test$rating)^2))

print(c("The RMSE for regularized movie effect + simple user, year of release and rating day of the week effect is:", rmse_moviereg), quote=FALSE, digits = 5)

# Regularizing user effect, with finding lambda that minimizes rmse. Previously narrowed it down from 0:10 range for speed of calculations purposes.
lambda <- seq(4.2,5.2,0.1)
rmse_reg_l <- sapply(lambda, function(l){
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarise(b_u=sum(rating-mu-b_i)/(n()+l))
  b_y <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    group_by(year_of_release) %>% 
    summarise(b_y=mean(rating-mu-b_i-b_u))
  b_d <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>% 
    group_by(rateday) %>% 
    summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
  b_g <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>%
    left_join(b_d, by="rateday") %>%
    group_by(genres) %>% 
    summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))
  pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(lambda,rmse_reg_l, xlab ="Regularization parameter for user effect (lambda)", ylab="RMSE")
lambda <- lambda[which.min(rmse_reg_l)]
print(c("The lambda parameter to get the smallest RMSE is",lambda), quote = FALSE)

b_u <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))

b_y <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  group_by(year_of_release) %>% 
  summarise(b_y=mean(rating-mu-b_i-b_u))

b_d <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>% 
  group_by(rateday) %>% 
  summarise(b_d=mean(rating-mu-b_i-b_u-b_y))

b_g <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>%
  left_join(b_d, by="rateday") %>%
  group_by(genres) %>% 
  summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))

pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)

rmse_userreg <- sqrt(mean((pred-edx_test$rating)^2))
print(c("The RMSE for regularized movie and user effect + simple year of release and rating day of the week effect is:", rmse_userreg), quote=FALSE, digits = 5)

# Regularizing year_of_release effect, with finding delta that minimizes rmse. Previously narrowed it down from 0:50 range for speed of calculations purposes.
delta <- seq(30,45,0.5)
rmse_reg_d <- sapply(delta, function(d){
  b_y <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    group_by(year_of_release) %>% 
    summarise(b_y=sum(rating-mu-b_i-b_u)/(n()+d))
  b_d <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>% 
    group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))
  b_g <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>%
    left_join(b_d, by="rateday") %>%
    group_by(genres) %>% 
    summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))
  pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(delta,rmse_reg_d, xlab ="Regularization parameter for year of release effect (delta)", ylab="RMSE")
delta <- delta[which.min(rmse_reg_d)]
print(c("The delta parameter to get the smallest RMSE is",delta), quote = FALSE)

b_y <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  group_by(year_of_release) %>% 
  summarise(b_y=sum(rating-mu-b_i-b_u)/(n()+delta))

b_d <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>% 
  group_by(rateday) %>% summarise(b_d=mean(rating-mu-b_i-b_u-b_y))

b_g <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>%
  left_join(b_d, by="rateday") %>%
  group_by(genres) %>% 
  summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))

pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)

rmse_yearreg <- sqrt(mean((pred-edx_test$rating)^2))
print(c("The RMSE for regularized movie, user and year of release effect + simple rating day of the week effect is:", rmse_yearreg), quote=FALSE, digits = 5)

# Regularizing rating day of the week effect, with finding kappa that minimizes rmse. Previously narrowed it down from 0:1,000,000 range for speed of calculations purposes.
kappa <- seq(165000,185000,1000)
rmse_reg_k <- sapply(kappa, function(k){
  b_d <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>% 
    group_by(rateday) %>% 
    summarise(b_d=sum(rating-mu-b_i-b_u-b_y)/(n()+k))
  b_g <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>%
    left_join(b_d, by="rateday") %>%
    group_by(genres) %>% 
    summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))
  pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(kappa,rmse_reg_k, xlab ="Regularization parameter for rating day of the week (kappa)", ylab="RMSE")
kappa <- kappa[which.min(rmse_reg_k)]
print(c("The kappa parameter to get the smallest RMSE is",delta), quote = FALSE)

b_d <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>% 
  group_by(rateday) %>% 
  summarise(b_d=sum(rating-mu-b_i-b_u-b_y)/(n()+kappa))

b_g <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>%
  left_join(b_d, by="rateday") %>%
  group_by(genres) %>% 
  summarise(b_g=mean(rating-mu-b_i-b_u-b_y-b_d))

pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)

rmse_ratedayreg <- sqrt(mean((pred-edx_test$rating)^2))
print(c("The RMSE for regularized movie, user, year of release effect and rating day of the week effect is:", rmse_ratedayreg), quote=FALSE, digits = 5)

# Regularizing rating day of the week effect, with finding omega that minimizes rmse. Previously narrowed it down from 0:100 range for speed of calculations purposes.
omega <- seq(0,15,0.5)
rmse_reg_o <- sapply(omega, function(o){
  b_g <- edx_train %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_y, by="year_of_release") %>%
    left_join(b_d, by="rateday") %>%
    group_by(genres) %>% 
    summarise(b_g=sum(rating-mu-b_i-b_u-b_y)/(n()+o))
  pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(omega,rmse_reg_o, xlab ="Regularization parameter for movie genres (omega)", ylab="RMSE")
omega <- omega[which.min(rmse_reg_o)]
print(c("The omega parameter to get the smallest RMSE is",omega), quote = FALSE)

b_g <- edx_train %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>%
  left_join(b_d, by="rateday") %>%
  group_by(genres) %>% 
  summarise(b_g=sum(rating-mu-b_i-b_u-b_y)/(n()+omega))

pred <- predictions(edx_test, b_i, b_u, b_y, b_d, b_g)

rmse_genrereg <- sqrt(mean((pred-edx_test$rating)^2))
print(c("The RMSE for regularized movie, user, year of release effect and rating day of the week effect is:", rmse_ratedayreg), quote=FALSE, digits = 5)

#####################################
# Final test of the prediction model 
#####################################

# Training the model on edx set
mu <- mean(edx$rating)
b_i <- edx %>% 
  group_by(movieId) %>% 
  summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
b_y <- edx %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  group_by(year_of_release) %>% 
  summarise(b_y=sum(rating-mu-b_i-b_u)/(n()+delta))
b_d <- edx %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>% 
  group_by(rateday) %>% 
  summarise(b_d=sum(rating-mu-b_i-b_u-b_y)/(n()+kappa))
b_g <- edx %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_y, by="year_of_release") %>%
  left_join(b_d, by="rateday") %>%
  group_by(genres) %>% 
  summarise(b_g=sum(rating-mu-b_i-b_u-b_y)/(n()+omega))

# Generating predictions on validation set
pred <- predictions(validation, b_i, b_u, b_y, b_d, b_g)

# Calculating final RMSE
rmse_val <- sqrt(mean((pred-validation$rating)^2))
print(c("Final RMSE is:", rmse_val), quote=FALSE, digits = 5)
