options(digits=3)

#Creating a test and training set for edx set
edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.5, list = FALSE)
edx_test <- edx %>% slice(edx_index)
edx_train <- edx %>% slice(-edx_index)

#Making sure that both edx_test and edx_train have same users and movies
edx_test <- edx_test %>% semi_join(edx_train, by = "movieId") %>% semi_join(edx_train, by = "userId")

#Calculating average of all ratings, equal to mu
mu <- mean(edx_train$rating)

#Estimating movie effect
movie_ef <- edx_train %>% group_by(movieId) %>% summarise(b_i = mean(rating-mu))

#Estimating user effect (+ left joining with movie_ef to have b_i accessible)
user_ef <- edx_train %>% left_join(movie_ef, by="movieId") %>% group_by(userId) %>% summarise(b_u=mean(rating-mu-b_i))

#Making predictions on average + movie effect + user effect
pred <- edx_test %>% left_join(movie_ef, by="movieId") %>% left_join(user_ef, by="userId") %>% mutate(pred=mu+b_i+b_u) %>% pull(pred)
rmse <- sqrt(mean((pred-edx_test$rating)^2))
rmse

#Regularizing movie effect, with finding alpha that minimizes rmse. Previously narrowed it down from 0:10 range
alpha <- seq(3,4,0.05)
rmse_reg_a <- sapply(alpha, function(a){
  b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+a))
  b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=mean(rating-mu-b_i))
  pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% mutate(pred=mu+b_i+b_u) %>% pull(pred)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(alpha,rmse_reg_a)
alpha <- alpha[which.min(rmse_reg_a)]
alpha

b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=mean(rating-mu-b_i))
pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% mutate(pred=mu+b_i+b_u) %>% pull(pred)
sqrt(mean((pred-edx_test$rating)^2))

#Regularizing user effect, with finding lambda that minimizes rmse. Previously narrowed it down from 0:10 range
lambda <- seq(4.5,5.5,0.05)
rmse_reg_l <- sapply(lambda, function(l){
  b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
  b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+l))
  pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% mutate(pred=mu+b_i+b_u) %>% pull(pred)
  sqrt(mean((pred-edx_test$rating)^2))
})
plot(lambda,rmse_reg_l)
lambda <- lambda[which.min(rmse_reg_l)]
lambda

b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% mutate(pred=mu+b_i+b_u) %>% pull(pred)
sqrt(mean((pred-edx_test$rating)^2))

#Final test of RMSE on test set
rmse <- sqrt(mean((pred-edx_test$rating)^2))
rmse

#Final test of RMSE on validation set
b_i <- edx %>% group_by(movieId) %>% summarise(b_i = sum(rating-mu)/(n()+alpha))
b_u <- edx %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u=sum(rating-mu-b_i)/(n()+lambda))
pred <- validation %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% mutate(pred=mu+b_i+b_u) %>% pull(pred)
rmse_val <- sqrt(mean((pred-validation$rating)^2))
rmse_val


