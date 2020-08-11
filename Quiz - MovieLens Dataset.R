#Question 1. How many rows and columns are there in the edx dataset?

dim(edx)

#Question 2. How many zeros were given as ratings in the edx dataset? How many threes were given as ratings in the edx dataset?

edx %>% filter(rating==0) %>% tally()
edx %>% filter(rating==3) %>% tally()

#Question 3. How many different movies are in the edx dataset?

n_distinct(edx$movieId)

#Question 4.How many different users are in the edx dataset?.

n_distinct(edx$userId)

#Question 5. How many movie ratings are in each of the following genres in the edx dataset?

genres <- c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g){
  sum(str_detect(edx$genres, g))
})

#Question 6. Which movie has the greatest number of ratings?

edx %>% group_by(movieId, title) %>% summarise(count=n()) %>% arrange(desc(count))

#Question 7. What are the five most given ratings in order from most to least?

edx %>% group_by(rating) %>% summarise(count=n()) %>% arrange(desc(count)) %>% top_n(5)

#Question 8. Are half star ratings are less common than whole star ratings?

edx %>% group_by(rating) %>% summarize(count = n())
