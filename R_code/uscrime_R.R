sort_order <- function(order) {
  # Define the fixed order
  fixed_order <- list(
    c(0),
    c(0, 3),
    c(0, 2),
    c(0, 2, 3),
    c(0, 1),
    c(0, 1, 3),
    c(0, 1, 2),
    c(0, 1, 2, 3)
  )
  fixed_order <- lapply(fixed_order, as.integer)
  
  # Function to find the index of a sublist in the fixed order
  find_index <- function(sublist) {
    for (i in seq_along(fixed_order)) {
      if (identical(sublist, fixed_order[[i]])) {
        return(i)
      }
    }
    return(NA)  # Return NA if not found
  }
  
  # Apply the find_index function to each element in the order list
  indices <- sapply(order, find_index)
  
  # Sort the original list based on the indices
  sorted_order <- order[order(indices)]
  
  # Return the indices of the sorted order in the original list
  sorted_indices <- match(sorted_order, order)
  
  return(sorted_indices)
}




####################################
###### US Crime ####################
####################################
path <- "D:/Jeff/MSU/Research Project/Project 1/code"
#data(UScrime, package = "MASS")
USCrime = read.csv(file.path(path, "Data", "USCrime.csv"))

sub_dt <- USCrime[,c(1,3,14,16)+1]
log_sub_dt <- log(sub_dt)
log_sub_dt$y <- round(log_sub_dt$y, 4)

# log center X
logC_sub_dt <- log_sub_dt
logC_sub_dt$M <- log_sub_dt$M - mean(log_sub_dt$M)
logC_sub_dt$Ed <- log_sub_dt$Ed - mean(log_sub_dt$Ed)
logC_sub_dt$Prob <- log_sub_dt$Prob - mean(log_sub_dt$Prob)

# log center scale X
logCS_sub_dt <- logC_sub_dt
logCS_sub_dt$M <- logC_sub_dt$M / sd(log_sub_dt$M)
logCS_sub_dt$Ed <- logC_sub_dt$Ed / sd(log_sub_dt$Ed)
logCS_sub_dt$Prob <- logC_sub_dt$Prob / sd(log_sub_dt$Prob)
logCS_sub_dt$y <- (logC_sub_dt$y-mean(logC_sub_dt$y)) / sd(logC_sub_dt$y)

# k=3 model
lm_crime <- bas.lm(y ~ .,
                   data = logCS_sub_dt,
                   prior = "g-prior",
                   modelprior = uniform(), initprobs = "eplogp", alpha = 47,
                   force.heredity = FALSE, pivot = TRUE)


summary(lm_crime)
coef_001 <- coef(lm_crime)
round(coef_001[["postprobs"]], 3)
post_probs = lm_crime[["postprobs"]]
order_old <- lm_crime[["which"]]

sorted_indices <- sort_order(order_old)
print(sorted_indices)

order_old[[8]]

# Reordered vector
reordered_post_probs <- post_probs[sorted_indices]
sum(round(reordered_post_probs, 3))

sum(reordered_post_probs)
reordered_post_probs



