library(BAS)

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





######################################
######### Simulated Data #############
######################################

LM <- read.csv(file.path(path, "Data/Toy_data", "LM_size100_rho08.csv"))
size = 100
data = LM[,2:5]

lm_001 <- bas.lm(Y ~ .,
                 data = data,
                 prior = "g-prior",
                 modelprior = uniform(), initprobs = "eplogp", alpha = size,
                 force.heredity = FALSE, pivot = TRUE
)


summary(lm_001)
coef_001 <- coef(lm_001)
round(coef_001[["postprobs"]], 3)
post_probs = lm_001[["postprobs"]]
order_old <- lm_001[["which"]]

sorted_indices <- sort_order(order_old)
print(sorted_indices)

order_old[[8]]

# Reordered vector
reordered_post_probs <- post_probs[sorted_indices]
sum(round(reordered_post_probs, 3))

sum(reordered_post_probs)
reordered_post_probs

