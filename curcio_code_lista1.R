  # Métodos Númericos - PS1

  # Required packages: 
  #install.packages("xtable")
  library(stargazer)
  library(ggplot2)
  library(gridExtra)

  # AR1: zt = ρzt−1 + et; ρ = 0.95 e σ = 0.007 (Cooley e Prescott)
  
  # 1. Discretization using Tauchen's Method with 9 points.
  # Parameters
  sigma = 0.007
  #rho = 0.95
  m = 3 # Scaling parameter
  #N = 9 # Number of points
  
  # We'll build a function for tauchen's method to be reapplied in other items
  tauchen <- function(N, rho){

  # Grid specification:
  zN = m * (sigma / sqrt(1 - rho^2)) # Upper bound
  grid_points <- seq(-zN, zN, length.out = N) # Grid points 

  # Transition Probabilities:
  # Initialize the transition probability matrix
  P <- matrix(0, N, N)

  # Precompute common expressions
  half_d <- (grid_points[2] - grid_points[1]) / 2

  # Compute transition probabilities
  for (i in 1:N) {
  # Compute probabilities for the first and last columns
  P[i, 1] <- pnorm((grid_points[1] - rho * grid_points[i] + half_d) / sigma)
  P[i, N] <- 1 - pnorm((grid_points[N] - rho * grid_points[i] - half_d) / sigma)
  
  # Compute probabilities for intermediate columns
  for (j in 2:(N - 1)) {
  H <- (grid_points[j] - rho * grid_points[i] + half_d) / sigma
  L <- (grid_points[j] - rho * grid_points[i] - half_d) / sigma
  P[i, j] <- pnorm(H) - pnorm(L)
  }
  }
  
  return(list(a=grid_points,b=P))
  }
  
  grid <- tauchen(9, 0.95)$a
  transition_matrix <- tauchen(9, 0.95)$b
  
  # LaTeX code
  stargazer(grid,summary=FALSE, rownames=FALSE)
  stargazer(transition_matrix, summary=FALSE, rownames=FALSE)
  
  
  # 2. Discretization using Rouwenhorst's Method with 9 points
  #We will also build a function for this item
  
  rouwenhorst <- function(N, rho){
  # Grid specification
  sigma_z = sqrt(sigma^2/(1-rho^2))
  zN = sigma_z * sqrt(N - 1)
  grid_points <- seq(- zN, zN, length.out=N)
  
  # Compute transition matrix recursively
  p <- (1 + rho) / 2
  P <- matrix(c(p, 1 - p, 1 - p, p), nrow = 2, byrow = TRUE) #P2 on the slide
  
  # Expand matrix using the direct formula from slides
  for (n in 3:N) {
  P <- p * rbind(cbind(P, 0), 0) + (1 - p) * rbind(cbind(0, P), 0) +
  (1 - p) * rbind(0, cbind(P, 0)) + p * rbind(0, cbind(0, P))
  }
  
  # Normalization
  P <- P / rowSums(P)
  
  return(list(a = grid_points,b = P))
  }
  
  grid <- rouwenhorst(9,0.95)$a
  transition_matrix <- rouwenhorst(9,0.95)$b
  
  # LaTeX code
  stargazer(grid,summary=FALSE, rownames=FALSE)
  stargazer(transition_matrix, summary=FALSE, rownames=FALSE)
  
  
  #3. Continuous process for 10000 periods
  # Simulation of the AR1
  AR1 <- function(periods, rho) {
  set.seed(123)
  error <- rnorm(periods, mean = 0, sd = sigma)
    
  # initialize vector
  zAR <- numeric(periods)  
  zAR[1] <- error[1] #first entry
    
  for (i in 2:periods) {
  zAR[i] <- rho * zAR[i - 1] + error[i] 
  }
    
  return(list(a = error, b = zAR))
  }
  
  # Access errors and AR1 process values 
  errors <- AR1(10000, 0.95)$a
  zAR <- AR1(10000, 0.95)$b
  
  # Also do it for tauchen and rouwenhorst
  # I will build a function that simulates using both methods
  
  discrete_simulation <- function(N, periods, method, rho) {
    
  # Simulate the errors using the continuous AR1 process
  errors <- AR1(periods, rho)$a
  cdf_errors <- pnorm(errors, mean = 0, sd = sigma)
    
  # Initialize variables based on method
  if (method == "tauchen") {
  grid <- tauchen(N, rho)$a
  transition_matrix <- tauchen(N, rho)$b
  } else if (method == "rouwenhorst") {
  grid <- rouwenhorst(N, rho)$a
  transition_matrix <- rouwenhorst(N, rho)$b
  }
    
  # Convert the transition probability matrix into a conditional CDF matrix
  cdf_matrix <- matrix(0, N, N)
  for (i in 1:N) {
  for (j in 1:N) {
  cdf_matrix[i, j] <- sum(transition_matrix[i, seq(1, j, 1)])
  }
  }
    
  # Initialize vector to store simulated data
  simulated_data <- numeric(periods)
    
  # Find the closest point on the grid to the first error realization
  simulated_data[1] <- grid[which.min(abs(grid - errors[1]))]
    
  # Recursively find values based on CDF comparisons
  for (i in 2:periods) {
  current_theta <- simulated_data[i - 1]
  grid_index <- which(grid == current_theta)
  cdf_values <- cdf_matrix[grid_index, ]
  condition <- cdf_values > cdf_errors[i]
  next_index <- min(which(condition))
  simulated_data[i] <- grid[next_index]
  }
    
  return(simulated_data)
  }
  
  # Simulated data 
  Z_tauchen <- discrete_simulation(9, 10000, "tauchen", 0.95)
  Z_rouwenhorst <- discrete_simulation(9, 10000, "rouwenhorst", 0.95)
  
  # Generate data for the comparison plots 
  comparison_data <- data.frame(
  Period = seq(1,10000,1),
  AR1 = zAR,
  tauchen = Z_tauchen,
  rouwenhorst = Z_rouwenhorst
  )
  
  # First plot: Continuous vs. Tauchen
  plot1_t <- ggplot(comparison_data, aes(x = Period)) +
  geom_line(aes(y = AR1, color = "AR(1)"), linetype = "solid") +
  geom_line(aes(y = tauchen, color = "Tauchen"), linetype = "solid") +
  labs(x = "Period", y = "Simulated Processes", 
  title = "(A) Continuous vs. Tauchen (9 points)") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(title = NULL)) +
  scale_linetype_manual(values = c("AR(1)" = "solid", "Tauchen" = "solid"))+
  guides(linetype = "none")
  
  # Second plot: Continuous vs. Rouwenhorst
  plot1_r <-ggplot(comparison_data, aes(x = Period)) +
  geom_line(aes(y = AR1, color = "AR(1)"), linetype = "solid") +
  geom_line(aes(y = rouwenhorst, color = "Rouwenhorst"), linetype = "solid") +
  labs(x = "Period", y = "Simulated Processes", 
  title = "(B) Continuous vs. Rouwenhorst (9 points)") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(title = NULL)) +
  scale_linetype_manual(values = c("AR(1)" = "solid", "Rouwenhorst" = "solid"))+
  guides(linetype = "none")
  

  #Let's add more point to see what happens
  # Simulated data 
  Z_tauchen <- discrete_simulation(21, 10000, "tauchen", 0.95)
  Z_rouwenhorst <- discrete_simulation(21, 10000, "rouwenhorst", 0.95)
  #I added way more points (51) just so we can see how the fit reacts
  
  comparison_data <- data.frame(
  Period = seq(1,10000,1),
  AR1 = zAR,
  tauchen = Z_tauchen,
  rouwenhorst = Z_rouwenhorst
  )
  
  plot2_t <- ggplot(comparison_data, aes(x = Period)) +
  geom_line(aes(y = AR1, color = "AR(1)"), linetype = "solid") +
  geom_line(aes(y = tauchen, color = "Tauchen"), linetype = "solid") +
  labs(x = "Period", y = "Simulated Processes", 
  title = "(C) Continuous vs. Tauchen (21 points)") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(title = NULL)) +
  scale_linetype_manual(values = c("AR(1)" = "solid", "Tauchen" = "solid"))+
  guides(linetype = "none")
  
  
  plot2_r <- ggplot(comparison_data, aes(x = Period)) +
  geom_line(aes(y = AR1, color = "AR(1)"), linetype = "solid") +
  geom_line(aes(y = rouwenhorst, color = "Rouwenhorst"), linetype = "solid") +
  labs(x = "Period", y = "Simulated Processes", 
  title = "(D) Continuous vs. Rouwenhorst (21 points)") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(title = NULL)) +
  scale_linetype_manual(values = c("AR(1)" = "solid", "Rouwenhorst" = "solid"))
  
  grid.arrange(plot1_t,plot1_r,plot2_t,plot2_r, ncol=2)
  
  # 4. Estimate an AR(1) using the simulated data
  # For Tauchen we have: Z_tauchen
  # For Rouwenhorst: Z_rouwenhorst
  
  # To estimate an AR(1) in R we use the arima function 
  # The order argument specifies the order of the ARIMA model, 
  # where c(1, 0, 0) indicates an AR(1) model (1st order autoregressive, 
  # 0th order differencing, and 0th order moving average).
  
  # Fit an AR(1) model
  model_tauchen <- arima(Z_tauchen, order = c(1, 0, 0), include.mean=F)
  model_rouwenhorst <- arima(Z_rouwenhorst, order = c(1,0,0), include.mean=F)
  model_tauchen
  model_rouwenhorst
  
  stargazer(model_tauchen,summary=FALSE, rownames=FALSE)
  stargazer(model_rouwenhorst,summary=FALSE, rownames=FALSE)
  
  
  # 5. Repeat the exercises for ρ = 0.99
  # 5.1 Discretization using Tauchen's Method with 9 points
  grid <- tauchen(9, 0.99)$a
  transition_matrix <- tauchen(9, 0.99)$b
  stargazer(grid,summary=FALSE, rownames=FALSE)
  stargazer(transition_matrix, summary=FALSE, rownames=FALSE)
  
  # 5.2 Discretization using Rouwenhorst's Method with 9 points
  grid <- rouwenhorst(9, 0.99)$a
  transition_matrix <- rouwenhorst(9, 0.99)$b
  stargazer(grid,summary=FALSE, rownames=FALSE)
  stargazer(transition_matrix, summary=FALSE, rownames=FALSE)
  
  # 5.3 Continuous process for 10000 periods and simulated data
  errors <- AR1(10000, 0.99)$a
  zAR <- AR1(10000, 0.99)$b
  Z_tauchen <- discrete_simulation(9, 10000, "tauchen", 0.99)
  Z_rouwenhorst <- discrete_simulation(9, 10000, "rouwenhorst", 0.99)
  
  # Data for the comparison plots 
  comparison_data <- data.frame(
  Period = seq(1,10000,1),
  AR1 = zAR,
  tauchen = Z_tauchen,
  rouwenhorst = Z_rouwenhorst
  )
  
  # Continuous vs. Tauchen
  plot1_t <- ggplot(comparison_data, aes(x = Period)) +
  geom_line(aes(y = AR1, color = "AR(1)"), linetype = "solid") +
  geom_line(aes(y = tauchen, color = "Tauchen"), linetype = "solid") +
  labs(x = "Period", y = "Simulated Processes", 
  title = "(A) Continuous vs. Tauchen (9 points)") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(title = NULL)) +
  scale_linetype_manual(values = c("AR(1)" = "solid", "Tauchen" = "solid"))
  
  
  # Continuous vs. Rouwenhorst
  plot1_r <-ggplot(comparison_data, aes(x = Period)) +
  geom_line(aes(y = AR1, color = "AR(1)"), linetype = "solid") +
  geom_line(aes(y = rouwenhorst, color = "Rouwenhorst"), linetype = "solid") +
  labs(x = "Period", y = "Simulated Processes", 
  title = "(B) Continuous vs. Rouwenhorst (9 points)") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(title = NULL)) +
  scale_linetype_manual(values = c("AR(1)" = "solid", "Rouwenhorst" = "solid"))
  
  
  #Let's add more point to see what happens
  # Simulated data 
  Z_tauchen <- discrete_simulation(21, 10000, "tauchen", 0.99)
  Z_rouwenhorst <- discrete_simulation(21, 10000, "rouwenhorst", 0.99)
  #I added way more points (51) just so we can see how the fit reacts
  
  comparison_data <- data.frame(
  Period = seq(1,10000,1),
  AR1 = zAR,
  tauchen = Z_tauchen,
  rouwenhorst = Z_rouwenhorst
  )
  
  plot2_t <- ggplot(comparison_data, aes(x = Period)) +
  geom_line(aes(y = AR1, color = "AR(1)"), linetype = "solid") +
  geom_line(aes(y = tauchen, color = "Tauchen"), linetype = "solid") +
  labs(x = "Period", y = "Simulated Processes",
  title = "(C) Continuous vs. Rouwenhorst (21 points)") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(title = NULL)) +
  scale_linetype_manual(values = c("AR(1)" = "solid", "Tauchen" = "solid"))
  
  
  plot2_r <- ggplot(comparison_data, aes(x = Period)) +
  geom_line(aes(y = AR1, color = "AR(1)"), linetype = "solid") +
  geom_line(aes(y = rouwenhorst, color = "Rouwenhorst"), linetype = "solid") +
  labs(x = "Period", y = "Simulated Processes", 
  title = "(D) Continuous vs. Rouwenhorst (21 points)") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(title = NULL)) +
  scale_linetype_manual(values = c("AR(1)" = "solid", "Rouwenhorst" = "solid"))
  
  grid.arrange(plot1_t,plot1_r,plot2_t,plot2_r, ncol=2)
  
  # 5.4 Fit an AR(1) model
  model_tauchen <- arima(Z_tauchen, order = c(1, 0, 0), include.mean=F)
  model_rouwenhorst <- arima(Z_rouwenhorst, order = c(1,0,0), include.mean=F)
  model_tauchen
  model_rouwenhorst
  
  stargazer(model_tauchen,summary=FALSE, rownames=FALSE)
  stargazer(model_rouwenhorst,summary=FALSE, rownames=FALSE)
  
  
  