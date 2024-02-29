using Distributions 

###########################
###### PS2 - Methods ######
###########################

# Model calibration
beta = 0.987
mu = 2 
alpha = 1/3
delta = 0.012
rho = 0.95
sigma = 0.007
m = 3 #from PS1
N_z = 7

###########################
######## Question 3 #######
###########################
# Complete model with uncertantiy
# Value function iteration method

# I will use the same function as I did in the last list, with a small modification to include the mu parameter
function tauchen(N_z, rho, sigma, m, mu)
    zN = m * (sigma / sqrt(1 - rho^2)) + mu
    z1 = -zN + 2*mu
    grid_points = range(-z1, stop=zN, length=N_z)
    
    P = zeros(N_z, N_z)
    d = Normal(0, 1) 
    half_d = (grid_points[2] - grid_points[1]) / 2 
    
    for i in 1:N_z
        # probability of moving from i to the first state (left boundary)
        left_bound_prob = (grid_points[1] - rho * grid_points[i] + half_d - (1-rho)*mu) / sigma
        P[i, 1] = cdf(d, left_bound_prob)
        
        # probability of moving from i to the last state (right boundary)
        right_bound_prob = (grid_points[N_z] - rho * grid_points[i] - half_d - (1-rho)*mu) / sigma
        P[i, N_z] = 1 - cdf(d, right_bound_prob)
        
        # probabilities of moving from i to intermediate states
        for j in 2:(N_z - 1)
            lower_prob = (grid_points[j] - rho * grid_points[i] - half_d - (1-rho)*mu) / sigma
            upper_prob = (grid_points[j] - rho * grid_points[i] + half_d - (1-rho)*mu) / sigma
            P[i, j] = cdf(d, upper_prob) - cdf(d, lower_prob)
        end
    end
    
    return grid_points, P
end

# Results for z's grid and transition matrix
z_grid, transition_matrix = tauchen(N_z, rho, sigma, m, mu)
println(collect(z_grid))
println(transition_matrix)

# Now we build the utility function
utility = function(c, mu)
    (c^(1-mu) - 1)/(1-mu)
end

# Function for k, to be used in the budget constraint
# Cobb douglas
f = function(k; alpha=alpha)
    float(k)^alpha
end


# And the capital grid
# Grid: 500 points linearly spaced in the interval [0.75kss, 1.25kss]
k_ss= (alpha / (beta^(-1) + delta - 1))^(1 / (1 - alpha)) # steady state capital from question 2
k1 = 0.75*k_ss # lower bound
kN = 1.25*k_ss # upper bound
N_k = 500 # grid points
k_grid = range(k1, stop = kN, length = N_k)


############################
###### MONOTONICITY ########
############################

# Set tolerance level and v0
tol = 10^(-5)
v0 = zeros(N_k, N_z)

# Now build function exploring monotonicity
function VFI_monotonicity(;k_grid, z_grid, delta, beta, tol, N_k, N_z, transition_matrix, v0)
    # Initializing variables
    v_new = zeros(N_k, N_z) 
    k_new = zeros(N_k, N_z)
    iteration = 0
    diff = 1 #convergence criterion
    inertia = 1

    # Convergence check 
    while convergence > tol
        for i in 1:N_z # for each productivity state
            for j in 1:N_k # for each capital level k
                # Find the feasible k's based on monotonicity: 
                # (k' >= k), skipping calculation if none are found 
                feasible_indices = findall(x -> x >= k_grid[j], k_grid)
                if isempty(feasible_indices)
                    continue
                end
                
                # Calculate consumption for all feasible k's
                # From the planner's budget constraint: c = z f(k) + (1−δ) k - k'
                c_feasible = z_grid[i] * f(k_grid[j]) + (1 - delta) * k_grid[j] - k_grid[feasible_indices]
                positive_c_indices = findall(c -> c > 0, c_feasible)
                if isempty(positive_c_indices)
                    continue
                end

                # Filter for positive consumption and k's
                c_positive = c_feasible[positive_c_indices]
                feasible_k = k_grid[feasible_k_indices]
                k_positive = feasible_k[positive_c_indices]

                # Calculate value function for feasible consumption values
                # From the planner's problem: V (k, z) = max u(c) + βEzV (k', z')
                value_function = zeros(length(c_positive))
                for idx in 1:length(c_positive)
                value_function[idx] = utility(c_positive[idx], mu) + beta * dot(transition_matrix[i, :], v0[feasible_k_indices[positive_c_indices[idx]], :])
                end

                # Find the optimal value and corresponding policy
                [val, optimal_index] = findmax(value_function)
                v_new[j, i] = val
                k_new[j, i] = k_positive[optimal_index]
            end
        end

      # Update for convergence
      diff = maximum(abs.(v_new - v0))
      v0 = (1 - inertia) * v0 .+ inertia * v_new
      iteration += 1
    end

    # Calculate consumption based on the policy function
    c = [z_grid[i] * f(k_grid[j]) + (1 - delta) * k_grid[j] - k_new[j, i] for j in 1:N_k, i in 1:N_z]

    return (v = v0, policy = k_new, iterations = iteration, consumption = c)
end


result_monotonicity = @time VFI_monotonicity(k_grid=k_grid, z_grid=z_grid, delta=delta, beta=beta, tol=tol, N_k=N_k, N_z=N_z, transition_matrix=transition_matrix, v0=v0)

# Access the number of iterations
println("Number of iterations: ", result_monotonicity.iterations)


