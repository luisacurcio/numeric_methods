using Distributions, LinearAlgebra, Plots, PlotlyJS, Interpolations, Base.Threads, Roots, Dierckx


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

# I will use the same function as I did in the last PS
function tauchen(N_z, rho, sigma, m)
    zN = m * (sigma / sqrt(1 - rho^2)) 
    z1 = -zN 
    grid_points = LinRange(z1, zN, N_z)
    
    P = zeros(N_z, N_z)
    d = Normal(0, 1) 
    half_d = (grid_points[2] - grid_points[1]) / 2 
    
    for i in 1:N_z
        # probability of moving from i to the first state (left boundary)
        left_bound_prob = (grid_points[1] - rho * grid_points[i] + half_d ) / sigma
        P[i, 1] = cdf(d, left_bound_prob)
        
        # probability of moving from i to the last state (right boundary)
        right_bound_prob = (grid_points[N_z] - rho * grid_points[i] - half_d ) / sigma
        P[i, N_z] = 1 - cdf(d, right_bound_prob)
        
        # probabilities of moving from i to intermediate states
        for j in 2:(N_z - 1)
            lower_prob = (grid_points[j] - rho * grid_points[i] - half_d ) / sigma
            upper_prob = (grid_points[j] - rho * grid_points[i] + half_d ) / sigma
            P[i, j] = cdf(d, upper_prob) - cdf(d, lower_prob)
        end
    end
    
    return (grid_points, P)
end

# Results for z's grid and transition matrix
# In the first list we worked with a different stochastic process, 
# this time we have log zt = rho logzt-1 + et 
# So we need to take the exponential of the z_grid

result_tauchen = tauchen(N_z, rho, sigma, m)
z_grid = exp.(result_tauchen[1])
transition_matrix = result_tauchen[2]

# Now we build the utility function
utility = function(c, mu=mu)
    (c^(1-mu) - 1)/(1-mu)
end

# Function for k, to be used in the budget constraint (calculate consumption)
production = function(k; alpha=alpha)
    float(k)^alpha
end


# And the capital grid
# Grid: 500 points linearly spaced in the interval [0.75kss, 1.25kss]
k_ss= (alpha / (beta^(-1) + delta - 1))^(1 / (1 - alpha)) # steady state capital from question 2
k1 = 0.75*k_ss 
kN = 1.25*k_ss 
N_k = 500 
k_grid = LinRange(k1, kN, N_k)


#............................................................................................................................#
############################
###### EULER ERRORS ########
############################

# Access the Euler errors 
# I will build a function that receives the results from the vfi as a parameter
function euler_errors(vfi_results, N_k, k_grid; N_z=N_z, z_grid=z_grid, transition_matrix=transition_matrix, mu=mu, alpha=alpha, delta=delta, beta=beta)
    # Get vfi results from k and consumption
    k_new = vfi_results[2]
    consumption = vfi_results[4]
    
    # Initialize the error matrix
    euler_error = zeros(N_k, N_z)
    
    # Precompute constant terms and vectors: this is to improve code efficiency
    k_term = k_new .^ (alpha - 1)
    c_inv_mu = consumption .^ (-mu)
    
    # Iterate over each state and capital grid point
    for i in 1:N_z
        π = transition_matrix[i, :]
        for j in 1:N_k
            # Find the index of the current policy in the capital grid
            index = findfirst(isequal(k_new[j, i]), k_grid)  
            
            # Calculate the expected marginal utility of consumption
            expected_mgu = (c_inv_mu[index, :] .* (alpha * z_grid * (k_term[j,i]) .+ (1-delta)))' * π  
            
            # Compute the Euler equation error
            euler_error[j, i] = log10(abs(1 - ((beta .* expected_mgu) .^ (-1 / mu)) ./ consumption[j, i]))
        end
    end
    
    return euler_error
end

#............................................................................................................................#
############################
###### MONOTONICITY ########
############################

# Set tolerance level and v0
toler = 10^(-5)
max_iter = 10000
v0 = zeros(N_k,N_z)

# Initial guess for v0 based on the problem written in the problem set
for z ∈ 1:N_z, k ∈ 1:N_k
    v0[k,z] = utility(z_grid[z]* k_grid[k]^alpha -delta*k_grid[k],mu)
end


function vfi_monotonicity(;k_grid, z_grid, delta, beta, toler, N_k, N_z, transition_matrix, v, mu)
    # Pre-allocate matrices to avoid reallocating memory in each iteration
    v_new = similar(v)
    k_new = zeros(N_k, N_z)
    policy_index = zeros(Int, N_k, N_z)
    
    iteration = 0
    supnorm = 1.0
    
    while supnorm > toler && iteration < max_iter
        for i in 1:N_z
            π = transition_matrix[i, :] # select column to improve effiency
            index=1
            for j in 1:N_k
                # Compute all possible next-period capitals from current capital
              
                k_prime_index = index:N_k  # Apply monotonicity constraint
                k_prime = k_grid[k_prime_index]
                
                # Compute positive consumption for all feasible k_prime
                consumption = max.(0, z_grid[i] * k_grid[j]^alpha + (1 - delta) * k_grid[j] .- k_prime)
                
                # Compute utility for all feasible consumptions
                utility_vals = utility.(consumption, mu)
                
                # Future value function 
                expected_future_value = v[k_prime_index, :] * π
                
                # Total value for each choice of k_prime
                total_value = utility_vals + beta * expected_future_value
                
                # Find optimal k_prime and its value
                max_value, max_idx = findmax(total_value)
                v_new[j, i] = max_value
                policy_index[j, i] = k_prime_index[max_idx]
                index = argmax(total_value)
            end
        end
        
        # Check for convergence
        supnorm = maximum(abs.(v_new - v))
        v .= v_new  # Update value function in place
        
        iteration += 1
        println("iteration ", iteration)
        println("supnorm ",supnorm)
    end
    
    # Construct k_new from policy_index for clarity
    for i in 1:N_z, j in 1:N_k
        k_new[j, i] = k_grid[policy_index[j, i]]
    end
    
    # Calculate consumption policy based on k_new
    consumption_policy = z_grid' .* production.(k_grid) .+ (1 - delta) .* k_grid - k_new
    
    return v, k_new, iteration, consumption_policy
end


result_monotonicity = @time vfi_monotonicity(k_grid=k_grid, z_grid=z_grid, delta=delta, beta=beta, toler=toler, N_k=N_k, N_z=N_z, transition_matrix=transition_matrix, v=v0, mu=mu)
# My code converged after 39.6 seconds and 848 iterations!


# Create labels for each state
labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]

# Monotonicity plots
p1 = Plots.plot(k_grid, result_monotonicity[1], title = "(a) Value Function", legend = :bottomright, titlefontsize = 10, label=labels)
p2 = Plots.plot(k_grid, result_monotonicity[2], title = "(b) Policy Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid, result_monotonicity[4], title = "(c) Consumption Function", legend = :bottomright, titlefontsize = 10, label = labels)
p4 = Plots.plot(k_grid, euler_errors(result_monotonicity, N_k, k_grid), title = "(d) Euler Equation Errors", label = labels, titlefontsize = 10, legend = false)

# Combine plots into a 2x2 grid layout
combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "monotonicity_results.png")

#............................................................................................................................#
############################
####### CONCAVITY ##########
############################

# We we'll compare each element of the value function with its predecessor 
# Our goal is to find a turning point where the value starts decreasing,
# indicating the point of maximum value (assuming the value function is concave)

# Set tolerance level and v0
toler = 10^(-5)
max_iter = 10000
v0 = zeros(N_k,N_z)

# Initial guess for v0 based on the problem written in the problem set
for z ∈ 1:N_z, k ∈ 1:N_k
    v0[k,z] = utility(z_grid[z]* k_grid[k]^alpha -delta*k_grid[k],mu)
end

vfi_concavity = function(k_grid=k_grid, z_grid=z_grid, delta=delta, beta=beta, toler=toler, N_k=N_k, N_z=N_z, transition_matrix=transition_matrix, v=v0, mu=mu)
    v_new = similar(v) 
    k_new = zeros(N_k, N_z)
    policy_index = zeros(Int, N_k, N_z)
    
    iteration = 0
    supnorm = 1.0
    
    while supnorm > toler && iteration < max_iter
        for i in 1:N_z
        π = transition_matrix[i, :] # select column to improve effiency
            for j in 1:N_k
            
                # Repeat previous steps 
                consumption = max.(0, z_grid[i] * k_grid[j]^alpha + (1 - delta) * k_grid[j] .- k_grid[1:N_k])
                utility_vals = utility.(consumption, mu)
                expected_future_value = v[1:N_k, :] * π
                total_value = utility_vals + beta * expected_future_value
            
                # Initialize variables to store the results of optimal choice
                optimal_value = -Inf
                optimal_index = j
                
                # Exploit concavity: Search for the peak in total_value
                for k in 2:length(total_value)
                    # If total_value is bigger than the previous, we will use total_value and continue searching    
                    if total_value[k] > total_value[k-1]
                        optimal_value = total_value[k]
                        optimal_index = k
                    # If total_value is smaller than the previous, then the optimal will be the previous
                    else
                       optimal_value = total_value[k-1]
                       optimal_index = k-1
                       #we've reached the optimal
                       break
                    end
                end
                
                # Update the new value function and policy index with the optimal choice
                v_new[j, i] = optimal_value
                policy_index[j, i] = optimal_index
            end
        end    
        # Check for convergence
        supnorm = maximum(abs.(v_new - v))
        v .= v_new  # Update value function in place
 
        iteration += 1
        println("Iteration: ", iteration, " Supnorm: ", supnorm)
    end

    # Construct k_new from policy_index for clarity
    for i in 1:N_z, j in 1:N_k
        k_new[j, i] = k_grid[policy_index[j, i]]
    end
    
    # Calculate consumption policy based on k_new
    consumption_policy = z_grid' .* production.(k_grid) .+ (1 - delta) .* k_grid - k_new
    
    return v, k_new, iteration, consumption_policy
end

result_concavity = @time vfi_concavity()
# My code converged after 113.16 second and 848 iterations!

labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]

# Create individual plots
p1 = Plots.plot(k_grid, result_concavity[1], title = "(a) Value Function", legend = :bottomright, label = labels, titlefontsize = 10)
p2 = Plots.plot(k_grid, result_concavity[2], title = "(b) Policy Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid, result_concavity[4], title = "(c) Consumption Function", titlefontsize = 10, label = labels, legend = :bottomright)
p4 = Plots.plot(k_grid, euler_errors(result_concavity, N_k, k_grid), title = "(d) Euler Equation Errors", titlefontsize = 10, legend = false)

# Combine plots into a 2x2 grid layout without a global legend setting
combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "concavity_results.png")

#............................................................................................................................#
############################
# CONCAVITY + MONOTONICITY #
############################

# Set tolerance level and v0
toler = 10^(-5)
max_iter = 10000
v0 = zeros(N_k,N_z)

# Initial guess for v0 based on the problem written in the problem set
for z ∈ 1:N_z, k ∈ 1:N_k
    v0[k,z] = utility(z_grid[z]* k_grid[k]^alpha -delta*k_grid[k],mu)
end


function vfi_monotonicity_concavity(k_grid, N_k, v; z_grid=z_grid, delta=delta, beta=beta, toler=toler, N_z=N_z, transition_matrix=transition_matrix, mu=mu)
    # Pre-allocate matrices to avoid reallocating memory in each iteration
    v_new = similar(v)
    k_new = zeros(N_k, N_z)
    policy_index = zeros(Int, N_k, N_z)
    
    iteration = 0
    supnorm = 1.0
    
    while supnorm > toler && iteration < max_iter
        for i in 1:N_z
            π = transition_matrix[i, :] # select column to improve effiency
            index = 1

            for j in 1:N_k
                k_prime_index = index:N_k  # Apply monotonicity constraint
                k_prime = k_grid[k_prime_index]


                consumption = max.(0, z_grid[i] * k_grid[j]^alpha + (1 - delta) * k_grid[j] .- k_prime)
                utility_vals = utility.(consumption, mu)
                expected_future_value = v[k_prime_index, :] * π 
                total_value = utility_vals + beta * expected_future_value

                # Explore concavity
                optimal_value = -Inf
                optimal_index = j

                for k in k_prime_index
                    total_index = k - index + 1  # Ajustar índice para total_value
                    if total_value[total_index] > optimal_value
                        optimal_value = total_value[total_index] 
                        optimal_index = k  # Aqui, mantemos o índice relativo ao k_grid
                    else
                        break  # Saímos do loop se não encontrarmos um valor maior
                    end
                end

                v_new[j, i] = optimal_value
                policy_index[j, i] = optimal_index  
                index = optimal_index  
            end
        end
        
        # Check for convergence
        supnorm = maximum(abs.(v_new - v))
        v .= v_new  # Update value function in place
        
        iteration += 1
        println("iteration ", iteration)
        println("supnorm ",supnorm)
    end
    
    # Construct k_new from policy_index for clarity
    for i in 1:N_z, j in 1:N_k
        k_new[j, i] = k_grid[policy_index[j, i]]
    end
    
    # Calculate consumption policy based on k_new
    consumption_policy = z_grid' .* production.(k_grid) .+ (1 - delta) .* k_grid - k_new
    
    return v, k_new, iteration, consumption_policy
end

result_monotonicity_concavity = @time vfi_monotonicity_concavity(k_grid, N_k, v0)
# My code converged after 18.19 seconds and 771 iterations!

labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]

# Concavity plots
p1 = Plots.plot(k_grid, result_monotonicity_concavity[1], title = "(a) Value Function", legend = :bottomright, label = labels, titlefontsize = 10)
p2 = Plots.plot(k_grid, result_monotonicity_concavity[2], title = "(b) Policy Function", legend = false, titlefontsize = 10)
p3 = Plots.plot(k_grid, result_monotonicity_concavity[4], title = "(c) Consumption Function", legend = :bottomright, titlefontsize = 10, label = labels)
p4 = Plots.plot(k_grid,euler_errors(result_monotonicity_concavity, N_k, k_grid), title = "(d) Euler Equation Errors", titlefontsize = 10, legend = false)

# Combine plots into a 2x2 grid layout
combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "monotonicity_concavity_results.png")


#............................................................................................................................#
###########################
######## Question 4 #######
###########################

# Set tolerance level and v0
toler = 10^(-5)
max_iter = 10000
v0 = zeros(N_k,N_z)

# Initial guess for v0 based on the problem written in the problem set
for z ∈ 1:N_z, k ∈ 1:N_k
    v0[k,z] = utility(z_grid[z]* k_grid[k]^alpha -delta*k_grid[k],mu)
end

# This item asks us to use the accelerator: 
# only perform the maximization in 10% of the iterations

function vfi_monotonicity_concavity_accelerator(k_grid=k_grid, z_grid=z_grid, delta=delta, beta=beta, toler=toler, N_k=N_k, N_z=N_z, transition_matrix=transition_matrix, v=v0, mu=mu)
    # Pre-allocate matrices to avoid reallocating memory in each iteration
    v_new = similar(v)
    k_new = zeros(N_k, N_z)
    policy_index = zeros(Int, N_k, N_z)
    
    iteration = 0
    supnorm = 1
    
    while supnorm > toler && iteration < max_iter
        for i in 1:N_z
            π = transition_matrix[i, :] # select column to improve effiency
            index=1
            for j in 1:N_k
            
                    k_prime_index = index:N_k  # Apply monotonicity constraint
                    k_prime = k_grid[k_prime_index]

                    consumption = max.(0, z_grid[i] * k_grid[j]^alpha + (1 - delta) * k_grid[j] .- k_prime)
                    utility_vals = utility.(consumption, mu)
                    expected_future_value = v[k_prime_index, :] * π
                    total_value = utility_vals + beta * expected_future_value
            
                if iteration%10 == 0 # In every ten iterations...

                    # Find optimal k_prime and its value through maximization
                    max_value, max_idx = findmax(total_value)
                    v_new[j, i] = max_value
                    policy_index[j, i] = k_prime_index[max_idx]
                    index = argmax(total_value) 

                else

                    # Explore concavity
                    optimal_value = -Inf
                    optimal_index = j
                   
                    for k in k_prime_index
                        total_index = k - index + 1  # Ajustar índice para total_value
                        if total_value[total_index] > optimal_value
                            optimal_value = total_value[total_index] 
                            optimal_index = k  # The right k_grid index
                        else
                            break  # We've reached the optimum
                        end
                    end
       
                    v_new[j, i] = optimal_value
                    policy_index[j, i] = optimal_index  
                    index = optimal_index 
     
                end         
            end
        end
        
        # Check for convergence
        supnorm = maximum(abs.(v_new - v))
        v .= v_new  # Update value function in place
        
        iteration += 1
        println("iteration ", iteration)
        println("supnorm ",supnorm)
    end
    
    # Construct k_new from policy_index for clarity
    for i in 1:N_z, j in 1:N_k
        k_new[j, i] = k_grid[policy_index[j, i]]
    end
    
    # Calculate consumption policy based on k_new
    consumption_policy = z_grid' .* production.(k_grid) .+ (1 - delta) .* k_grid - k_new
    
    return v, k_new, iteration, consumption_policy
end

result_monotonicity_concavity_accelerator = @time vfi_monotonicity_concavity_accelerator()
# My code converged after 24.9 seconds and 848 iterations!

# Create labels for each state
labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]

# Plot results
p1 = Plots.plot(k_grid, result_monotonicity_concavity_accelerator[1], title = "(a) Value Function", legend = :bottomright, titlefontsize = 10, label = labels)
p2 = Plots.plot(k_grid, result_monotonicity_concavity_accelerator[2], title = "(b) Policy Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid, result_monotonicity_concavity_accelerator[4], title = "(c) Consumption Function", legend = :bottomright, titlefontsize = 10, label = labels)
p4 = Plots.plot(k_grid,euler_errors(result_monotonicity_concavity_accelerator, N_k, k_grid), title = "(d) Euler Equation Errors", titlefontsize = 10, legend = false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "monotonicity_concavity_accelerator_results.png")

#............................................................................................................................#
###########################
######## Question 5 #######
###########################

# This item asks us to use the multigrid method
# 1. grid of 100 points,
# 2. 500 points
# 3. 5000 points 


# Set tolerance level and v0
toler = 10^(-5)
max_iter = 10000
v0 = zeros(N_k,N_z)

# Initial guess for v0 based on the problem written in the problem set
for z ∈ 1:N_z, k ∈ 1:N_k
    v0[k,z] = utility(z_grid[z]* k_grid[k]^alpha -delta*k_grid[k],mu)
end

# I will do the multigrid using the monotonicity concavity function I created
# We will build k_grids inside the multigrid function, so k_grid won't be a parameter anymore
vfi_monotonicity_concavity_multigrid = function(z_grid=z_grid, delta=delta, beta=beta, toler=toler, N_k=N_k, N_z=N_z, transition_matrix=transition_matrix, v=v0, mu=mu)
    
    # First build the grids
    k_ss = (alpha / (beta^(-1) + delta - 1))^(1 / (1 - alpha)) 
    k1 = 0.75*k_ss 
    kN = 1.25*k_ss 
    N_k1 = 100
    N_k2 = 500
    N_k3 = 5000
    k_grid1 = LinRange(k1, kN, N_k1)
    k_grid2 = LinRange(k1, kN, N_k2)
    k_grid3 = LinRange(k1, kN, N_k3)

    v01 = zeros(N_k1, N_z)
    # Build the initial guess for the first grid
    for z ∈ 1:N_z, k ∈ 1:N_k1
        v01[k,z] = utility(z_grid[z]* k_grid1[k]^alpha - delta * k_grid1[k], mu)
    end

    # Apply the monotonicity + concavity vfi
    result_monotonicity_concavity1 = vfi_monotonicity_concavity(k_grid1, N_k1, v01)
    v_1 = result_monotonicity_concavity1[1]
    index_1 = result_monotonicity_concavity1[3]

    #  To use the previous solution as a new initial guess for the next step we need to interpolate
    v02 = zeros(N_k2, N_z)
    # I will use parallel interporlation to improve efficiency
    @threads for z in 1:N_z
        y = v_1[:, z]
        x = k_grid1
        interpolation = LinearInterpolation(x, y, extrapolation_bc=Line())
        v02[:, z] = [interpolation(k) for k in k_grid2]
    end

    # Apply monotonicity + concavity to the second grid, using the interpolated v_1 (v02)
    result_monotonicity_concavity2 = vfi_monotonicity_concavity(k_grid2, N_k2, v02)
    v_2 = result_monotonicity_concavity2[1]
    index_2 = result_monotonicity_concavity2[3]

    v03 = zeros(N_k3, N_z)
    @threads for z in 1:N_z
        y = v_2[:, z]
        x = k_grid2
        interpolation = LinearInterpolation(x, y, extrapolation_bc=Line())
        v03[:, z] = [interpolation(k) for k in k_grid3]
    end

    result_monotonicity_concavity3 = vfi_monotonicity_concavity(k_grid3, N_k3, v03)

    return result_monotonicity_concavity3, result_monotonicity_concavity2, result_monotonicity_concavity1, index_1, index_2
end

result_monotonicity_concavity_multigrid = @time vfi_monotonicity_concavity_multigrid()
# Converged after 94.67 seconds
result_monotonicity_concavity_multigrid[4]
result_monotonicity_concavity_multigrid[5]


# Now for the plots we need
N_k2 = 500
N_k3 = 5000
N_k1 = 100
k_grid1 = LinRange(k1, kN, N_k1)
k_grid2 = LinRange(k1, kN, N_k2)
k_grid3 = LinRange(k1, kN, N_k3)

# Create labels for each state
labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]
result_monotonicity_concavity_multigrid[1][3]

# Value Function Plot: 5000 grid points
p1 = Plots.plot(k_grid3, result_monotonicity_concavity_multigrid[1][1], title = "(a) Value Function", legend = :bottomright, titlefontsize = 10, label = labels)
p2 = Plots.plot(k_grid3, result_monotonicity_concavity_multigrid[1][2], title = "(b) Policy Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid3, result_monotonicity_concavity_multigrid[1][4], title = "(c) Consumption Function", legend = :bottomright, titlefontsize = 10, label = labels)
p4 = Plots.plot(k_grid3, euler_errors(result_monotonicity_concavity_multigrid[1], N_k3, k_grid3), title = "(d) Euler Equation Errors", titlefontsize = 10, legend = false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "monotonicity_concavity_multigrid_5000_results.png")

# Value Function Plot: 500 grid points
p1 = Plots.plot(k_grid2, result_monotonicity_concavity_multigrid[2][1], title = "(a) Value Function", legend = :bottomright, titlefontsize = 10, label = labels)
p2 = Plots.plot(k_grid2, result_monotonicity_concavity_multigrid[2][2], title = "(b) Policy Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid2, result_monotonicity_concavity_multigrid[2][4], title = "(c) Consumption Function", legend = :bottomright, titlefontsize = 10, label = labels)
p4 = Plots.plot(k_grid2, euler_errors(result_monotonicity_concavity_multigrid[2], N_k2, k_grid2), title = "(d) Euler Equation Errors", titlefontsize = 10, legend = false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "monotonicity_concavity_multigrid_500_results.png")

# Value Function Plot: 100 grid points
p1 = Plots.plot(k_grid1, result_monotonicity_concavity_multigrid[3][1], title = "(a) Value Function", legend = :bottomright, titlefontsize = 10, label = labels)
p2 = Plots.plot(k_grid1, result_monotonicity_concavity_multigrid[3][2], title = "(b) Policy Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid1, result_monotonicity_concavity_multigrid[3][4], title = "(c) Consumption Function", legend = :bottomright, titlefontsize = 10, label = labels)
p4 = Plots.plot(k_grid1, euler_errors(result_monotonicity_concavity_multigrid[3], N_k1, k_grid1), title = "(d) Euler Equation Errors", titlefontsize = 10, legend = false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "monotonicity_concavity_multigrid_100_results.png")


#............................................................................................................................#
###########################
######## Question 6 #######
###########################


# This question asks us to use the Endogenous Grid Method
# First let's set our first guess for c and v
c_initial = zeros(N_k,N_z)
v0 = zeros(N_k,N_z)

for z ∈ 1:N_z, k ∈ 1:N_k
    c_initial[k,z] = z_grid[z]* k_grid[k]^alpha -delta*k_grid[k]
    v0[k,z] = utility(c_initial[k,z])
end


# Set tolerance level
toler = 10^(-5)

# Main EGM function with embedded FOC calculation and grid point finding
function vfiEGM(k_grid, z_grid, delta, alpha, beta, toler, transition_matrix, c, mu, v)
    N_k, N_z = length(k_grid), length(z_grid)

    k_new = zeros(N_k, N_z)  
    k_prime_new = zeros(N_k, N_z)
    c_new = zeros(N_k,N_z)
    v_new = zeros(N_k,N_z)
    v = v0

    iteration, supnorm  = 0, 1
    
    while supnorm > toler
        for i in 1:N_z
            π = transition_matrix[i,:]
            for j in 1:N_k
                c_previous = c[j,:]

                FOC = k -> begin
                expectation = (c_previous.^(-mu) .* (alpha * z_grid[i] * k^(alpha - 1) + (1 - delta)))' * π
                (1 - delta) * k - k_grid[j] - (beta * expectation)^(-1/mu) + z_grid[i] * k^alpha
                end

                # Update endogenous capital levels using root finding
                k_new[j, i] = fzero(FOC, k_grid[j])
            end
        end

        # Interpolation
        for z in 1:N_z, k in 1:N_k
            G = LinearInterpolation(k_new[:,z], k_grid, extrapolation_bc=Line())   
            k_prime_new[:,z] = G.(k_grid)                
        end
        

        for i in 1:N_z
            π = transition_matrix[i, :]
            for j in 1:N_k
                c_new[j,i] = z_grid[i] * k_grid[j]^alpha + (1 - delta) * k_grid[j] - k_prime_new[j,i]
                v_new[j, i] = utility(c[j,i], mu) + beta *  v[j, :]' * π
            end
        end     

        supnorm = maximum(abs.(c_new - c))
        c .= c_new  # Update initial consumption guess for next iteration
        v .= v_new
        iteration += 1
        println("interations: $iteration, supnorm: $supnorm")

    end


    return v, k_prime_new, iteration, c

end

# My code converged after 96.79 seconds and 855 iterations
result_egm = @time vfiEGM(k_grid, z_grid, delta, alpha, beta, toler, transition_matrix, c_initial, mu, v0)

# project the values of k_grid_prime onto the closest points available on the original capital grid k_grid
k_grid_prime = zeros(N_k, N_z)
k_projected = zeros(N_k, N_z)
for z ∈ 1:N_z
    for k ∈ 1:N_k
        k_grid_prime[k,z] =  z_grid[z] * k_grid[k]^alpha + (1 - delta) * k_grid[k] - result_egm[4][k,z]
        err = abs.(k_grid .- k_grid_prime[k,z])
        x = minimum(err)
        k_projected[k,z] = k_grid[err .== x][1]
    end
end

# I will make a quick update in my original Euler Error function to receive the new parameters
function euler_errors2(k_new, consumption_matrix; N_k=N_k, k_grid=k_grid, N_z=N_z, z_grid=z_grid, transition_matrix=transition_matrix, mu=mu, alpha=alpha, delta=delta, beta=beta)
    # Initialize the error matrix
    euler_error = zeros(N_k, N_z)
    
    # Precompute constant terms and vectors: this is to improve code efficiency
    k_term = k_new .^ (alpha - 1)
    c_inv_mu = consumption_matrix .^ (-mu)
    
    # Iterate over each state and capital grid point
    for i in 1:N_z
        π = transition_matrix[i, :]
        for j in 1:N_k
            # Find the index of the current policy in the capital grid
            index = findfirst(isequal(k_new[j, i]), k_grid)  
            
            # Calculate the expected marginal utility of consumption
            expected_mgu = (c_inv_mu[index, :] .* (alpha * z_grid * (k_term[j,i]) .+ (1-delta)))' * π  
            
            # Compute the Euler equation error
            euler_error[j, i] = log10(abs(1 - ((beta .* expected_mgu) .^ (-1 / mu)) ./ consumption_matrix[j, i]))
        end
    end
    
    return euler_error
end

labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]

p1 = Plots.plot(k_grid, result_egm[1], title = "(a) Value Function", legend =  :bottomright,label = labels, titlefontsize = 10)
p2 = Plots.plot(k_grid, result_egm[2], title = "(b) Policy Function", legend =  false, titlefontsize = 10, label = labels)
p3 = Plots.plot(k_grid, result_egm[4], title = "(c) Consumption Function", legend = :bottomright, titlefontsize = 10, label = labels)
p4 = Plots.plot(k_grid, euler_errors2(k_projected, result_egm[4]), title = "(d) Euler Equation Errors", label = labels, titlefontsize = 10, legend = false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "egm_results.png")


