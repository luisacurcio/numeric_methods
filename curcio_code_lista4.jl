using Distributions, LinearAlgebra, Plots, PlotlyJS, NLsolve

###########################
###### PS4 - Methods ######
###########################

# Model parameters
beta = 0.96
gamma = 1.0001
rho = 0.9
sigma = 0.01
Nz = 9 # we will be using 9 states this time

##########################################
############# A - Tauchen ################
##########################################

# Once again, we have our tauchen function
function tauchen(Nz, rho, sigma, m=3) # using 3 standard deviations for each side
    zN = m * (sigma / sqrt(1 - rho^2)) 
    z1 = -zN 
    grid_points = LinRange(z1, zN, Nz)
    
    P = zeros(Nz, Nz)
    d = Normal(0, 1) 
    half_d = (grid_points[2] - grid_points[1]) / 2 
    
    for i in 1:Nz
        # probability of moving from i to the first state (left boundary)
        left_bound_prob = (grid_points[1] - rho * grid_points[i] + half_d ) / sigma
        P[i, 1] = cdf(d, left_bound_prob)
        
        # probability of moving from i to the last state (right boundary)
        right_bound_prob = (grid_points[Nz] - rho * grid_points[i] - half_d ) / sigma
        P[i, Nz] = 1 - cdf(d, right_bound_prob)
        
        # probabilities of moving from i to intermediate states
        for j in 2:(Nz - 1)
            lower_prob = (grid_points[j] - rho * grid_points[i] - half_d ) / sigma
            upper_prob = (grid_points[j] - rho * grid_points[i] + half_d ) / sigma
            P[i, j] = cdf(d, upper_prob) - cdf(d, lower_prob)
        end
    end
    
    return (grid_points, P)
end

# Results
result_tauchen = tauchen(Nz, rho, sigma)
z_grid = exp.(result_tauchen[1])
transition_matrix = result_tauchen[2]

println(collect(z_grid)) # Notice we get the same results as in the R code
println(transition_matrix) # Also the same results as before


#######################################################
############# B - Solve individual problem ############
#######################################################

# This item asks us to discretize the asset space and solve the 
# individual's problem for each state variable

utility = function(c; gamma=gamma)
    consumption = (c^(1-gamma) - 1)/(1-gamma)
end

# Natural debt limit
# This commes form the natural debt limit in the slides given by:
# ϕ = -(s1w)/r
# but given that our constraint is: 
# c + a' = e^z + (1+r)a
r = 1/beta - 1 # using complete markets r
ϕ = - z_grid[1]/r
a1 = ϕ + 10^(-5) # lower bound for a
an = - ϕ  

# now we set the assets grid
Na = 500
a_grid = LinRange(a1, an, Na)
# I will use the same number of grid points we were using for k_grid in PS2 and PS3

# set v0 and tolerance
toler = 10^(-5)
max_iter = 10000
v0 = zeros(Na,Nz)

# here we consider a = a_prime
v0 = utility.(repeat(z_grid, 1, Na)' + r*repeat(a_grid, 1, Nz))/(1-beta)

# I will use the concavity + monotonicity method adapted to this problem
function vfi_monotonicity_concavity(r;k_grid=a_grid, N_k=Na, v=v0, z_grid=z_grid, beta=beta, toler=toler, N_z=Nz, transition_matrix=transition_matrix)
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


                consumption = max.(0, z_grid[i] + (1 + r) * k_grid[j] .- k_prime) # adapted
                utility_vals = utility.(consumption)
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
        #println("iteration ", iteration)
        #println("supnorm ",supnorm)
    end
    
    # Construct k_new from policy_index for clarity
    for i in 1:N_z, j in 1:N_k
        k_new[j, i] = k_grid[policy_index[j, i]]
    end
    
    # Calculate consumption policy based on k_new
    consumption_policy = z_grid' .+ (1 + r) .* k_grid - k_new
    
    return v, k_new, iteration, consumption_policy
end

result_B = @time vfi_monotonicity_concavity(r)

labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"]
p1 = Plots.plot(a_grid, result_B[1], title = "Value Function", legend = :bottomright, label = labels)
p2 = Plots.plot(a_grid, result_B[2], title = "Asset Policy Function", legend = :bottomright, label = labels)
p3 = Plots.plot(a_grid, result_B[4], title = "Consumption Policy Function", legend = :bottomright, label = labels)

combined_plot = Plots.plot(p1, p2, p3, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "item_b.png")


##############################################
########### C - Stationary π(z,a) ############
##############################################

# Find the stationary distribution π(z, a) and use it to compute 
# the aggregate savings in the economy. Find the equilibrium interest rate.


# This function computes the next iteration of the distribution π_t+1 based on
# the current distribution π_t and the transition matrix x.
# The matrix x encodes the probabilities of transitioning between states 
# and the policy function's indication of moving between asset levels (a to a').
# πt+1(a',z') is calculated by summing over all previous states and assets,
# weighted by the transition probabilities and the policy function's outcomes.
pi_t1 = function(pi_t,x) #Find the next (t+1) distribtution
    pi_t1=zeros(Na,Nz)
    for i in 1:Nz
        pi_t1[:,i] .= (sum(LinearAlgebra.dot(pi_t,x[j, i])) for j in 1:Na)
    end
    return pi_t1
end


# Function to compute the stationary distribution
function find_stationary_distribution(policy_function; a_grid=a_grid, transition_matrix=transition_matrix, tolerance = 1e-6, Na=Na, Nz=Nz)
    supnorm = 1
    iteration = 0

    # Initialize matrix probabilities based on policy and transition probabilities
    # Probabilities = P(z',z) J(g(a,z)=a')
    probabilities = [transition_matrix[:, i]' .* (policy_function .== a_grid[j]) for j in 1:Na, i in 1:Nz]

    # Initial distribution
    π_t = ones(Na, Nz) / (Na * Nz)
    π0 = pi_t1(π_t, probabilities)
    π_new = zeros(Na, Nz)

    # Iteratively find the stationary distribution
    while supnorm > tolerance
        π_new = pi_t1(π0, probabilities)
        supnorm = maximum(abs.(π_new - π0))
        π0 = π_new
        iteration += 1
    end

    return π_new
end

result_C = @time find_stationary_distribution(result_B[2])

# Before finding the equilibrium r, I build a function to compute aggregate demand
function compute_aggregate_demand(r; a_grid=a_grid)
    policy = vfi_monotonicity_concavity(r)[2]
    stationary_dist = find_stationary_distribution(policy)
    aggregate_demand = sum(a_grid .* stationary_dist)
    return aggregate_demand
end

function optimal_r(r0, rN; a_grid=a_grid, tolerance=1e-4)
    aggregate_demand_0 = compute_aggregate_demand(r0)
    aggregate_demand_N = compute_aggregate_demand(rN)

    # start bisection
    mean_r = (rN + r0) / 2
    mean_aggregate_demand = compute_aggregate_demand(mean_r)

        if mean_aggregate_demand > 0
            rN = mean_r
            mean_r = (rN + r0)/2
            aggregate_demand_N = mean_aggregate_demand
        else
            r0 = mean_r
            mean_r = (rN + r0)/2
            aggregate_demand_0 = mean_aggregate_demand
        end

    while abs(rN - r0) > tolerance
        mean_r = (rN + r0)/2
        mean_aggregate_demand = compute_aggregate_demand(mean_r)
       # println("r0: ", r0, ", rN: ", rN, ", mean_r: ", mean_r, ", Mean Aggregate Demand: ", mean_aggregate_demand)

        if mean_aggregate_demand > 0
            rN = mean_r
            aggregate_demand_N = mean_aggregate_demand
        else
            r0 = mean_r
            aggregate_demand_0 = mean_aggregate_demand
        end
    end

    return mean_r
end

# Initial guesses for r
r0 = 0
rN = 1/beta - 1

r_star_c = @time optimal_r(r0, rN)
println(r_star_c)

#Solving individuals problem for this interest rate:
result_c = vfi_monotonicity_concavity(r_star_c[1])
stationary_distribution_c = find_stationary_distribution(result_c[2]) 

p1 = Plots.plot(a_grid, result_c[1], title = "Value Function", legend = :bottomright, label = labels)
p2 = Plots.plot(a_grid, result_c[2], title = "Asset Policy Function", legend = :bottomright, label = labels)
p3 = Plots.plot(a_grid, result_c[4], title = "Consumption Policy Function", legend = :bottomright, label = labels)
p4 = Plots.plot(a_grid, sum(stationary_distribution_c, dims = 2), title = "Stationary Distribution",legend=false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "item_c.png")

##############################################
############### D - ρ = 0.97 #################
##############################################

# we redo the analysis for the new ρ
sigma = 0.01
rho = 0.97

result_tauchen = tauchen(Nz, rho, sigma)
z_grid = exp.(result_tauchen[1])
transition_matrix = result_tauchen[2]

ϕ = - z_grid[1]/r
a1 = ϕ + 10^(-5) 
an = - ϕ  

Na = 500
a_grid = LinRange(a1, an, Na)

v0 = utility.(repeat(z_grid, 1, Na)' + r * repeat(a_grid, 1, Nz))/(1-beta) 

r_star = @time optimal_r(r0, rN)
r_star_d = r_star
print(r_star_d)

result_d = vfi_monotonicity_concavity(r_star[1])
stationary_distribution_d = find_stationary_distribution(result_d[2]) 

p1 = Plots.plot(a_grid, result_d[1], title = "Value Function", legend = :bottomright, label = labels)
p2 = Plots.plot(a_grid, result_d[2], title = "Asset Policy Function", legend = :bottomright, label = labels)
p3 = Plots.plot(a_grid, result_d[4], title = "Consumption Policy Function", legend = :bottomright, label = labels)
p4 = Plots.plot(a_grid, sum(stationary_distribution_d, dims = 2), title = "Stationary Distribution",legend=false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "item_d.png")

##############################################
################# E - γ = 5 ##################
##############################################

# redo the analysis for gamma = 5
gamma = 5
rho = 0.9
sigma = 0.01

result_tauchen = tauchen(Nz, rho, sigma)
z_grid = exp.(result_tauchen[1])
transition_matrix = result_tauchen[2]

ϕ = - z_grid[1]/r
a1 = ϕ + 10^(-5) 
an = - ϕ  

Na = 500
a_grid = LinRange(a1, an, Na)

v0 = utility.(repeat(z_grid, 1, Na)' + r * repeat(a_grid, 1, Nz))/(1-beta) 

r_star_e = @time optimal_r(r0, rN)
result_e = vfi_monotonicity_concavity(r_star[1])
stationary_distribution_e = find_stationary_distribution(result_e[2]) 


p1 = Plots.plot(a_grid, result_e[1], title = "Value Function", legend = :bottomright, label = labels)
p2 = Plots.plot(a_grid, result_e[2], title = "Asset Policy Function", legend = :bottomright, label = labels)
p3 = Plots.plot(a_grid, result_e[4], title = "Consumption Policy Function", legend = :bottomright, label = labels)
p4 = Plots.plot(a_grid, sum(stationary_distribution_e, dims = 2), title = "Stationary Distribution",legend=false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "item_e.png")


##############################################
############### F - σ = 0.05 #################
##############################################

# redo the analysis for sigma = 0.05
sigma = 0.05
gamma = 1.0001

result_tauchen = tauchen(Nz, rho, sigma)
z_grid = exp.(result_tauchen[1])
transition_matrix = result_tauchen[2]

ϕ = - z_grid[1]/r
a1 = ϕ + 10^(-5) 
an = - ϕ  

Na = 500
a_grid = LinRange(a1, an, Na)

v0 = utility.(repeat(z_grid, 1, Na)' + r * repeat(a_grid, 1, Nz))/(1-beta) 

r_star_f = @time optimal_r(r0, rN)
print(r_star_f)
result_f = vfi_monotonicity_concavity(r_star_f[1])
stationary_distribution_f = find_stationary_distribution(result_f[2]) 


p1 = Plots.plot(a_grid, result_f[1], title = "Value Function", legend = :bottomright, label = labels)
p2 = Plots.plot(a_grid, result_f[2], title = "Asset Policy Function", legend = :bottomright, label = labels)
p3 = Plots.plot(a_grid, result_f[4], title = "Consumption Policy Function", legend = :bottomright, label = labels)
p4 = Plots.plot(a_grid, sum(stationary_distribution_f, dims = 2), title = "Stationary Distribution",legend=false)

combined_plot = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "item_f.png")


