using Distributions
using LinearAlgebra
using StatsBase  
using Random
using TimeSeries
using GR # for plotting
using DataFrames
using StateSpaceModels



###########################
###### PS1 - Methods ######
###########################

# In this code, i basically try to do the same step by step as I did in the R code

###########################
######## Question 1 #######
###########################
# Parameters
sigma = 0.007
rho = 0.95
m = 3  # Scaling parameter


function tauchen(N, rho, sigma, m)
    zN = m * (sigma / sqrt(1 - rho^2))
    grid_points = range(-zN, stop=zN, length=N)
    
    P = zeros(N, N)
    d = Normal(0, 1) 
    half_d = (grid_points[2] - grid_points[1]) / 2 
    
    for i in 1:N
        # probability of moving from i to the first state (left boundary)
        left_bound_prob = (grid_points[1] - rho * grid_points[i] + half_d) / sigma
        P[i, 1] = cdf(d, left_bound_prob)
        
        # probability of moving from i to the last state (right boundary)
        right_bound_prob = (grid_points[N] - rho * grid_points[i] - half_d) / sigma
        P[i, N] = 1 - cdf(d, right_bound_prob)
        
        # probabilities of moving from i to intermediate states
        for j in 2:(N - 1)
            lower_prob = (grid_points[j] - rho * grid_points[i] - half_d) / sigma
            upper_prob = (grid_points[j] - rho * grid_points[i] + half_d) / sigma
            P[i, j] = cdf(d, upper_prob) - cdf(d, lower_prob)
        end
    end
    
    return grid_points, P
end
 

# Applying Tauchen's Method
grid, transition_matrix = tauchen(9, rho, sigma,m)

println(collect(grid)) # Notice we get the same results as in the R code
println(transition_matrix) # Also the same results as before




###########################
######## Question 2 #######
############################ 
#Rouwenhorst Method


function rouwenhorst(N, rho, sigma)
    sigma_theta = sqrt(sigma^2 / (1 - rho^2))
    zN = sigma_theta * sqrt(N - 1)
    grid_points = range(-zN, stop=zN, length=N)
    
    p = (1 + rho) / 2
    P = [p 1-p; 1-p p]  # Base case for N=2
    
    for n in 3:N
        P_1 = zeros(n, n)
        P_1[1:n-1, 1:n-1] += p * P
        P_1[2:n, 1:n-1] += (1 - p) * P
        P_1[1:n-1, 2:n] += (1 - p) * P
        P_1[2:n, 2:n] += p * P
        P = P_1
    end
    
    # Normalization (ensure each row sums to 1)
    for row in 1:N
        P[row, :] ./= sum(P[row, :])
    end
    
    return grid_points, P
end


grid, transition_matrix = rouwenhorst(9, rho, sigma)
println(collect(grid)) 
println(transition_matrix)




###########################
######## Question 3 #######
###########################

# Define the AR1 simulation function
function simulate_AR1(periods, rho, sigma)
    Random.seed!(123)  # Set seed 
    errors = rand(Normal(0, sigma), periods)
    
    zAR = zeros(periods)
    zAR[1] = errors[1]  # Initialize with the first error
    
    for i in 2:periods
        zAR[i] = rho * zAR[i-1] + errors[i]
    end
    
    return errors, zAR
end

# simulation 
errors, zAR = simulate_AR1(10000, rho, sigma)


# discrete simulation function for Tauchen and Rouwenhorst methods
function discrete_simulation(N, periods, method, rho, sigma)
    errors, _ = simulate_AR1(periods, rho, sigma)
    cdf_errors = cdf.(Normal(0, sigma), errors)
    if method == "tauchen"
        grid, transition_matrix = tauchen(N, rho, sigma, m) 
    elseif method == "rouwenhorst"
        grid, transition_matrix = rouwenhorst(N, rho, sigma)
    end

    # Convert the transition probability matrix to a CDF matrix
    cdf_matrix = cumsum(transition_matrix, dims=2)
    simulated_data = zeros(periods)
    grid_array = collect(grid)  # grid to array

    # grid point closest to the first error term
    simulated_data[1] = grid_array[findmin(abs.(grid_array .- errors[1]))[2]]

    for i in 2:periods
        current_index = findfirst(x -> x == simulated_data[i-1], grid_array)
        cdf_values = cdf_matrix[current_index, :]
        next_index = findfirst(x -> x > cdf_errors[i], cdf_values)
        simulated_data[i] = grid_array[next_index]
    end

    return simulated_data
end


# Simulate discrete data
Z_tauchen = discrete_simulation(9, 10000, "tauchen", rho, sigma)
Z_rouwenhorst = discrete_simulation(9, 10000, "rouwenhorst", rho, sigma)
comparison_data = DataFrame(Period = 1:10000, AR1 = zAR, tauchen = Z_tauchen, rouwenhorst = Z_rouwenhorst)

# Update with more points
Z_tauchen_updated = discrete_simulation(21, 10000, "tauchen", rho, sigma)
Z_rouwenhorst_updated = discrete_simulation(21, 10000, "rouwenhorst", rho, sigma)
comparison_data_updated = DataFrame(Period = 1:10000, AR1 = zAR, tauchen = Z_tauchen_updated, rouwenhorst = Z_rouwenhorst_updated)

function plot_with_gr(x, y1, y2, title::String, label1::String, label2::String)
    GR.clearws()
    # Set the viewport and window based on the data range
    GR.setviewport(0.1, 0.9, 0.1, 0.9)
    GR.setwindow(minimum(x), maximum(x), minimum([y1; y2]), maximum([y1; y2]))
    
    # blue
    GR.setlinecolorind(4)
    GR.polyline(x, y1)

    # red
    GR.setlinecolorind(2)
    GR.polyline(x, y2)
    
    # labels and title
    GR.xlabel("Period")
    GR.ylabel("Value")
    GR.title(title)
    
    # add a legend
    GR.settextcolorind(4)
    GR.text(0.9 * maximum(x), 0.9 * maximum([y1; y2]), label1)
    GR.settextcolorind(2)
    GR.text(0.9 * maximum(x), 0.85 * maximum([y1; y2]), label2)
    
    # update the plot
    GR.updatews()
end
  

# Periods for plotting
periods = 1:10000

# Plotting
# blue line is the AR(1) and red line the discrete method
plot_with_gr(periods, comparison_data.AR1, comparison_data.tauchen, "(A) Continuous vs. Tauchen (9 points)", "AR(1)", "Tauchen")
plot_with_gr(periods, comparison_data.AR1, comparison_data.rouwenhorst, "(B) Continuous vs. Rouwenhorst (9 points)", "AR(1)", "Rouwenhorst")
plot_with_gr(periods, comparison_data_updated.AR1, comparison_data.tauchen, "(A) Continuous vs. Tauchen (21 points)", "AR(1)", "Tauchen")
plot_with_gr(periods, comparison_data_updated.AR1, comparison_data_updated.rouwenhorst, "(B) Continuous vs. Rouwenhorst (21 points)", "AR(1)", "Rouwenhorst")

# for some reason the Plots package is not working on my computer, so I had to turn to this other method


###########################
####### Question 4 ########
###########################

data_tauchen=DataFrame(tauchen=vec(Z_tauchen))
model_tauchen = SARIMA(data_tauchen.tauchen, order = (1,0,0))
fit!(model_tauchen)
model_tauchen.results

data_rouwenhorst=DataFrame(rouwenhorst=vec(Z_rouwenhorst))
model_rouwenhorst = SARIMA(data_rouwenhorst.rouwenhorst, order = (1,0,0))
fit!(model_rouwenhorst)
model_rouwenhorst.results



###########################
####### Question 5 ########
###########################
# New parameter
rho = 0.99

# Discretization using Tauchen's and Rouwenhorst's methods
grid_tauchen, transition_matrix_tauchen = tauchen(9, rho, sigma,m)
grid_rouwenhorst, transition_matrix_rouwenhorst = rouwenhorst(9, rho, sigma)

# simulate continuous process and discrete simulations
errors, zAR = simulate_AR1(10000, rho, sigma)
Z_tauchen = discrete_simulation(9, 10000, "tauchen", rho, sigma)
Z_rouwenhorst = discrete_simulation(9, 10000, "rouwenhorst", rho, sigma)
comparison_data = DataFrame(Period = 1:10000, AR1 = zAR, tauchen = Z_tauchen, rouwenhorst = Z_rouwenhorst)

# updated comparison data with more points
Z_tauchen = discrete_simulation(21, 10000, "tauchen", rho, sigma)
Z_rouwenhorst = discrete_simulation(21, 10000, "rouwenhorst", rho, sigma)
comparison_data_updated = DataFrame(Period = 1:10000, AR1 = zAR, tauchen = Z_tauchen, rouwenhorst = Z_rouwenhorst)

#plots 
plot_with_gr(periods, comparison_data.AR1, comparison_data.tauchen, "(A) Continuous vs. Tauchen (9 points)", "AR(1)", "Tauchen")
plot_with_gr(periods, comparison_data.AR1, comparison_data.rouwenhorst, "(B) Continuous vs. Rouwenhorst (9 points)", "AR(1)", "Rouwenhorst")
plot_with_gr(periods, comparison_data_updated.AR1, comparison_data.tauchen, "(A) Continuous vs. Tauchen (21 points)", "AR(1)", "Tauchen")
plot_with_gr(periods, comparison_data_updated.AR1, comparison_data_updated.rouwenhorst, "(B) Continuous vs. Rouwenhorst (21 points)", "AR(1)", "Rouwenhorst")

#estimation
data_tauchen=DataFrame(tauchen=vec(Z_tauchen))
model_tauchen = SARIMA(data_tauchen.tauchen, order = (1,0,0))
fit!(model_tauchen)
model_tauchen.results

data_rouwenhorst=DataFrame(rouwenhorst=vec(Z_rouwenhorst))
model_rouwenhorst = SARIMA(data_rouwenhorst.rouwenhorst, order = (1,0,0))
fit!(model_rouwenhorst)
model_rouwenhorst.results


