using Distributions, LinearAlgebra, Plots, PlotlyJS, NLsolve

###########################
###### PS3 - Methods ######
###########################

# Parameters
beta = 0.987
mu = 2
alpha = 1/3
delta = 0.012
rho = 0.95
sigma = 0.007
Nz = 7
m = 3

# Once again, we have our tauchen function
function tauchen(Nz, rho, sigma, m)
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
result_tauchen = tauchen(Nz, rho, sigma, m)
z_grid = exp.(result_tauchen[1])
transition_matrix = result_tauchen[2]

# Utility
utility = function(c, mu=mu)
    (c^(1-mu) - 1)/(1-mu)
end


# And the capital grid
# Grid: 500 points linearly spaced in the interval [0.75kss, 1.25kss]
k_ss= (alpha / (beta^(-1) + delta - 1))^(1 / (1 - alpha)) # steady state capital from question 2
k1 = 0.75*k_ss 
kN = 1.25*k_ss 
Nk = 500 
k_grid = LinRange(k1, kN, Nk)

###########################################
###### Q1 - Chebyshev + Collocation ######
###########################################


# Chebyshev polinomials take [−1,1]. So we need to transform k:
k_grid_cheby = function(k)
    k_grid_cheby = 2 .* (k .- k1)./(kN -  k1) .- 1
end
k_cheby = k_grid_cheby(k_grid) #new grid between -1 and 1
# has lenght (1,500)


# We will also need to map the roots into the k grid
cheby_k_grid = function(x)
    k_roots = k1 .+ ((x .+ 1) .* (kN - k1))./2
end # this function gets cheby vectors and map into k_grid
# returns a vector of same lenght as x

# sanity check
cheby_k_grid.(k_grid_cheby.(k_grid)) # returns to k_grid
k_grid

# Tm(x) = cos(mcosˆ-1(x))
cheby_polynomial = function(m,x)
    cos.(m * acos.(x))
end 
# this will give a vector of lenght(x)

# Roots: zi = -cos[((2i - 1)*pi)/2m]
# For a poly of order m, for each i, we get zi (the root)
roots_function = function(m)
    # m is the order of the Chebyshev polynomial
    # Returns a vector of size (m + 1) containing the roots
    roots = zeros(m + 1) # or d
    for i in 1:(m + 1)
        roots[i] = -cos(((2 * i - 1) * pi)/(2 * m))
    end
    return roots # vector of size m+1
end   

# for each state and capital, we generate the consumption for a set of gammas
c_hat_func = function(gamma, k, z)
    d = size(gamma,1) # size m+1
    c_hat = 0

    for i in 1:d
        c_hat += gamma[i, z] * cheby_polynomial(i-1, k)
    end

    c_hat # returns a vector of lenght(k) 
    # gamma[,z] is a vector of size m+1, which is the number of elements in the sum 
    # gamma is a matrix of size m+1 x 7
    # gamma[i,z] is size 1, cheby_polynomial returns a vector of lenght(k)
end


residual_function = function(gamma, z=z_grid)
    # z is the z_grid
    d = size(gamma, 1) # m+1 size
    #m = d-1
    c0 = zeros(d, Nz)
    c1 = zeros(d, Nz)
    residual = zeros(d, Nz)
    roots = roots_function(d) # vector of size m+1
    k = cheby_k_grid(roots) # roots are mapped into k_grid

    for i in 1:Nz
        for j in 1:d
        c0[j,i] = c_hat_func(gamma, roots[j], i) 

        # pick k0 ∈ X and evaluate  ĉ0 = ĉ(gamma,k0)
        k0 = k[j]

        # resource constraint: k1 = z*k0^α + (1-delta)k0 - ĉ0
        k1 = z_grid[i] * (k0^alpha) + (1 - delta) * k0 - c0[j,i] # for k0 in state z
        k1 = k_grid_cheby(k1)
        # k1 will be a float in cheby grid 

        # given k1 evaluate ĉ again to get ĉ1 = ĉ(gamma,k1)
        for z in 1:Nz
        c1[j,z] = c_hat_func(gamma, k1, z) 
        end
        # will be a vector of size j,Nz    

        # compute the residual function
        expectation = (c1[j,:]).^(-mu) .* (1 - delta .+ (z_grid) * alpha * cheby_k_grid(k1).^(alpha-1))
        residual[j,i] = ((c0[j,i]).^(-mu) .- beta * expectation' * transition_matrix[i,:])[1]  
        end
    end        
    return residual 
end



optimal_gamma = function(d; Nz=Nz)
    gamma = ones(2,Nz)
    for i in 2:d
        solution = nlsolve(residual_function, gamma) 
        gamma = solution.zero
        if i != d 
            gamma = [gamma; zeros(Nz)']        
        end
    end
    return gamma
end

optimal_gamma(6)


# Now we solve the model
function solve_chebyshev_model(d; z_grid=z_grid, k_grid=k_grid, Nk=Nk, Nz=Nz, delta=delta, beta=beta, mu=mu, transition_matrix=transition_matrix, alpha=alpha)
    # Initialize matrices
    euler_errors = zeros(Nk,Nz) 
    c1 = zeros(Nk,Nz)
    
    # Set parameters
    k_cheby = k_grid_cheby.(k_grid)
    gamma = optimal_gamma(d)

    # Compute the new capital and consumption for each state in the grid
    z_matrix = repeat(z_grid', Nk, 1)
    k_matrix = repeat(k_grid, 1, Nz) 

    # Apply c_hat_func to every point in the grid
    c_hat = reduce(hcat, [c_hat_func.((gamma,), k_cheby, i) for i in 1:Nz]) 

    # Compute new capital using the constraint
    k1 = z_matrix .* (k_matrix .^ alpha) + (1 - delta) .* k_matrix - c_hat
    k1 = k_grid_cheby.(k1) # map to cheby

    # Calculate new consumption and euler eq errors
    for i in 1:Nz
        for j in 1:Nk
            for z in 1:Nz
            c1[j,z] = c_hat_func(gamma, k1[j,i],z) 
            end

        # Now to find the euler errors    
        expectation = (c1[j,:]).^(-mu) .* (1 - delta .+ (z_grid) * alpha * cheby_k_grid(k1[j,i]).^(alpha-1))
        future_value = (beta * expectation' * transition_matrix[i,:])[1]   
        error = 1 - ((future_value .^ (-1/mu)) / c_hat[j,i]) 
        euler_errors[j, i] = log10(abs(error))
    end
end
    return (k1 = k1, c_hat = c_hat, euler = euler_errors)
end

collocation_result = @time solve_chebyshev_model(6)

labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]

# Collocation plots
p1 = Plots.plot(k_grid, collocation_result[1], title = "(a) Policy Function", legend = :bottomright, titlefontsize = 10, label=labels)
p2 = Plots.plot(k_grid, collocation_result[2], title = "(b) Consumption Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid, collocation_result[3], title = "(c) Euler Errors", legend = :bottomright, titlefontsize = 10, label = labels)

# Combine plots into a 2x2 grid layout
combined_plot = Plots.plot(p1, p2, p3, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "collocation_results.png")


###########################################
###### Q2 - Finite Element Analysis #######
###########################################

######### Collocation + FEA ##############

# Create collocation points between k1 and kN
num_points = 15
limit_points = zeros(num_points)
limit_points[1] = k1
limit_points[num_points] = kN

for i in 2:num_points-1
    index = Int(1 + floor(Nk / (num_points-1)) * (i - 1))
    limit_points[i] = k_grid[index]
end

# Define the basis function φ(i, k) for piecewise linear approximation
function psi_func(i, k; limit_points=limit_points, num_points=num_points)
    if i == 1
        return k < limit_points[i] ? 1 : (k >= limit_points[i] && k <= limit_points[i+1] ? (limit_points[i+1] - k) / (limit_points[i+1] - limit_points[i]) : 0)
    elseif i == num_points
        return k > limit_points[i] ? 1 : (k <= limit_points[i] && k >= limit_points[i-1] ? (k - limit_points[i-1]) / (limit_points[i] - limit_points[i-1]) : 0)
    else
        return k <= limit_points[i] && k >= limit_points[i-1] ? (k - limit_points[i-1]) / (limit_points[i] - limit_points[i-1]) :
               k >= limit_points[i] && k <= limit_points[i+1] ? (limit_points[i+1] - k) / (limit_points[i+1] - limit_points[i]) : 0
    end
end

# Compute the estimated consumption 
function c_hat_func_fe(A, k, z)
    return sum([A[i, z] * psi_func(i, k) for i in 1:num_points])
end

# Compute the residuals for the optimization problem
function residual_function_fe(A)
    k = limit_points
    k_new = zeros(num_points, Nz)
    c0 = zeros(num_points, Nz)
    c1 = zeros(num_points, Nz)
    residuals = zeros(num_points, Nz)

    # very similar to what we've done above
    for i in 1:Nz
        for j in 1:num_points
            c0[j,i] = c_hat_func_fe(A, k[j], i) 
            k0 = k[j]
    
            k1 = z_grid[i] * (k0^alpha) + (1 - delta) * k0 - c0[j,i] 

            for z in 1:Nz
            c1[j,z] = c_hat_func_fe(A, k1, z) 
            end
    
            expectation = (c1[j,:]).^(-mu) .* (1 - delta .+ (z_grid) * alpha * k1.^(alpha-1))
            residuals[j,i] = ((c0[j,i]).^(-mu) .- beta * expectation' * transition_matrix[i,:])[1]    
        end
    end

    return residuals
end


# Initial guess for coefficient matrix A using consumption intervals
function initial_guess(z_grid, k_grid, alpha, delta, num_points, Nz)
    z_matrix = repeat(z_grid', Nk)
    k_matrix = repeat(k_grid, 1, Nz)
    initial_consumption = z_matrix .* (k_matrix.^alpha) - delta * k_matrix
    consumption_range = LinRange(minimum(initial_consumption), maximum(initial_consumption), num_points * Nz)
    A_initial = reshape(consumption_range, Nz, num_points)'
    return A_initial
end

# Now we solve the model and compute the euler errors like before
function fea_collocation_model(A)

    # Initialize array for Euler equation errors and consumption
    euler_errors = zeros(Nk, Nz)
    c1 = zeros(Nk, Nz)

    # Solve for optimal coefficients A
    solution = nlsolve(residual_function_fe, A)
    optimal_A = solution.zero

    # Pre-compute repeated values for efficiency
    z_matrix = repeat(z_grid', Nk)
    k_matrix = repeat(k_grid, 1, Nz)

    # Apply c_hat_func_fe to every point in the grid
    c_hat = reduce(hcat, [c_hat_func_fe.((optimal_A,), k_grid, i) for i in 1:Nz]) 

    # Compute new capital using the constraint
    k1 = z_matrix .* (k_matrix .^ alpha) + (1 - delta) .* k_matrix - c_hat

    # Calculate Euler equation errors
    for i in 1:Nz
        for j in 1:Nk
            for z in 1:Nz
            c1[j,z] = c_hat_func_fe(optimal_A, k1[j,i], z)
            end 
   
            # Now to find the euler errors   
            expectation = (c1[j,:]).^(-mu) .* (1 - delta .+ z_grid * alpha .* k1[j,i]^(alpha-1))
            future_value = (beta * expectation' * transition_matrix[i,:])[1]   
            error = 1 - ((future_value .^ (-1/mu)) / c_hat[j,i]) 
            euler_errors[j, i] = log10(abs(error))
        end
    end

    return (euler_errors, c_hat, k1)
end

# Define parameters and call the optimization model
A_initial = initial_guess(z_grid, k_grid, alpha, delta, num_points, Nz)
fea_collocation_result = @time fea_collocation_model(A_initial)

labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]

# Collocation + FEA plots
p1 = Plots.plot(k_grid, fea_collocation_result[3], title = "(a) Policy Function", legend = :bottomright, titlefontsize = 10, label=labels)
p2 = Plots.plot(k_grid, fea_collocation_result[2], title = "(b) Consumption Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid, fea_collocation_result[1], title = "(c) Euler Errors", legend = :bottomright, titlefontsize = 10, label = labels)

# Combine plots into a 2x2 grid layout
combined_plot = Plots.plot(p1, p2, p3, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "fea_collocation_results.png")


######### Galerkin + FEA ##############

# Galerkin: takes the functions used in the approximation to impose moment conditions
# The method chooses a so that <R(.;a),ψ_i>=0
# R(k;a) = u'(c(k;a)) - beta u'(c(˜k';a))(1 + alpha ˜k'^(alpha-1) - delta)
# ˜k'= -c(k,a) + k^alpha + (1 - delta)k
# We adapt this to our problem

# We build a new residual function
function residual_function_fe_galerkin(A, k, z)
    c0 = c_hat_func_fe(A, k, z)
    k1 = - c0 + z_grid[z] * k^alpha + (1 - delta) * k
    c1 = zeros(Nz)

    for i in 1:Nz
        c1[i] = c_hat_func_fe(A, k1, i) 
    end
    
    expectation = (c1).^(-mu) .* (1 - delta .+ (z_grid) * alpha * k1.^(alpha-1))
    residuals = (c0.^(-mu) .- beta * expectation' * transition_matrix[z,:])[1]    

    return residuals
end


# Adjust the psi function to get upper and lower bounds
function psi_func_bounds(i, k; limit_points=limit_points, num_points=num_points)
    psi_lower_value = 0
    psi_upper_value = 0

    # Lower bound
    if i == 1
        if k < limit_points[i]
            psi_lower_value = 1
        elseif k <= limit_points[i] && k >= limit_points[i-1]
            psi_lower_value = (k - limit_points[i-1]) / (limit_points[i] - limit_points[i-1])
        end
    elseif i > 1 && i <= num_points
        if k <= limit_points[i] && k >= limit_points[i-1]
            psi_lower_value = (k - limit_points[i-1]) / (limit_points[i] - limit_points[i-1])
        end
    end

    # Upper bound
    if i == num_points
        if k > limit_points[i]
            psi_upper_value = 1
        end
    elseif i < num_points
        if k >= limit_points[i] && k <= limit_points[i+1]
            psi_upper_value = (limit_points[i+1] - k) / (limit_points[i+1] - limit_points[i])
        end
    end

    return (psi_lower_value, psi_upper_value)
end


function quadrature_function(A)
    # Generate Chebyshev roots for a polynomial of degree p
    roots = roots_function(num_points)
    
    # Initialize the matrix to store the integral results
    integral = zeros(num_points, Nz)

    for j in 1:Nz
        for i in 1:num_points
            a, b = i == 1 ? (limit_points[i], limit_points[i+1]) : (i == num_points ? (limit_points[i-1], limit_points[i]) : (limit_points[i-1], limit_points[i+1]))
            f = zeros(num_points)
            
            for n in 1:num_points
                k = ((roots[n] + 1) * (b - a)) / 2 + a
                f[n] = residual_function_fe_galerkin(A, k, j) * (i == 1 ? psi_func_bounds(i, k)[2] : (i == num_points ? psi_func_bounds(i, k)[1] : (psi_func_bounds(i, k)[1] + psi_func_bounds(i,k)[2]))) * sqrt(1 - roots[n]^2)
            end
            
            integral[i, j] = pi * (b - a) / (2num_points) * sum(f)
        end
    end

    return integral
end



# Now we solve the model and compute the euler errors: the function is very similar to the one we used before
function fea_garlekin_model(A)

    # Initialize array for Euler equation errors and consumption
    euler_errors = zeros(Nk, Nz)
    c1 = zeros(Nk, Nz)

    # Solve for optimal coefficients A
    solution = nlsolve(quadrature_function, A) # this is the only difference from the previous function
    optimal_A = solution.zero

    # Pre-compute repeated values for efficiency
    z_matrix = repeat(z_grid', Nk)
    k_matrix = repeat(k_grid, 1, Nz)

    # Apply c_hat_func_fe to every point in the grid
    c_hat = reduce(hcat, [c_hat_func_fe.((optimal_A,), k_grid, i) for i in 1:Nz]) 

    # Compute new capital using the constraint
    k1 = z_matrix .* (k_matrix .^ alpha) + (1 - delta) .* k_matrix - c_hat

    # Calculate Euler equation errors
    for i in 1:Nz
        for j in 1:Nk
            for z in 1:Nz
            c1[j,z] = c_hat_func_fe(optimal_A, k1[j,i], z)
            end 
   
            # Now to find the euler errors   
            expectation = (c1[j,:]).^(-mu) .* (1 - delta .+ z_grid * alpha .* k1[j,i]^(alpha-1))
            future_value = (beta * expectation' * transition_matrix[i,:])[1]   
            error = 1 - ((future_value .^ (-1/mu)) / c_hat[j,i]) 
            euler_errors[j, i] = log10(abs(error))
        end
    end

    return (euler_errors, c_hat, k1)
end

# Define parameters and call the optimization model
A_initial = initial_guess(z_grid, k_grid, alpha, delta, num_points, Nz) # the same as before
fea_garlekin_result = @time fea_garlekin_model(A_initial)

labels = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"]

# Garlekin + FEA plots
p1 = Plots.plot(k_grid, fea_garlekin_result[3], title = "(a) Policy Function", legend = :bottomright, titlefontsize = 10, label=labels)
p2 = Plots.plot(k_grid, fea_garlekin_result[2], title = "(b) Consumption Function", titlefontsize = 10, legend = false)
p3 = Plots.plot(k_grid, fea_garlekin_result[1], title = "(c) Euler Errors", legend = :bottomright, titlefontsize = 10, label = labels)

# Combine plots into a 2x2 grid layout
combined_plot = Plots.plot(p1, p2, p3, layout = (2, 2), size = (1200, 900))
Plots.savefig(combined_plot, "fea_garlekin_results.png")



