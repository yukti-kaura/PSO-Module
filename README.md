# PSO-Module (With/Without Penalty Function)
## PSO-Module with Penalty Function
### Function name <br>
**penalty_pso_function()** <br>

### Function File
**PENALTY_PSO_function.py**

### Function Usage
Has been shown in: <br>
**PENALTY_PSO_MODULE.ipynb**

### Function Parameters
Requires 7 function Parameters: <br>
**pso_function(parameter_values, bounds, n_particles, m_iterations, inertia, cognitive, social)** <br>
where, <br>
*parameter_values* = A list of all the **parameter values to be optimized** [List, length - Number of parameters] <br>
*bounds* = A list of limit boundaries (range) of the parameter values [List, length - Number of parameters] <br>
*n_particles* = Number of particles [Integer] <br>
*m_iterations* = Number of max iterations [Integer] <br>
*inertia* = Inertia weight [Float] <br>
*cognitive* = Cognitive weight [Float] <br>
*social* = Social weight [Float] <br>

### Function Necessities
To use the pso function you need to have 2 functions: <br>
*1. objective_function(para)* <br>
Takes *para* as function parameter, (expects a list same as *parameter_values*) <br>
Use this function to calculate the sum rate/other value<br>
Return the sum rate/other value [Float] <br>

*2. conditions(para)* <br>
Takes *para* as function parameter, (expects a list same as *parameter_values*) <br>
Use this function to **check the constraints, if not satisifed add a penalty value** <br>
Return penalty value [Float] <br>

### Note
Generally with PSO Algorithm, the penalty function differs from problem to problem. You can incorporate the penalty function in the *conditions(para)* function and return a penalty value. <br>
Penalty function used here: <br>
> **αᵢ = (constraint value)ᵢ - (constraint minimum/maximum value) <br>
Total penalty = Σ αᵢ ρᵢ** <br>
where, where *ρᵢ* = 1 is not satisfied and *ρᵢ* = 0 is satisfied. <br>

## PSO-Module without Penalty Function
### Function name <br>
**pso_function()** <br>

### Function Usage
Has been shown in: <br>
**PSO_MODULE.ipynb**

### Function File
**PSO_function.py**

### Function Parameters
Requires 7 function Parameters: <br>
**pso_function(parameter_values, bounds, n_particles, m_iterations, inertia, cognitive, social)** <br>
where, <br>
*parameter_values* = A list of all the **parameter values to be optimized** [List, length - Number of parameters] <br>
*bounds* = A list of limit boundaries (range) of the parameter values [List, length - Number of parameters] <br>
*n_particles* = Number of particles [Integer] <br>
*m_iterations* = Number of max iterations [Integer] <br>
*inertia* = Inertia weight [Float] <br>
*cognitive* = Cognitive weight [Float] <br>
*social* = Social weight [Float] <br>

### Function Necessities
To use the pso function you need to have 2 functions: <br>
*1. objective_function(para)* <br>
Takes *para* as function parameter, (expects a list same as *parameter_values*) <br>
Use this function to calculate the sum rate/other value<br>
Return the sum rate/other value [Float] <br>

*2. conditions(para)* <br>
Takes *para* as function parameter, (expects a list same as *parameter_values*) <br>
Use this function to check the constraints, <br>
Return true if satisfies else false [Boolean] <br>
