mutable struct ScaledLpProblem
    original_lp::LinearProgrammingProblem
    scaled_lp::LinearProgrammingProblem
    constraint_rescaling::Vector{Float64}
    variable_rescaling::Vector{Float64}
end

"""
Uses a modified Ruiz rescaling algorithm to rescale the constraint_matrix A, 
and returns the cumulative scaling vectors. More details of Ruiz rescaling algorithm can be
found at: http://www.numerical.rl.ac.uk/reports/drRAL2001034.pdf.

In the p=Inf case, both matrices approach having all row and column LInf norms
of M equal to 1 as the number of iterations goes to infinity. This convergence
is fast (linear).

In the p=2 case, the goal is all row L2 norms of A' equal to 1 
and all row L2 norms of A equal to sqrt(num_variables/num_constraints). 
Having a different goal for the row and col norms is required since the sum of squares of the
entries of the A matrix is the same when the sum is grouped by rows or grouped
by columns. In particular, all L2 norms of A must be
sqrt(num_variables/num_constraints) when all row L2 norm of A' equal to 1.

The Ruiz rescaling paper (link above) only analyzes convergence in the p < Inf
case when the matrix is square, and it does not preserve the symmetricity of
the matrix, and that is why we need to modify it for p=2 case.

# Arguments
- `problem::QuadraticProgrammingProblem`: the quadratic programming problem.
  This is modified to store the transformed problem.
- `num_iterations::Int64` the number of iterations to run Ruiz rescaling
  algorithm. Must be positive.
- `p::Float64`: which norm to use. Must be 2 or Inf.

# Returns
A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
the original problem is recovered by
`unscale_problem(problem, constraint_rescaling, variable_rescaling)`.
"""
function ruiz_rescaling(
  problem::LinearProgrammingProblem,
  num_iterations::Int64,
  p::Float64 = Inf,
)
    num_constraints, num_variables = size(problem.constraint_matrix)
    cum_constraint_rescaling = ones(num_constraints)
    cum_variable_rescaling = ones(num_variables)

    for i in 1:num_iterations
      constraint_matrix = problem.constraint_matrix

      if p == Inf
        variable_rescaling = vec(
          sqrt.(
            maximum(abs, constraint_matrix, dims = 1),
          ),
        )
      else
        @assert p == 2
        variable_rescaling = vec(
          sqrt.(
            sqrt.(
              l2_norm(constraint_matrix, 1) .^ 2,
            ),
          ),
        )
      end
      variable_rescaling[iszero.(variable_rescaling)] .= 1.0

      if num_constraints == 0
        constraint_rescaling = Float64[]
      else
        if p == Inf
          constraint_rescaling =
            vec(sqrt.(maximum(abs, constraint_matrix, dims = 2)))
        else
          @assert p == 2
          norm_of_rows = vec(l2_norm(problem.constraint_matrix, 2))

          # If the columns all have norm 1 and the row norms are equal they should
          # equal sqrt(num_variables/num_constraints) for LP.
          target_row_norm = sqrt(num_variables / num_constraints)
          
          constraint_rescaling = vec(sqrt.(norm_of_rows / target_row_norm))
        end
        constraint_rescaling[iszero.(constraint_rescaling)] .= 1.0
      end
      scale_problem(problem, constraint_rescaling, variable_rescaling)

      cum_constraint_rescaling .*= constraint_rescaling
      cum_variable_rescaling .*= variable_rescaling
    end

    return cum_constraint_rescaling, cum_variable_rescaling
end


"""
Rescales a linear programming problem by dividing each row and column of the
constraint matrix by the sqrt its respective L2 norm, adjusting the other
problem data accordingly.

# Arguments
- `problem::QuadraticProgrammingProblem`: The input quadratic programming
  problem. This is modified to store the transformed problem.

# Returns
A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
the original problem is recovered by
`unscale_problem(problem, constraint_rescaling, variable_rescaling)`.
"""
function l2_norm_rescaling(problem::LinearProgrammingProblem)
    num_constraints, num_variables = size(problem.constraint_matrix)
  
    norm_of_rows = vec(l2_norm(problem.constraint_matrix, 2))
    norm_of_columns = vec(l2_norm(problem.constraint_matrix, 1))
  
    norm_of_rows[iszero.(norm_of_rows)] .= 1.0
    norm_of_columns[iszero.(norm_of_columns)] .= 1.0
  
    column_rescale_factor = sqrt.(norm_of_columns)
    row_rescale_factor = sqrt.(norm_of_rows)
    scale_problem(problem, row_rescale_factor, column_rescale_factor)
  
    return row_rescale_factor, column_rescale_factor
end



"""Preprocesses the original problem, and returns a ScaledQpProblem struct.
Applies L_inf Ruiz rescaling for `l_inf_ruiz_iterations` iterations. If
`l2_norm_rescaling` is true, applies L2 norm rescaling. `problem` is not
modified.
"""
function rescale_problem(
  l_inf_ruiz_iterations::Int,
  l2_norm_rescaling_flag::Bool,
  verbosity::Int64,
  original_problem::LinearProgrammingProblem,
)
    problem = deepcopy(original_problem)
    if verbosity >= 4
      println("Problem before rescaling:")
      print_problem_details(original_problem)
    end
  
    num_constraints, num_variables = size(problem.constraint_matrix)
    constraint_rescaling = ones(num_constraints)
    variable_rescaling = ones(num_variables)
  
    if l_inf_ruiz_iterations > 0
      con_rescale, var_rescale = ruiz_rescaling(problem, l_inf_ruiz_iterations, Inf)
      constraint_rescaling .*= con_rescale
      variable_rescaling .*= var_rescale
    end
  
    if l2_norm_rescaling_flag
      con_rescale, var_rescale = l2_norm_rescaling(problem)
      constraint_rescaling .*= con_rescale
      variable_rescaling .*= var_rescale
    end
   
    scaled_problem = ScaledLpProblem(
      original_problem,
      problem,
      constraint_rescaling,
      variable_rescaling,
    )
  
    if verbosity >= 3
      if l_inf_ruiz_iterations == 0 && !l2_norm_rescaling
        println("No rescaling.")
      else
        print("Problem after rescaling ")
        print("(Ruiz iterations = $l_inf_ruiz_iterations, ")
        println("l2_norm_rescaling = $l2_norm_rescaling_flag):")
        print_problem_details(scaled_problem.scaled_lp)
      end
    end
  
    return scaled_problem
end



"""
Rescales `problem` in place. If we let `D = diag(cum_variable_rescaling)` and
`E = diag(cum_constraint_rescaling)`, then `problem` is modified such that:
    objective_matrix = D^-1 objective_matrix D^-1
    objective_vector = D^-1 objective_vector
    objective_constant = objective_constant
    variable_lower_bound = D variable_lower_bound
    variable_upper_bound = D variable_upper_bound
    constraint_matrix = E^-1 constraint_matrix D^-1
    right_hand_side = E^-1 right_hand_side
The scaling vectors must be positive.
"""
function scale_problem(
  problem::LinearProgrammingProblem,
  constraint_rescaling::Vector{Float64},
  variable_rescaling::Vector{Float64},
)
    @assert all(t -> t > 0, constraint_rescaling)
    @assert all(t -> t > 0, variable_rescaling)
    problem.objective_vector ./= variable_rescaling
    problem.variable_upper_bound .*= variable_rescaling
    problem.variable_lower_bound .*= variable_rescaling
    problem.right_hand_side ./= constraint_rescaling
    problem.constraint_matrix =
      Diagonal(1 ./ constraint_rescaling) *
      problem.constraint_matrix *
      Diagonal(1 ./ variable_rescaling)
    return
end

"""
Recovers the original problem from the scaled problem and the scaling vectors
in place. The inverse of `scale_problem`. This function should be only used for
testing.
"""
function unscale_problem(
  problem::LinearProgrammingProblem,
  constraint_rescaling::Vector{Float64},
  variable_rescaling::Vector{Float64},
)
    scale_problem(problem, 1 ./ constraint_rescaling, 1 ./ variable_rescaling)
    return
end

"""
Returns the l2 norm of each row or column of a matrix. The method rescales
the sum-of-squares computation by the largest absolute value if nonzero in order
to avoid overflow.

# Arguments
- `matrix::SparseMatrixCSC{Float64, Int64}`: a sparse matrix.
- `dimension::Int64`: the dimension we want to compute the norm over. Must be
  1 or 2.
  
# Returns
An array with the l2 norm of a matrix over the given dimension.
"""
function l2_norm(matrix::SparseMatrixCSC{Float64,Int64}, dimension::Int64)
    scale_factor = vec(maximum(abs, matrix, dims = dimension))
    scale_factor[iszero.(scale_factor)] .= 1.0
    if dimension == 1
      scaled_matrix = matrix * Diagonal(1 ./ scale_factor)
      return scale_factor .*
            vec(sqrt.(sum(t -> t^2, scaled_matrix, dims = dimension)))
    end

    if dimension == 2
      scaled_matrix = Diagonal(1 ./ scale_factor) * matrix
      return scale_factor .*
            vec(sqrt.(sum(t -> t^2, scaled_matrix, dims = dimension)))
    end
end