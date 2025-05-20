mutable struct ScaledLpProblem
  original_lp::LinearProgrammingProblem
  scaled_lp::LinearProgrammingProblem
  constraint_rescaling::Vector{Float64}
  variable_rescaling::Vector{Float64}
end

function ruiz_rescaling(
  problem::LinearProgrammingProblem,
  num_iterations::Int64,
  p::Float64=Inf,
)
  num_constraints, num_variables = size(problem.A)
  cum_constraint_rescaling = ones(num_constraints)
  cum_variable_rescaling = ones(num_variables)

  for i in 1:num_iterations
    A = problem.A

    if p == Inf
      variable_rescaling = vec(
        sqrt.(
          maximum(abs, A, dims=1),
        ),
      )
    else
      @assert p == 2
      variable_rescaling = vec(
        sqrt.(
          sqrt.(
            l2_norm(A, 1) .^ 2,
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
          vec(sqrt.(maximum(abs, A, dims=2)))
      else
        @assert p == 2
        norm_of_rows = vec(l2_norm(problem.A, 2))

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


function l2_norm_rescaling(problem::LinearProgrammingProblem)
  num_constraints, num_variables = size(problem.A)

  norm_of_rows = vec(l2_norm(problem.A, 2))
  norm_of_columns = vec(l2_norm(problem.A, 1))

  norm_of_rows[iszero.(norm_of_rows)] .= 1.0
  norm_of_columns[iszero.(norm_of_columns)] .= 1.0

  column_rescale_factor = sqrt.(norm_of_columns)
  row_rescale_factor = sqrt.(norm_of_rows)
  scale_problem(problem, row_rescale_factor, column_rescale_factor)

  return row_rescale_factor, column_rescale_factor
end


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

  num_constraints, num_variables = size(problem.A)
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
    cost = D^-1 cost
    objective_constant = objective_constant
    lb = D lb
    ub = D ub
    A = E^-1 A D^-1
    b = E^-1 b
The scaling vectors must be positive.
"""
function scale_problem(
  problem::LinearProgrammingProblem,
  constraint_rescaling::Vector{Float64},
  variable_rescaling::Vector{Float64},
)
  @assert all(t -> t > 0, constraint_rescaling)
  @assert all(t -> t > 0, variable_rescaling)
  problem.cost ./= variable_rescaling
  problem.ub .*= variable_rescaling
  problem.lb .*= variable_rescaling
  problem.b ./= constraint_rescaling
  problem.A =
    Diagonal(1 ./ constraint_rescaling) *
    problem.A *
    Diagonal(1 ./ variable_rescaling)
  return
end


function unscale_problem(
  problem::LinearProgrammingProblem,
  constraint_rescaling::Vector{Float64},
  variable_rescaling::Vector{Float64},
)
  scale_problem(problem, 1 ./ constraint_rescaling, 1 ./ variable_rescaling)
  return
end


function l2_norm(matrix::SparseMatrixCSC{Float64,Int64}, dimension::Int64)
  scale_factor = vec(maximum(abs, matrix, dims=dimension))
  scale_factor[iszero.(scale_factor)] .= 1.0
  if dimension == 1
    scaled_matrix = matrix * Diagonal(1 ./ scale_factor)
    return scale_factor .*
           vec(sqrt.(sum(t -> t^2, scaled_matrix, dims=dimension)))
  end

  if dimension == 2
    scaled_matrix = Diagonal(1 ./ scale_factor) * matrix
    return scale_factor .*
           vec(sqrt.(sum(t -> t^2, scaled_matrix, dims=dimension)))
  end
end