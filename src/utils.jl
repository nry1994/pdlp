import MathOptInterface as MOI

function create_stats_data_frame()
    return DataFrames.DataFrame(
      # Record the current iteration.
      iteration = Int64[],
      # Record the cumulative number of matrix-vector product.
      cumulative_kkt_passes = Float64[],
      # Record the time spent.
      time_spent = Float64[],
      # Primal objectives of the iterates; the ith entry corresponds to the primal
      # objective of the ith iterate.
      primal_objectives = Float64[],
      # Dual objectives of the iterates; the ith entry corresponds to the dual
      # objective of the ith iterate.
      dual_objectives = Float64[],
      # Primal norms of the iterates; the ith entry corresponds to the primal
      # norm of the ith iterate.
      primal_solution_norms = Float64[],
      # Dual norms of the iterates; the ith entry corresponds to the dual
      # norm of the ith iterate.
      dual_solution_norms = Float64[],
      # Primal delta norms of the iterates; the ith entry corresponds to the
      # primal delta norm of the ith iterate.
      primal_delta_norms = Float64[],
      # Dual delta norms of the iterates; the ith entry corresponds to the dual
      # delta norm of the ith iterate.
      dual_delta_norms = Float64[],
      # Primal feasibility of the iterates; the ith entry corresponds to the
      # primal feasibility of the ith iterate.
      primal_feasibility = Float64[],
      # Dual feasibility of the iterates; the ith entry corresponds to the
      # dual feasibility of the ith iterate.
      dual_feasibility = Float64[],
      # KKT error of the iterates; the ith entry corresponds to the
      # KKT error of the ith iterate.
      kkt_error = Float64[],
    )
  end


"""
This code computes the unscaled iteration stats.
The input iterates to this function have been scaled according to
scaled_problem.
"""
function evaluate_unscaled_iteration_stats(
  scaled_problem::ScaledLpProblem,
  iteration::Int64,
  cumulative_kkt_passes::Float64,
  time_spent::Float64,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
  primal_delta::Vector{Float64},
  dual_delta::Vector{Float64},
)
  # Unscale iterates.
  original_primal_solution::Vector{Float64} =
    primal_solution ./ scaled_problem.variable_rescaling
  original_dual_solution::Vector{Float64} =
    dual_solution ./ scaled_problem.constraint_rescaling

  original_primal_delta::Vector{Float64} =
    primal_delta ./ scaled_problem.variable_rescaling
  original_dual_delta::Vector{Float64} =
    dual_delta ./ scaled_problem.constraint_rescaling

  return compute_stats(
    scaled_problem.original_lp,
    iteration,
    cumulative_kkt_passes,
    time_spent,
    original_primal_solution,
    original_dual_solution,
    original_primal_delta,
    original_dual_delta,
  )
end


"""
Computes statistics for the current iteration. The arguments primal_delta and
dual_delta correspond to the difference between the last two primal and dual
iterates, respectively.
"""
function compute_stats(
  problem::LinearProgrammingProblem,
  iteration::Int64,
  cumulative_kkt_passes::Float64,
  time_spent::Float64,
  primal_solution::Vector{Float64},
  dual_solution::Vector{Float64},
  primal_delta::Vector{Float64},
  dual_delta::Vector{Float64},
)
  
  primal_objective = problem.objective_vector'*primal_solution
  dual_objective = problem.right_hand_side'*dual_solution
	
  primal_feasibility_inequality = max.(
    -problem.right_hand_side[1:problem.num_equalities]+
    problem.constraint_matrix[1:problem.num_equalities,:]*primal_solution, 0)
  primal_feasibility_equation = max.(
    problem.right_hand_side-
    problem.constraint_matrix*primal_solution, 0)
  dual_feasibility = max.(
    -problem.objective_vector+
    problem.constraint_matrix'*dual_solution, 0)
  duality_gap = max.(
    problem.objective_vector'*primal_solution-
    problem.right_hand_side'*dual_solution, 0)

  kkt_error = sqrt(
    sum(primal_feasibility_inequality.^2)+
    sum(primal_feasibility_equation.^2)+
    sum(dual_feasibility.^2)+
    sum(duality_gap.^2)
    )

  return DataFrames.DataFrame(
    iteration = iteration,
    cumulative_kkt_passes = cumulative_kkt_passes,
    time_spent = time_spent,
    primal_objectives = primal_objective,
    dual_objectives = dual_objective,
    primal_solution_norms = norm(primal_solution),
    dual_solution_norms = norm(dual_solution),
    primal_delta_norms = norm(primal_delta),
    dual_delta_norms = norm(dual_delta),
    primal_feasibility = sqrt(sum(primal_feasibility_inequality.^2)+sum(primal_feasibility_equation.^2)),
    dual_feasibility = norm(dual_feasibility),
    kkt_error = kkt_error,
  )
end

"""
  display_iteration_stats_heading()
The heading for the iteration stats table. See
README.md for documentation on what each heading means.
"""
function display_iteration_stats_heading()
  Printf.@printf(
    "%s | %s | %s |",
    rpad("runtime", 24),
    rpad("residuals", 26),
    rpad(" solution information", 26),
    # rpad("relative residuals", 23)
  )
  println("")
  Printf.@printf(
    "%s %s %s | %s %s  %s | %s %s %s |",
    rpad("#iter", 7),
    rpad("#kkt", 8),
    rpad("seconds", 7),
    rpad("pr norm", 8),
    rpad("du norm", 8),
    rpad("gap", 7),
    rpad(" pr obj", 9),
    rpad("pr norm", 8),
    rpad("du norm", 7),
    # rpad("rel pr", 7),
    # rpad("rel du", 7),
    # rpad("rel gap", 7)
  )
  print("\n")
end

"""
Make sure that a float is of a constant length, irrespective if it is negative
or positive.
"""
function lpad_float(number::Float64)
  return lpad(Printf.@sprintf("%.1e", number), 8)
end

"""
Displays a row of the iteration stats table.
"""
function display_iteration_stats(
  iteration_stats::DataFrames.DataFrame,
)
    Printf.@printf(
      "%s  %.1e  %.1e | %.1e  %.1e  %s | %s  %.1e  %.1e |",
      rpad(string(iteration_stats[:,"iteration"][end]), 6),
      iteration_stats[:,"cumulative_kkt_passes"][end],
      iteration_stats[:,"time_spent"][end],
      iteration_stats[:,"primal_feasibility"][end],
      iteration_stats[:,"dual_feasibility"][end],
      lpad_float(
        abs(iteration_stats[:,"primal_objectives"][end] -
        iteration_stats[:,"dual_objectives"][end]),
      ),
      lpad_float(iteration_stats[:,"primal_objectives"][end]),
      iteration_stats[:,"primal_solution_norms"][end],
      iteration_stats[:,"dual_solution_norms"][end]
    )
  print("\n")
end



"""
Logging while the algorithm is running.
"""
function log_iteration(
  problem::LinearProgrammingProblem,
  iteration_stats::DataFrames.DataFrame,
)
  Printf.@printf(
    "iteration = %5d objectives=(%9g, %9g) norms=(%9g, %9g) res_norm=(%9g, %9g) kkt=%.2e\n",
    iteration_stats[:,"iteration"][end],
    iteration_stats[:,"primal_objectives"][end],
    iteration_stats[:,"dual_objectives"][end],
    iteration_stats[:,"primal_solution_norms"][end],
    iteration_stats[:,"dual_solution_norms"][end],
    iteration_stats[:,"primal_delta_norms"][end],
    iteration_stats[:,"dual_delta_norms"][end],
    iteration_stats[:,"kkt_error"][end],
  )
end


function get_row_l2_norms(matrix::SparseMatrixCSC{Float64,Int64})
    row_norm_squared = zeros(size(matrix, 1))
    nzval = nonzeros(matrix)
    rowval = rowvals(matrix)
    for i in 1:length(nzval)
      row_norm_squared[rowval[i]] += nzval[i]^2
    end
  
    return sqrt.(row_norm_squared)
  end
  
function get_col_l2_norms(matrix::SparseMatrixCSC{Float64,Int64})
    col_norms = zeros(size(matrix, 2))
    for j in 1:size(matrix, 2)
      col_norms[j] = norm(nonzeros(matrix[:, j]), 2)
    end
    return col_norms
end
  
  
function get_row_l_inf_norms(matrix::SparseMatrixCSC{Float64,Int64})
    row_norm = zeros(size(matrix, 1))
    nzval = nonzeros(matrix)
    rowval = rowvals(matrix)
    for i in 1:length(nzval)
      row_norm[rowval[i]] = max(abs(nzval[i]), row_norm[rowval[i]])
    end
  
    return row_norm
end
  
function get_col_l_inf_norms(matrix::SparseMatrixCSC{Float64,Int64})
    col_norms = zeros(size(matrix, 2))
    for j in 1:size(matrix, 2)
      col_norms[j] = norm(nonzeros(matrix[:, j]), Inf)
      typeof(matrix[:, j])
    end
    return col_norms
end

"""
  print_problem_details(lp)
This is primarily useful for detecting when a problem is poorly conditioned and
needs rescaling.
"""
function print_problem_details(lp::LinearProgrammingProblem)
  println(
    "  There are ",
    size(lp.constraint_matrix, 2),
    " variables, ",
    size(lp.constraint_matrix, 1),
    " constraints (including ",
    lp.num_equalities,
    " equalities) and ",
    SparseArrays.nnz(lp.constraint_matrix),
    " nonzero coefficients.",
  )

  print("  Absolute value of nonzero constraint matrix elements: ")
  Printf.@printf(
    "largest=%f, smallest=%f, avg=%f\n",
    maximum(abs, nonzeros(lp.constraint_matrix)),
    minimum(abs, nonzeros(lp.constraint_matrix)),
    sum(abs, nonzeros(lp.constraint_matrix)) /
    length(nonzeros(lp.constraint_matrix))
  )

  col_norms = get_col_l_inf_norms(lp.constraint_matrix)
  row_norms = get_row_l_inf_norms(lp.constraint_matrix)

  print("  Constraint matrix, infinity norm: ")
  Printf.@printf(
    "max_col=%f, min_col=%f, max_row=%f, min_row=%f\n",
    maximum(col_norms),
    minimum(col_norms),
    maximum(row_norms),
    minimum(row_norms)
  )

  print("  Absolute value of objective vector elements: ")
  Printf.@printf(
    "largest=%f, smallest=%f, avg=%f\n",
    maximum(abs, lp.objective_vector),
    minimum(abs, lp.objective_vector),
    sum(abs, lp.objective_vector) / length(lp.objective_vector)
  )

  print("  Absolute value of rhs vector elements: ")
  Printf.@printf(
    "largest=%f, smallest=%f, avg=%f\n",
    maximum(abs, lp.right_hand_side),
    minimum(abs, lp.right_hand_side),
    sum(abs, lp.right_hand_side) / length(lp.right_hand_side)
  )

  bound_gaps = lp.variable_upper_bound - lp.variable_lower_bound
  finite_bound_gaps = bound_gaps[isfinite.(bound_gaps)]

  print("  Gap between upper and lower bounds: ")
  Printf.@printf(
    "#finite=%i of %i, largest=%f, smallest=%f, avg=%f\n",
    length(finite_bound_gaps),
    length(bound_gaps),
    length(finite_bound_gaps) > 0 ? maximum(finite_bound_gaps) : NaN,
    length(finite_bound_gaps) > 0 ? minimum(finite_bound_gaps) : NaN,
    length(finite_bound_gaps) > 0 ?
    sum(finite_bound_gaps) / length(finite_bound_gaps) : NaN
  )

  println()
end


##########################################################
##########################################################
##########################################################

MOI.Utilities.@product_of_sets(RHS, MOI.Zeros)

const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing,
        },
        Vector{Float64},
        RHS{Float64},
    },
}

function MOI.add_constrained_variables(
    model::OptimizerCache,
    set::MOI.Nonnegatives,
)
    x = MOI.add_variables(model, MOI.dimension(set))
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Nonnegatives}(x[1].value)
    return x, ci
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    x_primal::Dict{MOI.VariableIndex,Float64}
    termination_status::MOI.TerminationStatusCode

    function Optimizer()
        return new(Dict{MOI.VariableIndex,Float64}(), MOI.OPTIMIZE_NOT_CALLED)
    end
end

function MOI.is_empty(model::Optimizer)
    return isempty(model.x_primal) &&
        model.termination_status == MOI.OPTIMIZE_NOT_CALLED
end

function MOI.empty!(model::Optimizer)
    empty!(model.x_primal)
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    return
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{MOI.Zeros},
)
    return true
end

MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Reals}) = false

function MOI.supports_add_constrained_variables(
    ::Optimizer,
    ::Type{MOI.Nonnegatives},
)
    return true
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    @assert all(iszero, cache.variables.lower)
    @assert all(==(Inf), cache.variables.upper)
    A = convert(
        SparseArrays.SparseMatrixCSC{Float64,Int},
        cache.constraints.coefficients,
    )
    b = -cache.constraints.constants
    c = zeros(size(A, 2))

    offset = cache.objective.scalar_affine.constant
    for term in cache.objective.scalar_affine.terms
        c[term.variable.value] += term.coefficient
    end
    # if cache.objective.sense == MOI.MAX_SENSE
    #     c *= -1
    # end

    # construct lp
    lp = LinearProgrammingProblem(zeros(size(A,2)),repeat([Inf],size(A,2)),c,A,b,size(A,1))
    #
    solver_output = solve(lp,20000,1e-6,zeros(size(A,2)), zeros(size(A,1)))
    if solver_output.status == STATUS_OPTIMAL
        dest.termination_status = MOI.OPTIMAL
    else
      dest.termination_status = MOI.OTHER_ERROR
    end
    x_primal = solver_output.primal_solution

    for x in MOI.get(src, MOI.ListOfVariableIndices())
        dest.x_primal[x] = x_primal[index_map[x].value]
    end
    return index_map, false
end

function MOI.get(model::Optimizer, ::MOI.VariablePrimal, x::MOI.VariableIndex)
    return model.x_primal[x]
end

#

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.termination_status == MOI.OPTIMAL ? 1 : 0
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return "$(model.termination_status)"
end

#

MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status

function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
    if model.termination_status == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    else
        return MOI.NO_SOLUTION
    end
end

MOI.get(model::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

MOI.get(::Optimizer, ::MOI.SolverName) = "SimplePDLP"