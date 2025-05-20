
mutable struct PrimalDualOptimizerParameters
  step_size::Union{Float64,Nothing}
  rescale_flag::Bool
  record_every::Int64
  print_every::Int64
  verbosity::Bool
  iteration_limit::Int64
  kkt_tolerance::Float64
  initial_primal_solution::Vector{Float64}
  initial_dual_solution::Vector{Float64}
end


@enum SolutionStatus STATUS_OPTIMAL STATUS_ITERATION_LIMIT
struct PrimalDualOutput
  primal_solution::Vector{Float64}
  dual_solution::Vector{Float64}
  iteration_stats::DataFrames.DataFrame
  status::SolutionStatus
end

function projection_primal(
  primal_iterate::Vector{Float64},
  problem::LinearProgrammingProblem,)

  lower_bound, upper_bound = problem.lb, problem.ub
  for idx in 1:length(primal_iterate)
    primal_iterate[idx] = min(upper_bound[idx], max(lower_bound[idx], primal_iterate[idx]),)
  end
  return primal_iterate
end

# function projection_dual(
#   dual_iterate::Vector{Float64},
#   problem::LinearProgrammingProblem)

#   for idx in (problem.nRows+1):length(dual_iterate)
#     dual_iterate[idx] = max(dual_iterate[idx], 0.0)
#   end
#   return dual_iterate
# end

function take_pdhg_step(
  problem::LinearProgrammingProblem,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  omega::Float64,
  eta_prev::Float64,
  total_iter::Int)
  max_attempts = 10
  eta = eta_prev

  cost, A, b = problem.cost, problem.A, problem.b
  for attempt in 1:max_attempts
    temp_x = current_primal_solution - (eta / omega) .* (cost - A' * current_dual_solution)
    next_primal = projection_primal(temp_x, problem)
    temp_y = current_dual_solution - eta * omega .* (A * (2 * next_primal - current_primal_solution) - b)
    next_dual = temp_y

    diff_x = next_primal .- current_primal_solution
    diff_y = next_dual .- current_dual_solution
    # 计算加权范数平方
    norm_diff_omega_sq = omega * norm(diff_x)^2 + (norm(diff_y)^2 / omega)
    # 计算分母
    denominator = 2 * dot(diff_y, A * diff_x)
    if denominator > 0
      eta_bar = norm_diff_omega_sq / denominator
    else
      eta_bar = 1e8  # 处理分母 <= 0 的情况
    end

    k = total_iter
    alpha = 1 - (k + 1)^(-0.3)
    beta = 1 + (k + 1)^(-0.6)
    eta_prime = min(alpha * eta_bar, beta * eta)

    if eta <= eta_bar || attempt == max_attempts
      return next_primal, next_dual, eta
    else
      eta = eta_prime
      return next_primal, next_dual, eta
    end
  end
end


function optimize(
  params::PrimalDualOptimizerParameters,
  problem::LinearProgrammingProblem,
  restart_beta::Float64=0.2,  # β_sufficient
  restart_beta_necessary::Float64=0.8,
  restart_beta_artificial::Float64=0.36,
  max_epoch::Int=100,
  inner_max::Int=10_000,
  verbose::Bool=true,
)
  basic_sizes = Int[]
  unchanged_lengths = Int[]

  if params.rescale_flag
    scaled_problem = rescale_problem(10, true, 4, problem)
    scaled_lp = scaled_problem.scaled_lp
  else
    scaled_problem = rescale_problem(0, false, 0, problem)
    scaled_lp = scaled_problem.scaled_lp
  end

  primal_size = length(scaled_lp.lb)
  dual_size = length(scaled_lp.b)
  Kmat = scaled_lp.A

  # 初始化原权重
  epsilon_zero = 1e-10
  c_norm = norm(scaled_lp.cost)
  q_norm = norm(scaled_lp.b)
  current_omega = (c_norm > epsilon_zero && q_norm > epsilon_zero) ? c_norm / q_norm : 1.0

  # 跟踪前一 epoch 起始点
  last_z_start_primal = copy(params.initial_primal_solution)
  last_z_start_dual = copy(params.initial_dual_solution)
  just_restarted = false


  # ==== restart结构变量 ====
  epoch = 1
  total_iter = 0
  kkt_history = Float64[]

  # 提取上下界向量，定义公差
  lower_bound = scaled_lp.lb
  upper_bound = scaled_lp.ub
  tol = 1e-8
  # 初始化基集合跟踪变量
  prev_basic_mask = nothing
  basic_sets = Vector{Int64}[]
  basic_set_unchanged_count = 0

  # 初始化步长
  default_step_size = 0.99 / opnorm(Matrix(Kmat))
  step_size = isnothing(params.step_size) ? default_step_size : params.step_size
  current_eta = step_size  # 自适应步长跟踪

  iteration_limit = params.iteration_limit
  stats = create_stats_data_frame()

  current_primal_solution = params.initial_primal_solution
  current_dual_solution = params.initial_dual_solution
  primal_delta, dual_delta = zeros(primal_size), zeros(dual_size)



  iteration = 0
  display_iteration_stats_heading()
  cumulative_kkt_passes = 0.0
  KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

  start_time = time()

  # stats0 = evaluate_unscaled_iteration_stats(
  #   scaled_problem, total_iter, cumulative_kkt_passes, time() - start_time,
  #   current_primal_solution, current_dual_solution,
  #   zeros(primal_size), zeros(dual_size)
  # )
  # init_kkt = stats0.kkt_error[end]
  # push!(kkt_history, init_kkt)

  while epoch <= max_epoch
    if verbose
      println("==== Epoch $epoch ====")
    end
    avg_primal = zeros(primal_size)
    avg_dual = zeros(dual_size)
    sum_weight = 0.0

    # ==== primal weight 更新 ====
    if just_restarted
      delta_x = norm(current_primal_solution - last_z_start_primal)
      delta_y = norm(current_dual_solution - last_z_start_dual)
      if delta_x > epsilon_zero && delta_y > epsilon_zero
        theta = 0.5
        log_ratio = log(delta_y / delta_x)
        current_omega = exp(theta * log_ratio + (1 - theta) * log(current_omega))
      end
      just_restarted = false
    end
    last_z_start_primal = copy(current_primal_solution)
    last_z_start_dual = copy(current_dual_solution)

    # 记录epoch初始点的KKT error
    stats0 = evaluate_unscaled_iteration_stats(
      scaled_problem, total_iter, cumulative_kkt_passes, time() - start_time,
      current_primal_solution, current_dual_solution,
      zeros(primal_size), zeros(dual_size), current_omega,
    )
    init_kkt = stats0.kkt_error[end]
    push!(kkt_history, init_kkt)
    prev_kkt_error_candidate = init_kkt  # 初始化用于无局部进展检查
    if verbose
      println("  Initial KKT error: $init_kkt")
    end

    primal_delta, dual_delta = zeros(primal_size), zeros(dual_size)
    terminate = false

    for inner_iter in 1:inner_max
      total_iter += 1
      # ---- 1. PDHG step ----
      next_primal, next_dual, current_eta = take_pdhg_step(
        scaled_lp,
        current_primal_solution,
        current_dual_solution,
        current_omega,
        current_eta,
        total_iter
      )
      primal_delta = next_primal .- current_primal_solution
      dual_delta = next_dual .- current_dual_solution
      current_primal_solution .= next_primal
      current_dual_solution .= next_dual
      cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

      # ---- 2. 平均变量 ----
      sum_weight += 1.0
      avg_primal .+= current_primal_solution
      avg_dual .+= current_dual_solution
      avg_primal_now = avg_primal ./ sum_weight
      avg_dual_now = avg_dual ./ sum_weight

      # ---- 3. 统计/日志/终止 ----
      store_stats = mod(total_iter, params.record_every) == 0
      print_stats = params.verbosity && (mod(total_iter, params.record_every * params.print_every) == 0)

      if store_stats || inner_iter == 1
        # 计算当前迭代点和平均迭代点的KKT误差
        current_stats = evaluate_unscaled_iteration_stats(
          scaled_problem, total_iter, cumulative_kkt_passes, time() - start_time,
          current_primal_solution, current_dual_solution, primal_delta, dual_delta, current_omega,
        )
        avg_stats = evaluate_unscaled_iteration_stats(
          scaled_problem, total_iter, cumulative_kkt_passes, time() - start_time,
          avg_primal_now, avg_dual_now, primal_delta, dual_delta, current_omega,
        )
        kkt_error_current = current_stats.kkt_error[end]
        kkt_error_avg = avg_stats.kkt_error[end]

        # 选择KKT误差较低的重启候选点
        if kkt_error_current < kkt_error_avg
          kkt_error_candidate = kkt_error_current
          restart_primal = copy(current_primal_solution)
          restart_dual = copy(current_dual_solution)
        else
          kkt_error_candidate = kkt_error_avg
          restart_primal = copy(avg_primal_now)
          restart_dual = copy(avg_dual_now)
        end

        # 更新prev_kkt_error_candidate
        prev_kkt_error_candidate = kkt_error_candidate

        # 检查终止条件
        if kkt_error_candidate <= params.kkt_tolerance
          if verbose
            println("Optimal solution found at epoch $epoch, iter $inner_iter, KKT error: $kkt_error_candidate")
          end
          terminate = true
          break
        end

        # 检查重启条件
        sufficient_decay = kkt_error_candidate <= restart_beta * init_kkt
        necessary_decay_no_progress = kkt_error_candidate <= restart_beta_necessary * init_kkt && kkt_error_candidate > prev_kkt_error_candidate
        long_inner_loop = inner_iter >= restart_beta_artificial * total_iter

        if sufficient_decay || necessary_decay_no_progress || long_inner_loop
          if verbose
            reason = sufficient_decay ? "sufficient decay" : necessary_decay_no_progress ? "necessary decay + no progress" : "long inner loop"
            println("Restart triggered at epoch $epoch, iter $inner_iter, KKT error: $kkt_error_candidate ($reason)")
          end
          # 从候选点重启
          current_primal_solution .= restart_primal
          current_dual_solution .= restart_dual
          break
        end

        # 记录统计数据（保持原逻辑，使用平均迭代点）
        this_iteration_stats = avg_stats
        if print_stats || (inner_iter == 1 && verbose)
          display_iteration_stats(this_iteration_stats)
        end
        append!(stats, this_iteration_stats)
      end

      if total_iter >= iteration_limit
        if verbose
          println("Iteration limit reached")
        end
        terminate = true
        break
      end
    end

    if terminate
      break
    end

    epoch += 1

    # 识别当前集合（达到上下界的变量）
    fixed_mask = (lower_bound .== upper_bound)
    on_lb_mask = (lower_bound .<= current_primal_solution .<= (lower_bound .+ tol)) .& .!fixed_mask
    on_ub_mask = (upper_bound .>= current_primal_solution .>= (upper_bound .- tol)) .& .!fixed_mask
    basic_mask = fixed_mask .| on_lb_mask .| on_ub_mask
    basic_indices = findall(basic_mask)
    push!(basic_sets, basic_indices)

    # 比较并统计集合变化情况
    if prev_basic_mask !== nothing && basic_mask == prev_basic_mask
      basic_set_unchanged_count += 1
    else
      basic_set_unchanged_count = 1
    end
    prev_basic_mask = basic_mask

    basic_size = length(basic_indices)
    push!(basic_sizes, basic_size)
    push!(unchanged_lengths, basic_set_unchanged_count)
    println("Iter = $(iteration) │ basic_size = $(basic_size) │ ",
      "unchanged_rounds = $(basic_set_unchanged_count)")
    if !isempty(basic_sizes)
      plt1 = plot(
        1:length(basic_sizes), basic_sizes,
        xlabel="Iteration",
        ylabel="Basis set size",
        title="Evolution of basis size during PDHG",
        lw=2,
      )
      savefig(plt1, "basis_size_evolution.png")
    end
  end

  # 返回
  original_primal_solution = current_primal_solution ./ scaled_problem.variable_rescaling
  original_dual_solution = current_dual_solution ./ scaled_problem.constraint_rescaling
  optimize_output = PrimalDualOutput(
    original_primal_solution,
    original_dual_solution,
    stats,
    total_iter >= iteration_limit ? STATUS_OPTIMAL : STATUS_OPTIMAL,
  )
  return optimize_output
end

function solve(
  problem::LinearProgrammingProblem,
  iteration_limit::Int64,
  kkt_tolerance::Float64,
  initial_primal_solution::Vector{Float64},
  initial_dual_solution::Vector{Float64},
)

  #status = MOI.OTHER_ERROR
  println("solving problem with: ")
  print("rows = ", size(problem.A, 1), ", ")
  print("cols = ", size(problem.A, 2), ", ")
  println("nnz = ", length(problem.A.nzval), ".")
  println()

  params = PrimalDualOptimizerParameters(
    nothing, # step_size (forces the solver to use a provably correct step size)
    true, # rescaling
    5, # record every
    4, # print every
    true, # verbose
    iteration_limit, # iteration limit
    kkt_tolerance, # kkt tolerance
    initial_primal_solution, # initial primal solution
    initial_dual_solution,  # initial dual solution
  )
  solver_output = optimize(params, problem)

  return solver_output
end




