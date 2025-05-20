using SparseArrays
using LinearAlgebra
using JuMP

@enum SENSE MAXIMIZE = -1 FEASIBILITY = 0 MINIMIZE = 1

Base.@kwdef mutable struct GeneralLinearProgramming
    nRows::Int64 = 0
    nCols::Int64 = 0
    cost::Vector{Float64} = []
    A::SparseArrays.SparseMatrixCSC{Float64,Int64} = SparseMatrixCSC{Float64,Int64}(undef, 0, 0)
    lhs::Vector{Float64} = []
    rhs::Vector{Float64} = []
    lb::Vector{Float64} = []
    ub::Vector{Float64} = []
    sense::SENSE = MINIMIZE
    offset::Float64 = 0.0
end

Base.@kwdef mutable struct LinearProgrammingProblem
    nRows::Int64 = 0
    nCols::Int64 = 0
    cost::Vector{Float64} = []
    A::SparseArrays.SparseMatrixCSC{Float64,Int64} = SparseMatrixCSC{Float64,Int64}(undef, 0, 0)
    b::Vector{Float64} = []
    lb::Vector{Float64} = []
    ub::Vector{Float64} = []
    sense::SENSE = MINIMIZE
    offset::Float64 = 0.0
end


function read_general_linear_programming(file::String)

    model = read_from_file(file)
    relax_integrality(model)
    lp = lp_matrix_data(model)

    nRows, nCols = size(lp.A)
    cost = lp.c
    A = lp.A
    lhs = lp.b_lower
    rhs = lp.b_upper
    lb = lp.x_lower
    ub = lp.x_upper
    sense = lp.sense == MIN_SENSE ? MINIMIZE :
            lp.sense == MAX_SENSE ? MAXIMIZE :
            lp.sense == FEASIBILITY_SENSE ? FEASIBILITY : MINIMIZE
    offset = lp.c_offset

    return GeneralLinearProgramming(nRows, nCols, cost, A, lhs, rhs, lb, ub, sense, offset)

end


function reformulate_general_lp(glp::GeneralLinearProgramming, full_row_rank::Bool=false)

    if full_row_rank
        return reformulate_full_row_rank_general_lp(glp)
    else
        return reformulate_std_general_lp(glp)
    end

end

function reformulate_std_general_lp(glp::GeneralLinearProgramming)

    idx_eq = (glp.lhs .== glp.rhs)
    idx_ge = (glp.lhs .> -Inf) .& .!idx_eq
    idx_le = (glp.rhs .< +Inf) .& .!idx_eq

    n_eq = sum(idx_eq)
    n_ge = sum(idx_ge)
    n_le = sum(idx_le)

    nRows = n_eq + n_ge + n_le
    nCols = glp.nCols + n_ge + n_le

    cost = [Int(glp.sense) * glp.cost; zeros(n_ge + n_le)]
    A = SparseMatrixCSC{Float64,Int64}(
        [[glp.A[idx_eq, :];
            -glp.A[idx_ge, :];
            glp.A[idx_le, :]] [spzeros(n_eq, n_ge + n_le); sparse(I, n_ge + n_le, n_ge + n_le)]]
    )
    b = [glp.lhs[idx_eq]; -glp.lhs[idx_ge]; glp.rhs[idx_le]]
    lb = [glp.lb; zeros(n_ge + n_le)]
    ub = [glp.ub; fill(Inf, n_ge + n_le)]

    return LinearProgrammingProblem(nRows, nCols, cost, A, b, lb, ub, glp.sense, glp.offset)
end

function reformulate_full_row_rank_general_lp(glp::GeneralLinearProgramming)

    nRows = glp.nRows
    nCols = glp.nCols + glp.nRows

    cost = [Int(glp.sense) * glp.cost; zeros(glp.nRows)]
    A = SparseMatrixCSC{Float64,Int64}(
        [glp.A -sparse(I, glp.nRows, glp.nRows)]
    )
    b = zeros(glp.nRows)
    lb = [glp.lb; glp.lhs]
    ub = [glp.ub; glp.rhs]

    return LinearProgrammingProblem(nRows, nCols, cost, A, b, lb, ub, glp.sense, glp.offset)
end


