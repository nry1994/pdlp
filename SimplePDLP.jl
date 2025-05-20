using LinearAlgebra

using JuMP
using Logging
using Printf
using SparseArrays
using DataFrames
using Plots
using JLD2

using GZip
using Statistics
using StructTypes



const Diagonal = LinearAlgebra.Diagonal
const diag = LinearAlgebra.diag
const dot = LinearAlgebra.dot
const norm = LinearAlgebra.norm
const opnorm = LinearAlgebra.opnorm
const nzrange = SparseArrays.nzrange
const nnz = SparseArrays.nnz
const nonzeros = SparseArrays.nonzeros
const rowvals = SparseArrays.rowvals
const sparse = SparseArrays.sparse
const SparseMatrixCSC = SparseArrays.SparseMatrixCSC
const spdiagm = SparseArrays.spdiagm
const spzeros = SparseArrays.spzeros

include("io.jl")
include("preprocess.jl")
include("utils.jl")
include("solver.jl")
