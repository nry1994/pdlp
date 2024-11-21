import LinearAlgebra

import Logging
import Printf
import SparseArrays
import DataFrames
import Plots
import JLD2

import GZip
import QPSReader
import Statistics
import StructTypes
import MathOptInterface as MOI


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
