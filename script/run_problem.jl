include("../src/SimplePDLP.jl")
using ArgParse

# @assert length(ARGS) == 4
# problem_folder = ARGS[1]
# output_directory = ARGS[2]
# problem_name = ARGS[3]
# kkt_tolerance = ARGS[4]

"""
Defines parses and args.

# Returns
A dictionary with the values of the command-line arguments.
"""
function parse_command_line()
  arg_parse = ArgParse.ArgParseSettings()

  ArgParse.@add_arg_table! arg_parse begin
    "--problem_folder"
    help = "The directory for instances."
    arg_type = String
    required = true

    "--output_directory"
    help = "The directory for output files."
    arg_type = String
    required = true


    "--problem_name"
    help = "The instance to solve."
    arg_type = String
    default = "neos5"

    "--kkt_tolerance"
    help = "KKT tolerance of the solution"
    arg_type = Float64
    default = 1e-6

    "--iteration_limit"
    help = "Maximum iteration number."
    arg_type = Int64
    default = 20000
  end

  return ArgParse.parse_args(arg_parse)
end

function main()
    parsed_args = parse_command_line()
    problem_folder = parsed_args["problem_folder"]
    output_directory = parsed_args["output_directory"]
    problem_name = parsed_args["problem_name"]
    kkt_tolerance = parsed_args["kkt_tolerance"]
    iteration_limit = parsed_args["iteration_limit"]

    instance_path = joinpath("$(problem_folder)", "$(problem_name).mps.gz")
    lp = qps_reader_to_standard_form(instance_path)
    m,n = size(lp.constraint_matrix)
    
    solver_output = solve(lp, iteration_limit, kkt_tolerance, zeros(n), zeros(m))
    JLD2.jldsave(joinpath("$(output_directory)","$(problem_name).jld2"); solver_output)
end

main()


