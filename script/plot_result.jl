import Plots
import JLD2
using ArgParse
include("../src/SimplePDLP.jl")

# @assert length(ARGS) == 3
# result_folder = ARGS[1]
# figure_directory = ARGS[2]
# problem_name = ARGS[3]

"""
Defines parses and args.

# Returns
A dictionary with the values of the command-line arguments.
"""
function parse_command_line()
  arg_parse = ArgParse.ArgParseSettings()

  ArgParse.@add_arg_table! arg_parse begin
    "--directory_for_solver_output"
    help = "The directory for solver output."
    arg_type = String
    required = true

    "--figure_directory"
    help = "The directory for figures."
    arg_type = String
    required = true


    "--problem_name"
    help = "The instance to plot."
    arg_type = String
    default = "neos5"
  end

  return ArgParse.parse_args(arg_parse)
end

function main()
    parsed_args = parse_command_line()
    directory_for_solver_output = parsed_args["directory_for_solver_output"]
    figure_directory = parsed_args["figure_directory"]
    problem_name = parsed_args["problem_name"]
    solver_output = JLD2.load(joinpath("$(directory_for_solver_output)", "$(problem_name).jld2"))
    solver_output = solver_output["solver_output"]
    
    kkt_error = solver_output.iteration_stats[:,"kkt_error"]
    
    kkt_plt = Plots.plot()
    
    Plots.plot!(
        1:5:(5*length(kkt_error)),
        kkt_error,
        linewidth=1,
        #color = "green",
        #legend = :topright,
        xlabel = "iterations",
        ylabel = "KKT residual",
        #xaxis=:log,
        xguidefontsize=12,
        yaxis=:log,
        yguidefontsize=12,
        label=problem_name
    )
    
    Plots.savefig(kkt_plt,joinpath("$(figure_directory)", "$(problem_name).png")) 
end

main()
