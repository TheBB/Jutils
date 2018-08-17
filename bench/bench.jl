import Pkg
Pkg.activate(".")

import Base: print
using ArgParse
using BenchmarkTools
using Printf
using Statistics


include("run.jl")

function timetostring(time::Float64)
    time < 1e3 && return @sprintf "%10.2f ns" time/1e0
    time < 1e6 && return @sprintf "%10.2f μs" time/1e3
    time < 1e9 && return @sprintf "%10.2f ms" time/1e6
    @sprintf "%10.2f  s" time/1e9
end

function bytestostring(bytes::Int)
    bytes == 0   && return @sprintf   "%10s    " ""
    bytes < 2^10 && return @sprintf "%10.2f   B" float(bytes)
    bytes < 2^20 && return @sprintf "%10.2f KiB" bytes/2^10
    bytes < 2^30 && return @sprintf "%10.2f MiB" bytes/2^20
    @sprintf "%10.2f GiB" bytes/2^30
end

function numtostring(num::Int)
    num == 0 && return @sprintf "%10s" ""
    @sprintf "%10d" num
end

tagtostring(tag::String) = tag
tagtostring(tag::Tuple) = join((tagtostring(t) for t in tag), ",")
tagtostring(tag::Vector{Any}) = join((tagtostring(t) for t in tag), "/")

function print(tag, trial::BenchmarkTools.TrialEstimate)
    out = @sprintf(
        "%40s %s %s %s %s",
        tagtostring(tag),
        timetostring(trial.time),
        timetostring(trial.gctime),
        bytestostring(trial.memory),
        numtostring(trial.allocs),
    )
    println(out)
end

function print(tag, trial::BenchmarkTools.TrialJudgement)
    out = @sprintf(
        "%40s %+12.2f%% %+12.2f%% %+13.2f%% %+9.2f%%",
        tagtostring(tag),
        (trial.ratio.time - 1) * 100,
        (trial.ratio.gctime - 1) * 100,
        (trial.ratio.memory - 1) * 100,
        (trial.ratio.allocs - 1) * 100,
    )
    print(out)

    if trial.time == :regression
        printstyled("   regression"; color=:red, bold=true)
    elseif trial.time == :improvement
        printstyled("   improvement"; color=:green, bold=true)
    end
    println("")
end

function summary(results::BenchmarkTools.BenchmarkGroup)
    println(@sprintf "%40s %13s %13s %14s %10s" "Name" "Time" "GC Time" "Memory" "Allocs")
    println("----------------------------------------------------------------------------------------------")
    for (tag, trial) in leaves(results)
        print(tag, trial)
    end
end


function cmd_run(cmd)
    results = median(runbench())

    if cmd["out"] != nothing
        BenchmarkTools.save(cmd["out"], results)
        return
    end

    summary(results)
end

function cmd_summary(cmd)
    results = BenchmarkTools.load(cmd["filename"])[1]
    summary(results)
end

function cmd_compare(cmd)
    before = BenchmarkTools.load(cmd["before"])[1]
    after = BenchmarkTools.load(cmd["after"])[1]
    results = judge(after, before)
    summary(results)
end


function main()
    interface = ArgParseSettings()
    @add_arg_table interface begin
        "run"
            help = "run benchmarks"
            action = :command
        "summary"
            help = "summarize benchmarks in file"
            action = :command
        "compare"
            help = "compare two benchmarks"
            action = :command
    end

    @add_arg_table interface["run"] begin
        "--filter", "-f"
            help = "filter which benchmarks to run"
        "--out", "-o"
            help = "output file"
    end

    @add_arg_table interface["summary"] begin
        "filename"
            help = "filename"
            required = true
    end

    @add_arg_table interface["compare"] begin
        "before"
            help = "baseline to compare against"
            required = true
        "after"
            help = "new results to compare"
            required = true
    end

    cmd = parse_args(ARGS, interface)
    cmd["%COMMAND%"] == "run" && cmd_run(cmd[cmd["%COMMAND%"]])
    cmd["%COMMAND%"] == "summary" && cmd_summary(cmd[cmd["%COMMAND%"]])
    cmd["%COMMAND%"] == "compare" && cmd_compare(cmd[cmd["%COMMAND%"]])
end


main()
