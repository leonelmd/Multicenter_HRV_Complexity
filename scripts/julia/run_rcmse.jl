# -*- coding: utf-8 -*-
#
# Generic rcMSE driver — computes the multiscale entropy curves that every
# complexity result in this study is built on.
#
# Reads a directory of per-subject RR-interval files, computes refined-composite
# multiscale entropy with a FIXED tolerance r, and writes the curves plus the
# nAUC / LRS summaries.
#
# The tolerance is computed ONCE per subject from the original RRi series
# (r = rfactor * std(RRi)) and passed unchanged to every scale. See entropy.jl.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   julia --project=. run_rcmse.jl --input <dir> --out <file.csv> [options]
#
#   --input    directory of per-subject files (required)
#   --format   peaks | rri            (default: peaks)
#                peaks = CSV with a `sample` column of R-peak indices; RRi is
#                        diff(sample) in ms (requires --fs, default 1000)
#                rri   = two columns (time_s, rri_s) or one column of RRi
#   --group    directory-name | prefix | metadata:<csv>   (default: directory-name)
#   --out      output CSV for the curves (required)
#   --etype    sample | fuzzy         (default: sample)
#   --m        embedding dimension    (default: 2)
#   --rfactor  tolerance factor       (default: 0.2)
#   --scales   max scale              (default: 20)
#   --minratio minimum points per coarse-grained series (default: 1 = compute all
#              requested scales). Reliability is enforced downstream when the
#              complexity index is formed (CETRAM/Cruces use tau<=5, Nagoya
#              tau<=20), so this driver stays a pure computation tool and the
#              released *_mse.csv files carry the full 1..20 curves. Set
#              --minratio 200 to have the driver apply the N/tau rule itself.
#   --fs       sampling rate in Hz for peaks format (default: 1000)
#
# Examples
#   # CETRAM: cleaned peak files under results/cleaned_detections/{Control,PD}/
#   julia --project=. run_rcmse.jl \
#         --input ../../../../CETRAM/public_release/results/cleaned_detections \
#         --out ../../data/chile_mse.csv --scales 20
#
#   # Nagoya scale profile out to tau=60
#   julia --project=. run_rcmse.jl --input <nagoya_rri_dir> --format rri \
#         --group metadata:<metadata.csv> --out japan_profile.csv --scales 60
#
# Reproduce the Julia environment first:
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'

using CSV, DataFrames, Statistics, Printf

include(joinpath(@__DIR__, "entropy.jl"))

# ---------------------------------------------------------------- args
function getarg(flag, default = nothing)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("missing value for $flag")
    return ARGS[i + 1]
end

INPUT    = getarg("--input")
OUTCSV   = getarg("--out")
FORMAT   = getarg("--format", "peaks")
GROUPSPEC = getarg("--group", "directory-name")
ETYPE    = getarg("--etype", "sample")
M        = parse(Int,     getarg("--m", "2"))
RFACTOR  = parse(Float64, getarg("--rfactor", "0.2"))
MAXSCALE = parse(Int,     getarg("--scales", "20"))
MINRATIO = parse(Int,     getarg("--minratio", "1"))
FS       = parse(Float64, getarg("--fs", "1000"))

(INPUT === nothing || OUTCSV === nothing) &&
    error("--input and --out are required; see the header of this file")

const RRI_MIN, RRI_MAX = 300.0, 2000.0   # physiological guard (ms)

# ---------------------------------------------------------------- io
"""Return RRi in ms from one subject file, already physiologically bounded."""
function load_rri(path::String, format::String)
    if format == "peaks"
        df = CSV.read(path, DataFrame)
        col = "sample" in names(df) ? :sample : Symbol(names(df)[1])
        pk = Float64.(skipmissing(df[!, col]))
        length(pk) < 3 && return Float64[]
        rri = diff(pk) ./ FS .* 1000.0
    else
        df = CSV.read(path, DataFrame; header = false)
        v = Float64.(skipmissing(df[!, ncol(df) >= 2 ? 2 : 1]))
        rri = maximum(v) < 20 ? v .* 1000.0 : v       # seconds -> ms
    end
    return filter(x -> RRI_MIN < x < RRI_MAX, rri)
end

function resolve_groups(spec::String, input::String)
    if startswith(spec, "metadata:")
        md = CSV.read(replace(spec, "metadata:" => ""), DataFrame)
        idc = "Subject_ID" in names(md) ? :Subject_ID : Symbol(names(md)[1])
        return Dict(string(r[idc]) =>
                    (lowercase(string(r.Group)) == "pd" ? "PD" : "Control")
                    for r in eachrow(md))
    end
    return nothing   # directory-name or prefix resolved inline
end

GROUPMAP = resolve_groups(GROUPSPEC, INPUT)

"""(subject_id, group, path) for every subject file under `input`."""
function collect_files(input::String)
    out = Tuple{String,String,String}[]
    subdirs = filter(d -> isdir(joinpath(input, d)), readdir(input))
    dirs = isempty(subdirs) ? [("", input)] : [(d, joinpath(input, d)) for d in subdirs]
    for (dname, dpath) in dirs
        for f in sort(readdir(dpath))
            (endswith(f, ".csv") || endswith(f, ".txt")) || continue
            sub = replace(replace(replace(f, "_cleaned.csv" => ""),
                                  "_RRi.txt" => ""), r"\.(csv|txt)$" => "")
            grp = if GROUPMAP !== nothing
                get(GROUPMAP, sub, "Unknown")
            elseif GROUPSPEC == "prefix"
                startswith(sub, "E") ? "Control" :
                    (startswith(sub, "B") || startswith(sub, "C")) ? "PD" : "Other"
            else
                isempty(dname) ? "Unknown" : dname
            end
            push!(out, (sub, grp, joinpath(dpath, f)))
        end
    end
    return out
end

# ---------------------------------------------------------------- main
println("="^74)
println("  rcMSE — refined composite multiscale entropy, FIXED tolerance r")
println("="^74)
@printf("  input     : %s\n", INPUT)
@printf("  format    : %s        group: %s\n", FORMAT, GROUPSPEC)
@printf("  entropy   : %s        m = %d\n", ETYPE, M)
@printf("  r         : %.3f x SD(RRi)   — computed once, fixed across all scales\n", RFACTOR)
@printf("  scales    : 1-%d, retained while N/tau >= %d\n", MAXSCALE, MINRATIO)
@printf("  RRi guard : %.0f-%.0f ms\n", RRI_MIN, RRI_MAX)
println("="^74)

files = collect_files(INPUT)
isempty(files) && error("no subject files found under $INPUT")

curves = DataFrame(Subject = String[], Group = String[], Scales = Int[], MSE = Float64[])
summary = DataFrame(Subject = String[], Group = String[], n_beats = Int[],
                    r = Float64[], meanRR = Float64[], HR = Float64[],
                    max_tau = Int[], nAUC_1_5 = Float64[], nAUC_all = Float64[],
                    LRS_all = Float64[])

for (sub, grp, path) in files
    rri = load_rri(path, FORMAT)
    if length(rri) < 200
        @printf("  %-10s %-8s SKIP (%d usable beats)\n", sub, grp, length(rri))
        continue
    end

    # *** r computed ONCE on the original series, never per scale ***
    r = RFACTOR * std(rri)

    taus = [t for t in 1:MAXSCALE if div(length(rri), t) >= MINRATIO]
    isempty(taus) && (taus = [1])
    mse = refined_composite_multiscale_entropy(rri, M, r, ETYPE, taus)

    for (i, t) in enumerate(taus)
        push!(curves, (sub, grp, t, mse[i]))
    end

    finite5 = [mse[i] for i in eachindex(taus) if taus[i] <= 5 && isfinite(mse[i])]
    push!(summary, (sub, grp, length(rri), r, mean(rri), 60000.0 / mean(rri),
                    maximum(taus),
                    isempty(finite5) ? NaN : compute_nAUC(finite5),
                    compute_nAUC(mse),
                    length(taus) > 1 ? compute_LRS(mse, Float64.(taus)) : NaN))

    @printf("  %-10s %-8s N=%5d  r=%7.2f  tau<=%2d  S1=%7.4f  nAUC(1-5)=%7.4f\n",
            sub, grp, length(rri), r, maximum(taus), mse[1], summary[end, :nAUC_1_5])
end

mkpath(dirname(abspath(OUTCSV)))
CSV.write(OUTCSV, curves)
sumpath = replace(OUTCSV, r"\.csv$" => "_summary.csv")
CSV.write(sumpath, summary)

println("\n  curves  -> ", OUTCSV, "  (", nrow(curves), " rows)")
println("  summary -> ", sumpath, "  (", nrow(summary), " subjects)")
println("  groups  : ", join(["$k=$(count(==(k), summary.Group))"
                              for k in unique(summary.Group)], "  "))
