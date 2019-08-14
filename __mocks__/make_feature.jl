
# https://juliadata.github.io/DataFrames.jl/stable/man/joins/#Database-Style-Joins-1

using DataFrames
using CSV


df = CSV.read("input/structures.csv")


join(a, b, on = [(:col, :col1), (:)]


