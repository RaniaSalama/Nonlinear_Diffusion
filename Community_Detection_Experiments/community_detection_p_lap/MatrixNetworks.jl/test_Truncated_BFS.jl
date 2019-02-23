# This code is taken from https://github.com/dgleich/MatrixNetworks.jl

include("Truncated_BFS.jl")
using TruncatedBFS

A = sparse(convert(Array{Int64,1}, [2;3;4;5;6;7;8;9;10;11;12;13;14;15;16]), convert(Array{Int64,1}, [1;1;1;1;1;2;2;3;3;4;4;5;5;6;6]), convert(Array{Int64,1},[1;1;1;1;1;1;1;1;1;1;1;1;1;1;1]), 16, 16)
println(A)
start_node = 1
Ssize =12
S = TruncatedBFS.runTruncatedBFS(start_node, A, Ssize);
println(S)