module DiffusionUtils
export LoadGraph

function LoadGraph(dataFilename::AbstractString)
    (rows,header) = readdlm(dataFilename;header=true)
	rowVal = convert(Array{Int64,1},rows[1:parse(Int,header[3]),1])+1
	colVal = convert(Array{Int64,1},rows[1:parse(Int,header[3]),2])+1
	# As symmtric.
    A = sparse(vcat(rowVal, colVal),
               vcat(colVal, rowVal),
               convert(Array{Float64,1},ones(2*parse(Int,header[3]))),
               parse(Int,header[1]), 
               parse(Int,header[2])
               )
    return A
end

function LoadCommunities(dataFilename::AbstractString)
    file = open(dataFilename)
    rows = Int64[]
    cols = Int64[]
    data = Int64[]
    commCount = 1
    for line in eachline(file)
            community = split(line)
            for member in community
                push!(rows, parse(Int,member) + 1)
                push!(cols, commCount)
                push!(data, 1)
            end
            commCount = commCount + 1
    end
    close(file)
    C = sparse(rows, cols, data)
    return C
end

end