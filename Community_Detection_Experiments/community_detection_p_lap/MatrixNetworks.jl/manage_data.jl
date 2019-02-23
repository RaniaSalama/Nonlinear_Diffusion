# This code is taken from https://github.com/dgleich/MatrixNetworks.jl

#using Lint

function readSMAT(filename::AbstractString)
    (rows,header) = readdlm(filename;header=true)
    A = sparse(
               convert(Array{Int64,1},rows[1:parse(Int,header[3]),1])+1, 
               convert(Array{Int64,1},rows[1:parse(Int,header[3]),2])+1, 
               convert(Array{Int64,1},rows[1:parse(Int,header[3]),3]),
               parse(Int,header[1]), 
               parse(Int,header[2])
               )
    return A
end

type MatrixNetworkMetadata
    A::SparseMatrixCSC{Int64,Int64}
    labels::Vector{AbstractString}
    xy::Array{Float64,2}
    source::AbstractString
end

function load_matrix_network_all(name::AbstractString)
    A = load_matrix_network(name)
    pathname = joinpath(Pkg.dir("MatrixNetworks"),"data")
    
    meta_source = joinpath(pathname,"$(name).source")
    if isfile(meta_source)
        source = open(readall, meta_source)
    else
        source = "(None given)"
    end
    
    meta_xy = joinpath(pathname,"$(name).xy")
    if isfile(meta_xy)
        xy = readdlm(meta_xy)
    else
        xy = zeros(0,2)
    end
    
    meta_labels = joinpath(pathname,"$(name).labels")
    if isfile(meta_labels)
        labels = open(readlines, meta_labels)
    else
        labels = map(x -> @sprintf("%i",x), 1:size(A,1))
    end

    return MatrixNetworkMetadata(A,labels,xy,source)
end


function load_matrix_network(name::AbstractString)
    pathname = joinpath(Pkg.dir("MatrixNetworks"),"data")
    smatfile = joinpath(pathname,"$(name).smat")
    if isfile(smatfile)
        return readSMAT(smatfile)
    else
        error(@sprintf "The example datafile '%s' does not seem to exist where it should\n" name)
    end
end

function load_matrix_network_metadata(name::AbstractString)
    pathname = joinpath(Pkg.dir("MatrixNetworks"),"data")
    smatfile = joinpath(pathname,"$(name).smat")
    meta_xy = joinpath(pathname,"$(name).xy")
    meta_labels = joinpath(pathname,"$(name).labels")
    if isfile(smatfile)
        if isfile(meta_xy) && isfile(meta_labels)
            xy = readdlm(meta_xy)
            labels = readdlm(meta_labels)
            return (readSMAT(smatfile),xy,labels)
        end
    else
        error(@sprintf "The example datafile '%s' does not seem to exist where it should\n" name)
    end
end

function matrix_network_datasets()
    datasets_location = joinpath(Pkg.dir("MatrixNetworks"),"data")
    content = readdir(datasets_location)
    smat_files = filter(x->contains(x,".smat"),content)
    for i = 1:length(smat_files)
        # no need for this lintpragma anymore
        # @lintpragma( "Ignore use of undeclared variable end")
        smat_files[i] = smat_files[i][1:end-5]
    end
    return smat_files
end

