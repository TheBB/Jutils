module Integration

import SparseArrays: sparse

import ..Elements: ReferenceElement, quadrule
import ..Functions: callable, CompiledArray, CompiledSparseArray
import ..Topologies: Topology, refelems

export integrate


function integrate(func::CompiledArray{T}, domain::Topology, npts::Int) where T
    quadrules = Dict{ReferenceElement, Tuple{Array{Float64,2}, Vector{Float64}}}(
        elem => quadrule(elem, npts) for elem in refelems(domain)
    )

    result = zeros(T, size(func)...)
    kernel = callable(func)
    for elem in domain
        (pts, wts) = quadrules[elem.reference]
        for i = 1:length(wts)
            result .+= kernel(pts[i,:], elem) .* wts[i]
        end
    end

    result
end

function integrate(func::CompiledSparseArray{T,1}, domain::Topology, npts::Int) where T
    quadrules = Dict{ReferenceElement, Tuple{Array{Float64,2}, Vector{Float64}}}(
        elem => quadrule(elem, npts) for elem in refelems(domain)
    )

    result = zeros(T, length(func))
    ikernel, dkernel = callable(func)
    for elem in domain
        (pts, wts) = quadrules[elem.reference]
        blockI, = ikernel(pts[1,:], elem) :: Tuple{Vector{Int}}
        for i = 1:length(wts)
            data = dkernel(pts[i,:], elem) :: Array{T,1}
            result[blockI] += data .* wts[i]
        end
    end

    result
end

function integrate(func::CompiledSparseArray{T,2}, domain::Topology, npts::Int) where T
    quadrules = Dict{ReferenceElement, Tuple{Array{Float64,2}, Vector{Float64}}}(
        elem => quadrule(elem, npts) for elem in refelems(domain)
    )

    nelems = length(domain)
    I = zeros(Int, func.blockshape..., nelems) :: Array{Int, 3}
    J = zeros(Int, func.blockshape..., nelems) :: Array{Int, 3}
    V = zeros(T, func.blockshape..., nelems) :: Array{T, 3}

    ikernel, dkernel = callable(func)
    for (elemid, elem) in enumerate(domain)
        (pts, wts) = quadrules[elem.reference]

        blockI, blockJ = ikernel(pts[1,:], elem) :: Tuple{Vector{Int}, Vector{Int}}
        I[:, :, elemid] .= blockI
        J[:, :, elemid] .= reshape(blockJ, 1, :)

        for i = 1:length(wts)
            data = dkernel(pts[i,:], elem) :: Array{T,2}
            V[:, :, elemid] += data .* wts[i]
        end
    end

    sparse(I[:], J[:], V[:], size(func)...)
end

end # module
