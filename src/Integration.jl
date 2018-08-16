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

function integrate(func::CompiledSparseArray{T,2}, domain::Topology, npts::Int) where T
    quadrules = Dict{ReferenceElement, Tuple{Array{Float64,2}, Vector{Float64}}}(
        elem => quadrule(elem, npts) for elem in refelems(domain)
    )

    # Compute total amount of data
    totlength = prod(func.blockshape)
    nelems = length(domain)

    I = zeros(Int, totlength, nelems)
    J = zeros(Int, totlength, nelems)
    V = zeros(T, totlength, nelems)

    ikernel, dkernel = callable(func)
    for (elemid, elem) in enumerate(domain)
        (pts, wts) = quadrules[elem.reference]

        # TODO: This can surely be done better
        blockI, blockJ = ikernel(pts[1,:], elem)
        I[:, elemid] = Int[i for (i, _) in Iterators.product(blockI, blockJ)][:]
        J[:, elemid] = Int[j for (_, j) in Iterators.product(blockI, blockJ)][:]

        for i = 1:length(wts)
            data = dkernel(pts[i,:], elem)
            V[:, elemid] += data[:] .* wts[i]
        end
    end

    sparse(I[:], J[:], V[:], size(func)...)
end

end # module
