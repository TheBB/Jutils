module Integration

import SparseArrays: sparse

import ..Elements: ReferenceElement, quadrule
import ..Functions: CompiledDenseArrayFunction, CompiledSparseArrayFunction
import ..Topologies: Topology, Line, refelems

export integrate


function integrate(func::CompiledDenseArrayFunction{T}, domain::Topology, npts::Int) where T
    quadrules = Dict{ReferenceElement, Tuple{Vector{Float64}, Vector{Float64}}}(
        elem => quadrule(elem, npts) for elem in refelems(domain)
    )

    result = zeros(T, size(func)...)
    for elem in domain
        (pts, wts) = quadrules[elem.reference]
        for i = 1:length(wts)
            result .+= func([pts[i]], elem) .* wts[i]
        end
    end

    result
end

function integrate(func::CompiledSparseArrayFunction{T,2}, domain::Topology, npts::Int) where T
    quadrules = Dict{ReferenceElement, Tuple{Vector{Float64}, Vector{Float64}}}(
        elem => quadrule(elem, npts) for elem in refelems(domain)
    )

    # Compute total amount of data
    blocklengths = [prod(shape) for shape in func.blockshapes]
    blockindices = [cs-l+1:l for (l,cs) in zip(blocklengths, cumsum(blocklengths))]
    totlength = sum(blocklengths)
    nelems = length(domain)

    I = zeros(Int, totlength, nelems)
    J = zeros(Int, totlength, nelems)
    V = zeros(T, totlength, nelems)

    for (elemid, elem) in enumerate(domain)
        (pts, wts) = quadrules[elem.reference]

        for (ix, (blockI, blockJ)) in zip(blockindices, func.indices([pts[1]], elem))
            I[ix, elemid] = Int[i for (i, _) in Iterators.product(blockI, blockJ)][:]
            J[ix, elemid] = Int[j for (_, j) in Iterators.product(blockI, blockJ)][:]
        end

        for i = 1:length(wts)
            data = func.data([pts[i]], elem)
            for (ix, d) in zip(blockindices, data)
                V[ix, elemid] .+= d[:] .* wts[i]
            end
        end
    end

    sparse(I[:], J[:], V[:])
end

end # module
