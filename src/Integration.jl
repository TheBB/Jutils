module Integration

import ..Elements: ReferenceElement, quadrule
import ..Functions: CompiledDenseArrayFunction
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

end # module
