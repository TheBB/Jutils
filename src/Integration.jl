module Integration

import ..Elements: ReferenceElement, quadrule
import ..Functions: CompiledArrayFunction
import ..Topologies: Topology, Line, refelems

export integrate_dense


function integrate_dense(func::CompiledArrayFunction{T}, domain::Topology, npts::Int) where T
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
