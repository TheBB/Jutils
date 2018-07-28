module Runtime

import Base.Iterators: repeated
import LinearAlgebra: dot

export matmul


function matmul(L::Array{TL}, R::Array{TR}) where {TL, TR}
    ntpts = max(size(L, ndims(L)), size(R, ndims(R)))
    nlpts = size(L, ndims(L)) == 1 ? repeated(1) : (1:size(L, ndims(L)))
    nrpts = size(R, ndims(R)) == 1 ? repeated(1) : (1:size(R, ndims(R)))

    lsize = size(L)[1:end-2]
    rsize = size(R)[2:end-1]
    result = zeros(promote_type(TL, TR), (lsize..., rsize..., ntpts))

    nk = size(L, ndims(L) - 1)

    for (lp, rp, tp) in zip(nlpts, nrpts, 1:ntpts)
        for ri in CartesianIndices(rsize)
            for li in CartesianIndices(lsize)
                result[li, ri, tp] = dot(view(L, li, :, lp), view(R, :, ri, rp))
            end
        end
    end

    result
end

end # module
