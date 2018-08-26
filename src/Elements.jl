module Elements

using FastGaussQuadrature
import Base.Iterators: product

import ..Transforms: TransformChain

export Simplex, Element, Tensor, quadrule


abstract type ReferenceElement end


struct Simplex{NDim} <: ReferenceElement end

Base.ndims(::Simplex{n}) where n = n

quadrule(::Simplex{0}, npts::Int) = zeros(Float64, 1, 0), [1.0]

function quadrule(::Simplex{N}, npts::Int) where N
    (pts, wts) = gausslegendre(npts)
    pts = (pts .+ 1.0) ./ 2
    wts = wts ./ 2

    pts = hcat([collect(k) for k in product((pts for _ in 1:N)...)]...)
    wts = .*((reshape(wts, fill(1, k)..., npts) for k in 0:N-1)...)
    pts', vec(wts)
end


struct Tensor <: ReferenceElement
    terms :: Array{Simplex}
end

Base.ndims(self::Tensor) = sum(ndims(term) for term in self.terms)


struct Element
    reference :: ReferenceElement
    index :: Int
    transform :: Tuple{TransformChain, TransformChain}
end

Element(ref::ReferenceElement, index::Int; transform::TransformChain=(), dimcorr::TransformChain=()) =
    Element(ref, index, (dimcorr, transform))

Base.ndims(self::Element) = ndims(self.reference)

end # module
