module Elements

using FastGaussQuadrature
import ..Transforms: TransformChain

export Simplex, Element, Tensor, quadrule


abstract type ReferenceElement end


struct Simplex{NDim} <: ReferenceElement end

Base.ndims(::Simplex{n}) where n = n
function quadrule(::Simplex{1}, npts::Int)
    (pts, wts) = gausslegendre(npts)
    (pts .+ 1.0)/2, wts/2
end


struct Tensor <: ReferenceElement
    terms :: Array{Simplex}
end

Base.ndims(self::Tensor) = sum(ndims(term) for term in self.terms)


struct Element
    reference :: ReferenceElement
    index :: Int
    dimcorr :: TransformChain
    transform :: TransformChain
end

Element(ref::ReferenceElement, index::Int; transform::TransformChain=(), dimcorr::TransformChain=()) =
    Element(ref, index, dimcorr, transform)

Base.ndims(self::Element) = ndims(self.reference)

end # module
