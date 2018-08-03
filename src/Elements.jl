module Elements

using EllipsisNotation
import Base: ndims

import ..Transforms: TransformChain

export Simplex, Element


abstract type ReferenceElement end


struct Simplex{NDim} <: ReferenceElement end

ndims(::Simplex{n}) where n = n


struct Tensor <: ReferenceElement
    terms :: Array{Simplex}
end

ndims(self::Tensor) = sum(ndims(term) for term in self.terms)


struct Element
    reference :: ReferenceElement
    index :: Int
    transform :: TransformChain
end

ndims(self::Element) = ndims(self.reference)

end # module
