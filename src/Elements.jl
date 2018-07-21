module Elements

using EllipsisNotation
import Base: ndims
import Base.Iterators: repeated

import ..Transforms: AbstractTransform, apply

abstract type AbstractElement end
abstract type ReferenceElement end


struct Simplex{NDim} <: ReferenceElement end

ndims(::Simplex{n}) where n = n
nvertices(::Simplex{n}) where n = n+1
vertices(::Simplex{1}) = [0.0 1.0]
vertices(::Simplex{2}) = [0.0 1.0 0.0; 0.0 0.0 1.0]
vertices(::Simplex{3}) = [0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]


struct Tensor <: ReferenceElement
    terms :: Array{Simplex}
end

ndims(self::Tensor) = sum(ndims(term) for term in self.terms)
nvertices(self::Tensor) = prod(nvertices(term) for term in self.terms)
function vertices(self::Tensor)
    ret = Array{Float64}(undef, ndims(self), (nvertices(term) for term in self.terms)...)
    rootdim = 1
    for (i, term) in enumerate(self.terms)
        verts = vertices(term)
        verts = reshape(verts, size(verts)[1], repeated(1, i-1)..., size(verts)[2:end]...)
        ret[rootdim:rootdim+ndims(term)-1, ..] .= verts
        rootdim += ndims(term)
    end
    reshape(ret, ndims(self), nvertices(self))
end


struct Element <: AbstractElement
    reference :: ReferenceElement
    transform :: Tuple{Vararg{AbstractTransform}}
end

Element(ref::ReferenceElement, transforms::AbstractTransform...) = Element(ref, transforms)

ndims(self::Element) = ndims(self.reference)
nvertices(self::Element) = nvertices(self.reference)
vertices(self::Element) = apply(vertices(self.reference), self.transform...)

end # module
