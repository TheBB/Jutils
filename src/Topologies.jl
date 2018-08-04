module Topologies

using EllipsisNotation
import Base: getindex

using ..Transforms
using ..Elements
using ..Functions

export Line, Lagrange, basis, refelems

abstract type Topology{N} <: AbstractArray{Element,N} end
struct Lagrange end


struct Line <: Topology{1}
    nelems :: Int
end

Base.size(self::Line) = (self.nelems,)
Base.IndexStyle(::Type{Line}) = IndexLinear()

@inline function Base.getindex(self::Line, i::Int)
    @boundscheck checkbounds(self, i)
    Element(Simplex{1}(), i, (Shift([float(i) - 1]),))
end

refelems(::Line) = (Simplex{1}(),)

function basis(self::Line, ::Type{Lagrange}, degree::Int)
    dofs = 1 .+ cat((range(elemid*degree, length=degree+1) for elemid in range(0, length=self.nelems))..., dims=2)
    ndofs = self.nelems * degree + 1
    dofmap = GetIndex(Constant(dofs), (:, elemindex))

    poly = Monomials(GetIndex(Point{1}(), (Constant(1),)), degree)
    coeffs = hcat((
        [binomial(degree,nu) * binomial(degree-nu,k-nu) * (isodd(k-nu) ? -1 : 1) for nu in 0:degree]
        for k in 0:degree
    )...)

    Inflate(Matmul(Constant(coeffs), poly), (dofmap,), (ndofs,))
end

end # module
