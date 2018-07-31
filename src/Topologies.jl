module Topologies

using EllipsisNotation
import Base: getindex

using ..Transforms
using ..Elements
import ..Functions: Elementwise, Monomials, elemindex, Points, Constant, Matmul

export Line, Lagrange, basis

abstract type Topology{N} <: AbstractArray{Element,N} end
struct Lagrange end


struct Line <: Topology{1}
    nelems :: Int
end

Base.size(self::Line) = (self.nelems,)
Base.IndexStyle(::Type{Line}) = IndexLinear()

@inline function Base.getindex(self::Line, i::Int)
    @boundscheck checkbounds(self, i)
    Element(Simplex{1}(), i, Shift([float(i) - 1]))
end

function basis(self::Line, ::Type{Lagrange}, degree::Int)
    dofs = 1 .+ hcat((range(elemid*degree, length=degree+1) for elemid in range(0, length=self.nelems))...)
    dofmap = Elementwise(dofs, elemindex)
    poly = Monomials(Points{Float64,1}(), degree)
    coeffs = hcat([[binomial(degree,nu) * binomial(degree-nu,k-nu) * (isodd(k-nu) ? -1 : 1) for nu in 0:degree] for k in 0:degree]...)
    Matmul(Constant(coeffs), poly)
end

end # module
