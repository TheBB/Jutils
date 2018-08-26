module Topologies

using EllipsisNotation
import Base: getindex

using ..Transforms
using ..Elements
using ..Functions

export Lagrange
export StructuredTopology
export boundary, mkbasis, refelems

abstract type Topology{N} <: AbstractArray{Element,N} end

struct Lagrange end


struct StructuredTopology{N} <: Topology{N}
    elemids :: Array{Int,N}
    dimcorr :: Tuple{Vararg{AbstractTransform}}
    parent :: Union{StructuredTopology, Nothing}
end

function StructuredTopology(shape::Vararg{Int,N}) where N
    elemids = Array{Int,N}(undef, shape...)
    elemids[:] .= 1:prod(shape)
    StructuredTopology{N}(elemids, (), nothing)
end

Base.size(self::StructuredTopology) = size(self.elemids)
Base.length(self::StructuredTopology) = length(self.elemids)
Base.IndexStyle(::Type{StructuredTopology{N}}) where N = IndexCartesian()

@inline function Base.getindex(self::StructuredTopology{N}, I::Vararg{Int,N}) where N
    @boundscheck checkbounds(self, I...)
    if self.parent == nothing
        trans = (Shift([float(i-1) for i in I]),)
        Element(Simplex{N}(), self.elemids[I...], (self.dimcorr, trans))
    else
        super = self.parent[self.elemids[I...]]
        Element(Simplex{N}(), super.index, ((self.dimcorr..., super.dimcorr...), super.transform))
    end
end

function boundary(self::StructuredTopology{N}, axis::Int, loc::Symbol) where N
    index = Any[(:) for _ in 1:N-1]
    insert!(index, axis, loc == :first ? 1 : size(self.elemids, axis))
    elemids = self.elemids[index...]
    dimcorr = Updim{N-1,axis}(loc == :first ? 0.0 : 1.0)
    StructuredTopology{N-1}(elemids, (dimcorr,), self)
end

refelems(::StructuredTopology{N}) where N = (Simplex{N}(),)

function mkbasis(self::StructuredTopology{N}, ::Type{Lagrange}, degree::Int) where N
    # Generate N single-dimensional Lagrangian bases of the right degree
    poly = Monomials(point(N), degree)
    coeffs = hcat((
        [binomial(degree,nu) * binomial(degree-nu,k-nu) * (isodd(k-nu) ? -1 : 1) for nu in 0:degree]
        for k in 0:degree
    )...)
    basis1d = Contract(Constant(coeffs), poly)

    # Formulate the tensor product
    factors = [InsertAxis(basis1d[:, k], [fill(1, k-1); fill(2, N-k)]) for k in 1:N]
    basisNd = .*(factors...)

    # Number the DoFs
    ndofs = [n*degree+1 for n in size(self.elemids)]
    dofgrid = reshape(1:prod(ndofs), ndofs...)
    elemwindows = [1+(k-1)*degree:1+k*degree for k in 1:maximum(ndofs)]
    dofmap = Array{Int,2}(undef, (degree+1)^N, length(self.elemids))
    for (i, windows) in enumerate(Iterators.product([elemwindows[1:n] for n in size(self.elemids)]...))
        dofmap[:,i] = dofgrid[windows...][:]
    end

    # Return inflation
    Inflate(reshape(basisNd, (degree+1)^N), (prod(ndofs),), Constant(dofmap)[:, elemindex])
end

end # module
