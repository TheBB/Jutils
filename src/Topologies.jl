module Topologies

using ..Transforms
using ..Elements
import Base: getindex

export Line

abstract type Topology{N} <: AbstractArray{Element,N} end


struct Line <: Topology{1}
    nelems :: Int
end

Base.size(self::Line) = (self.nelems,)
Base.IndexStyle(::Type{Line}) = IndexLinear()

@inline function Base.getindex(self::Line, i::Int)
    @boundscheck checkbounds(self, i)
    Element(Simplex{1}(), (Shift([float(i) - 1]),), i)
end

end # module
