module Mesh

using ..Functions
using ..Topologies

export line, rectilinear

line(nelems::Int) = (StructuredTopology(nelems), rootcoords(1))

function line(nodes::StepRangeLen{T}) where T <: Real
    offset = nodes[1]
    nelems = length(nodes) - 1
    scale = (nodes[end] - offset) / nelems
    (StructuredTopology(nelems), rootcoords(1) * scale + offset)
end

function rectilinear(nelems::Vararg{Int,N}) where N
    (StructuredTopology(nelems...), rootcoords(N))
end

end # module
