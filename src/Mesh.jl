module Mesh

using ..Functions
using ..Topologies

export line

line(nelems::Int) = (Line(nelems), rootcoods(1))

function line(nodes::StepRangeLen{T}) where T <: Real
    offset = nodes[1]
    nelems = length(nodes) - 1
    scale = (nodes[end] - offset) / nelems
    (Line(nelems), rootcoords(1) * scale + offset)
end

end # module
