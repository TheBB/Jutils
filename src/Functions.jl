module Functions

import OrderedCollections: OrderedDict
import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful

import ..Transforms: TransformChain
import ..Elements: Element

# Base exports
export ast, compile, isconstant, iselconstant, optimize, restype
export CompiledDenseArrayFunction, CompiledFunction

# Type exports
export Shape, Index, Indices
export ApplyTransform, Constant, GetIndex, Inflate, InsertAxis, Matmul, Monomials, Outer, Point, Product, Reshape, Sum, Tupl

# Constructor exports
export inflate, insertaxis
export elemindex, trans, rootcoords


# Some typedefs
Shape = Tuple{Vararg{Int}}

include("Functions/bases.jl")

# More typedefs
Index = Union{ArrayEvaluable{Int}, Colon}
Indices = Vector{Index}

include("Functions/utils.jl")
include("Functions/definitions.jl")
include("Functions/constructors.jl")

end # module
