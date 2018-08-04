module Functions

import OrderedCollections: OrderedDict
import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful

import ..Transforms: TransformChain
import ..Elements: Element

export ast, generate, isconstant, iselconstant, restype, simplify
export CompiledArrayFunction, CompiledFunction
export elemindex, trans, Point
export ApplyTransform, Constant, GetIndex, Inflate, Matmul, Monomials, Outer, Product, Reshape, Sum
export rootcoords

include("Functions/bases.jl")
include("Functions/utils.jl")
include("Functions/definitions.jl")
include("Functions/constructors.jl")

end # module
