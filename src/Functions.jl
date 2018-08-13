module Functions

import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful
import LinearAlgebra: Diagonal, I
import OrderedCollections: OrderedDict

import TensorOperations.@tensor

import ..Transforms: TransformChain
import ..Elements: Element

# Base exports
export ast, callable, compile, isconstant, iselconstant, optimize, restype, typetree
export CompiledArray, CompiledSparseArray, Compiled

# Type exports
export Shape, Index, Indices
export ApplyTransform, ApplyTransformGrad
export Constant, Contract, GetIndex
export Inflate, InsertAxis, Inv
export Monomials, Neg, Outer, Point, Product, Reshape, Sum, Tupl, Zeros

# Constructor exports
export @contract
export grad, elemindex, trans, rootcoords, outer


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
