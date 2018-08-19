module Functions

import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful
import LinearAlgebra: Diagonal, I
import OrderedCollections: OrderedDict

import TensorOperations.@tensor

import ..Transforms: TransformChain
import ..Elements: Element

# Base exports
export ast, callable, compile, isconstant, iselconstant, restype, typetree
export CompiledArray, CompiledSparseArray, Compiled

# Type exports
export Shape, Index, Indices
export Add, ApplyTransform, ApplyTransformGrad
export Constant, Contract, GetIndex
export Inflate, InsertAxis, Inv
export Monomials, Neg, Outer, Product, Reshape, Tupl, Zeros

# Constructor exports
export dimtrans, fulltrans, rawpoint, point
export grad, elemindex, rootcoords, outer


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
