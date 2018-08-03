module Functions

import OrderedCollections: OrderedDict
import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful
import Base: convert, size, ndims

import ..Transforms: TransformChain
import ..Elements: Element

export ast, generate, restype
export elemindex, trans, Point
export ApplyTransform, Constant, GetItem, Inflate, Matmul, Monomials, Outer, Product, Sum
export rootcoords


include("Functions/bases.jl")
include("Functions/utils.jl")



# Generic evaluation arguments

struct Argument{T} <: Evaluable{T}
    expression :: Union{Symbol, Expr}
end

Base.show(io::IO, self::Argument) = print(io, "Argument($(self.expression))")
arguments(::Argument) where T = ()
prealloc(::Argument) = []
codegen(self::Argument) = self.expression

# Element transformation
const trans = Argument{TransformChain}(:(element.transform))

# Element index
const elemindex = Argument{Int}(:(element.index))



# Quadrature point (reference coordinates)

struct Point{N} <: ArrayEvaluable{Float64,1} end

arguments(self::Point) = ()
size(self::Point{N}) where {T,N} = (N::Int,)
prealloc(::Point) = []
codegen(self::Point) = :point



# ApplyTransform

struct ApplyTransform <: ArrayEvaluable{Float64,1}
    trans :: Evaluable{TransformChain}
    arg :: ArrayEvaluable{Float64,1}
    dims :: Int
end

arguments(self::ApplyTransform) = (self.trans, self.arg)
size(self::ApplyTransform) = (self.dims,)
prealloc(self::ApplyTransform) = []
codegen(self::ApplyTransform, trans, arg) = :(applytrans($arg, $trans))



# Constant

struct Constant{T,N} <: ArrayEvaluable{T,N}
    value :: Array{T,N}
end

# Scalars are wrapped as zero-dimensional arrays
Constant(v::T) where T<:Number = Constant(fill(v, ()))

arguments(::Constant) = ()
size(self::Constant) = size(self.value)
prealloc(self::Constant) = [self.value]
codegen(::Constant, alloc) = alloc



# GetItem
# TODO: More general indexing expressions?

struct GetItem{T,N} <: ArrayEvaluable{T,N}
    value :: ArrayEvaluable
    indices :: Tuple

    function GetItem(value::ArrayEvaluable{T,N}, indices::Tuple) where {T,N}
        legalindices(indices) || error("Illegal indexing expression")
        length(indices) == ndims(value) || error("Inconsistent indexing")

        # Attempt type inference. TODO: Make this more robust
        indextypes = [isa(i, Evaluable) ? restype(i) : typeof(i) for i in indices]
        (_, rtype) = code_typed(getindex, (Array{T,N}, indextypes...))[end]

        rtype <: Array || error("Result of indexing must be an array")
        rtype.isconcretetype || error("Result of indexing is not a concrete type")
        (RT, RN) = rtype.parameters
        new{RT,RN}(value, indices)
    end
end

arguments(self::GetItem) = (self.value, (i for i in self.indices if isa(i, Evaluable))...)
size(self::GetItem) = resultsize(size(self.value), self.indices)
prealloc(self::GetItem) = []

function codegen(self::GetItem, value, varindices...)
    weaved = []
    varindices = collect(varindices)
    for s in self.indices
        if isa(s, Colon)
            push!(weaved, :(:))
        elseif isa(s, Int) || isa(s, Array{Int})
            push!(weaved, :($s))
        else
            push!(weaved, popfirst!(varindices))
        end
    end
    :(view($value, $(weaved...)))
end



# Inflate

struct Inflate{T,N} <: ArrayEvaluable{T,N}
    data :: ArrayEvaluable{T}
    indices :: Tuple{Vararg{ArrayEvaluable{Int,1}}}
    size :: Tuple{Vararg{Int}}

    function Inflate(data::ArrayEvaluable{T}, indices, shape) where T
        legalindices(indices) || error("Illegal indexing expression")
        length(indices) == length(shape) || error("Inconsistent indexing")
        resultsize(shape, indices) == size(data) || error("Inconsistent dimensions")
        new{T,length(shape)}(data, indices, shape)
    end
end

arguments(self::Inflate) = (self.data, (i for i in self.indices if isa(i, Evaluable))...)
size(self::Inflate) = self.size
prealloc(self::Inflate{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Inflate{T}, data, indices...) where T
    target, varindices = indices[end], collect(indices[1:end-1])
    weaved = indexweave(self.indices, varindices)
    quote
        $target[:] .= $(zero(T))
        $target[$(weaved...)] = $data
        $target
    end
end



# Matmul
# TODO: Generalize using TensorOperations.jl when compatible

struct Matmul{T,L,R,N} <: ArrayEvaluable{T,N}
    left :: ArrayEvaluable{L}
    right :: ArrayEvaluable{R}

    function Matmul(left::ArrayEvaluable{L}, right::ArrayEvaluable{R}) where {L <: Number, R <: Number}
        ndims(left) > 0 || error("Expected at least a one-dimensional array")
        ndims(right) > 0 || error("Expected at least a one-dimensional array")
        size(left, ndims(left)) == size(right, 1) || error("Inconsistent dimensions for contraction")
        new{promote_type(L,R), L, R, ndims(left) + ndims(right) - 2}(left, right)
    end
end

arguments(self::Matmul) = (self.left, self.right)
size(self::Matmul) = (size(self.left)[1:end-1]..., size(self.right)[2:end]...)
prealloc(self::Matmul{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Matmul{T}, left, right, result) where T
    (i, j, k) = gensym("i"), gensym("j"), gensym("k")
    quote
        $result[:] .= $(zero(T))
        for $i = CartesianIndices($(size(self.left)[1:end-1])), $j = CartesianIndices($(size(self.right)[2:end]))
            @simd for $k = 1:$(size(self.right, 1))
                $result[$i,$j] += $left[$i,$k] * $right[$k,$j]
            end
        end
        $result
    end
end



# Monomials

struct Monomials{T,N} <: ArrayEvaluable{T,N}
    points :: ArrayEvaluable{T}
    degree :: Int

    function Monomials(points::ArrayEvaluable{T,M}, degree::Int) where {T,M}
        new{T,M+1}(points, degree)
    end
end

arguments(self::Monomials) = (self.points,)
size(self::Monomials) = (self.degree + 1, size(self.points)...)
prealloc(self::Monomials{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Monomials, points, target)
    j = gensym("j")

    loopcode = Any[:($target[1,$j] = 1.0)]
    for i in 1:self.degree
        push!(loopcode, :($target[$(i+1),$j] = $target[$i,$j] * $points[$j]))
    end

    Expr(:block, Expr(:for, :($j = CartesianIndices(size($points))), Expr(:block, loopcode...)), target)
end



# Outer

struct Outer{T,N} <: ArrayEvaluable{T,N}
    left :: ArrayEvaluable
    right :: ArrayEvaluable

    function Outer(left::ArrayEvaluable{L}, right::ArrayEvaluable{R}) where {L,R}
        ndims(left) > 0 || error("Expected at least a one-dimensional array")
        ndims(right) > 0 || error("Expected at least a one-dimensional array")
        size(left)[2:end] == size(right)[2:end] || error("Inconsistent dimensions")
        new{promote_type(L,R), ndims(left)+1}(left, right)
    end
end

arguments(self::Outer) = (self.left, self.right)
size(self::Outer) = (size(self.left, 1), size(self.right, 1), size(self.left)[2:end]...)
prealloc(self::Outer{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Outer, left, right, target)
    (i, j) = gensym("i"), gensym("j")
    nleft = size(self.left, 1)
    nright = size(self.right, 1)
    colons = Tuple(Iterators.repeated(:, ndims(self) - 2))
    quote
        for $i = 1:$nleft, $j = 1:$nright
            $target[$i,$j,$(colons...)] = $left[$i,$(colons...)] .* $right[$j,$(colons...)]
        end
        $target
    end
end



# Product

struct Product{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Product(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (arraytype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Product(terms...) = Product(terms)

arguments(self::Product) = self.terms
size(self::Product) = broadcast_shape((size(term) for term in self.terms)...)
prealloc(self::Product{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

# Note: element-wise product!
function codegen(self::Product, args...)
    code = args[1]
    for arg in args[2:end-1]
        code = :($code .* $arg)
    end
    :($(args[end])[:] = $code; $(args[end]))
end



# Sum

struct Sum{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Sum(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (arraytype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Sum(terms...) = Sum(terms)

arguments(self::Sum) = self.terms
size(self::Sum) = broadcast_shape((size(term) for term in self.terms)...)
prealloc(self::Sum{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Sum, args...)
    code = args[1]
    for arg in args[2:end-1]
        code = :($code .+ $arg)
    end
    :($(args[end])[:] = $code; $(args[end]))
end



# Miscellaneous

rootcoords(ndims::Int) = ApplyTransform(trans, Point{ndims}(), ndims)

end # module
