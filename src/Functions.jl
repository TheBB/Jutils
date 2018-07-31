module Functions

import OrderedCollections: OrderedDict
import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful
import Base: convert, size, ndims

import ..Transforms: TransformChain
import ..Elements: Element

export ast, generate, restype
export rootcoords, trans, elemindex
export Constant, Elementwise, Matmul, Monomials, Polynomials, Product, Sum


abstract type AbstractFunction end

ast(self::AbstractFunction) = ast!(self, Set{AbstractFunction}(), dependencies(self))
function ast!(self::AbstractFunction, seen::Set{AbstractFunction}, indices::OrderedDict{AbstractFunction,Int})
    self in seen && return string("%", string(indices[self]))

    s = ""
    args = Stateful(arguments(self))
    for arg in args
        bridge = isdone(args) ? flatten((("└ ",), repeated("  "))) : flatten((("├ ",), repeated("│ ")))
        sublines = split(ast!(arg, seen, indices), "\n")
        subtree = join((string(b, s) for (b, s) in zip(bridge, sublines)), "\n")
        s = string(s, "\n", subtree)
    end

    push!(seen, self)
    string("%", string(indices[self]), " = ", repr(self), s)
end

function dependencies(self::AbstractFunction)
    indices = OrderedDict{AbstractFunction,Int}()
    dependencies!(indices, self)
    indices
end

function dependencies!(indices::OrderedDict{AbstractFunction,Int}, self::AbstractFunction)
    haskey(indices, self) && return
    for func in arguments(self)
        dependencies!(indices, func)
    end
    indices[self] = length(indices) + 1
end

function generate(func::AbstractFunction)
    indices = dependencies(func)
    symbols = Dict{AbstractFunction,Symbol}(func => gensym(string(index)) for (func, index) in indices)

    code = Array{Expr,1}()
    for (func, index) in indices
        argsyms = [symbols[arg] for arg in arguments(func)]
        tgtsym = symbols[func]
        push!(code, :($tgtsym = $(codegen(func, argsyms...))))
    end
    code = Expr(:block, code...)

    typeinfo = [(:points, Array{Float64,2}), (:element, Element)]
    paramlist = Expr(:parameters, (:($sym::$tp) for (sym, tp) in typeinfo)...)
    prototype = Expr(:call, :evaluate, paramlist)
    definition = Expr(:function, prototype, code)

    mod = Module()
    Core.eval(mod, :(import Jutils.Elements: Element))
    Core.eval(mod, :(import Jutils.Transforms: applytrans))
    Core.eval(mod, :(using Jutils.Runtime))
    Core.eval(mod, :(using EllipsisNotation))
    Core.eval(mod, definition)
    return mod.evaluate
end



# Generic evaluation arguments

struct Argument{T} <: AbstractFunction
    expression :: Union{Symbol, Expr}
end

Base.show(io::IO, self::Argument) = print(io, "Argument($(self.expression))")
arguments(::Argument) where T = ()
codegen(self::Argument) = self.expression
restype(self::Argument{T}) where T = T

# Element transformation
const trans = Argument{TransformChain}(:(element.transform))

# Element index
const elemindex = Argument{Int}(:(element.index))



# Array functions

abstract type AbstractArrayFunction{T,N} <: AbstractFunction end

asarray(v::AbstractArrayFunction) = v
asarray(v::Real) = Constant(v)
asarray(v::AbstractArray) = Constant(v)

Base.ndims(self::AbstractArrayFunction{T,N}) where {T,N} = N :: Int
Base.show(io::IO, self::AbstractArrayFunction) = print(io, string(typeof(self).name.name), size(self))
Base.:+(self::AbstractArrayFunction, rest...) = Sum(self, (asarray(v) for v in rest)...)
Base.:*(self::AbstractArrayFunction, rest...) = Product(self, (asarray(v) for v in rest)...)
restype(self::AbstractArrayFunction{T,N}) where {T,N} = Array{T,N}



# Quadrature points (reference coordinates)

struct Points{T,N} <: AbstractArrayFunction{T,1} end

arguments(self::Points) = ()
size(self::Points{T,N}) where {T,N} = (N::Int,)
codegen(self::Points) = :points



# ApplyTransform

struct ApplyTransform <: AbstractArrayFunction{Float64,2}
    trans :: AbstractFunction
    arg :: AbstractFunction
    dims :: Int

    function ApplyTransform(trans, arg, dims)
        restype(trans) == TransformChain || error("Transformation must be TransformChain")
        restype(arg) == Array{Float64,1} || error("Argument must be vector")
        new(trans, arg, dims)
    end
end

arguments(self::ApplyTransform) = (self.trans, self.arg)
size(self::ApplyTransform) = (self.dims,)
codegen(self::ApplyTransform, trans, arg) = :(applytrans($arg, $(trans)...))



# Constant

struct Constant{T,N} <: AbstractArrayFunction{T,N}
    value :: Array{T,N}
end

# Scalars are wrapped as zero-dimensional arrays
Constant(v::T) where T<:Number = (wrap = Array{T,0}(undef); wrap[] = v; Constant(wrap))

arguments(::Constant) = ()
size(self::Constant) = size(self.value)
codegen(self::Constant) = reshape(self.value, size(self.value)..., 1)



# Elementwise

struct Elementwise{T,N} <: AbstractArrayFunction{T,N}
    data :: Array{T}
    index :: AbstractFunction

    function Elementwise(data::Array{T}, index) where T
        ndims(data) > 0 || error("Expected at least a one-dimensional array")
        restype(index) == Int || error("Index must be integer")
        new{T, ndims(data)-1}(data, index)
    end
end

arguments(self::Elementwise) = (self.index,)
size(self::Elementwise) = size(self.data)[1:end-1]

# Use a range for the last index to generate a dummy dimension for broadcasting
codegen(self::Elementwise, index) = :($(self.data)[.., $index:$index])



# Matmul
# TODO: For now, this only does a single tensor contraction

struct Matmul{T,L,R,N} <: AbstractArrayFunction{T,N}
    left :: AbstractArrayFunction{L}
    right :: AbstractArrayFunction{R}

    function Matmul(left::AbstractArrayFunction{L}, right::AbstractArrayFunction{R}) where {L <: Number, R <: Number}
        ndims(left) > 0 || error("Expected at least a one-dimensional array")
        ndims(right) > 0 || error("Expected at least a one-dimensional array")
        size(left)[end] == size(right)[1] || error("Inconsistent dimensions for contraction")
        new{promote_type(L, R), L, R, ndims(left) + ndims(right) - 2}(left, right)
    end
end

arguments(self::Matmul) = (self.left, self.right)
size(self::Matmul) = (size(self.left)[1:end-1]..., size(self.right)[2:end]...)

function codegen(self::Matmul, left, right)
    (value, l, r, j) = gensym("value"), gensym("l"), gensym("r"), gensym("j")
    (lrange, rrange) = gensym("lrange"), gensym("rrange"), gensym("trange")
    code = :(matmul($left, $right))
end



# Polynomials

struct Monomials{T,N} <: AbstractArrayFunction{T,N}
    points :: AbstractArrayFunction{T}
    degree :: Int

    function Monomials(points::AbstractArrayFunction{T,M}, degree::Int) where {T,M}
        new{T,M+1}(points, degree)
    end
end

arguments(self::Monomials) = (self.points,)
size(self::Monomials) = (self.degree + 1, size(self.points)...)

function codegen(self::Monomials, points)
    (value, j) = gensym("value"), gensym("j")
    code = Any[:($value = zeros(Float64, $(self.degree+1), size($points)...))]

    loopcode = Any[:($value[1,$j] = 1.0)]
    for i in 1:self.degree
        push!(loopcode, :($value[$(i+1),$j] = $value[$i,$j] * $points[$j]))
    end
    push!(code, Expr(:for, :($j = CartesianIndices(size($points))), Expr(:block, loopcode...)))

    push!(code, :($value))
    Expr(:block, code...)
end



# Product

struct Product{T,N} <: AbstractArrayFunction{T,N}
    terms :: Tuple{Vararg{AbstractArrayFunction}}

    function Product(terms::Tuple{Vararg{AbstractArrayFunction}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        new{reduce(promote_type, (restype(term) for term in terms)), length(newshape)}(terms)
    end
end

Product(terms::AbstractArrayFunction...) = Product(terms)

arguments(self::Product) = self.terms
size(self::Product) = broadcast_shape((size(term) for term in self.terms)...)

# Note: element-wise product!
function codegen(self::Product, args...)
    code = args[1]
    for arg in args[2:end]
        code = :($code .* $arg)
    end
    code
end



# Sum

struct Sum{T,N} <: AbstractArrayFunction{T,N}
    terms :: Tuple{Vararg{AbstractArrayFunction}}

    function Sum(terms::Tuple{Vararg{AbstractArrayFunction}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        new{reduce(promote_type, (restype(term) for term in terms)), length(newshape)}(terms)
    end
end

Sum(terms::AbstractArrayFunction...) = Sum(terms)

arguments(self::Sum) = self.terms
size(self::Sum) = broadcast_shape((size(term) for term in self.terms)...)
function codegen(self::Sum, args...)
    code = args[1]
    for arg in args[2:end]
        code = :($code .+ $arg)
    end
    code
end



# Miscellaneous

rootcoords(ndims::Int) = ApplyTransform(trans, points{Float64,2}, ndims)

end # module
