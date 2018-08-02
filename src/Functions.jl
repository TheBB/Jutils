module Functions

import OrderedCollections: OrderedDict
import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful
import Base: convert, size, ndims

import ..Transforms: TransformChain
import ..Elements: Element

export ast, generate, restype
export elemindex, trans, Point
export ApplyTransform, Constant, Elementwise, Matmul, Monomials, Product, Sum


abstract type Evaluable{T} end

restype(self::Evaluable{T}) where T = T

ast(self::Evaluable) = ast!(self, Set{Evaluable}(), dependencies(self))

function ast!(self::Evaluable, seen::Set{Evaluable}, indices::OrderedDict{Evaluable,Int})
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

function dependencies(self::Evaluable)
    indices = OrderedDict{Evaluable,Int}()
    dependencies!(indices, self)
    indices
end

function dependencies!(indices::OrderedDict{Evaluable,Int}, self::Evaluable)
    haskey(indices, self) && return
    for func in arguments(self)
        dependencies!(indices, func)
    end
    indices[self] = length(indices) + 1
end

function generate(func::Evaluable; show::Bool=false)
    indices = dependencies(func)
    symbols = Dict{Evaluable,Symbol}(func => gensym(string(index)) for (func, index) in indices)

    code = Array{Expr,1}()
    for (func, index) in indices
        argsyms = [symbols[arg] for arg in arguments(func)]
        tgtsym = symbols[func]
        push!(code, :($tgtsym = $(codegen(func, argsyms...))))
    end
    code = Expr(:block, code...)

    typeinfo = [(:point, Vector{Float64}), (:element, Element)]
    paramlist = Expr(:parameters, (:($sym::$tp) for (sym, tp) in typeinfo)...)
    prototype = Expr(:call, :evaluate, paramlist)
    definition = Expr(:function, prototype, code)

    show && @show definition

    mod = Module()
    Core.eval(mod, :(import Jutils.Elements: Element))
    Core.eval(mod, :(import Jutils.Transforms: applytrans))
    Core.eval(mod, :(using EllipsisNotation))
    Core.eval(mod, definition)
    return mod.evaluate
end



# Generic evaluation arguments

struct Argument{T} <: Evaluable{T}
    expression :: Union{Symbol, Expr}
end

Base.show(io::IO, self::Argument) = print(io, "Argument($(self.expression))")
arguments(::Argument) where T = ()
codegen(self::Argument) = self.expression

# Element transformation
const trans = Argument{TransformChain}(:(element.transform))

# Element index
const elemindex = Argument{Int}(:(element.index))



# Array functions

abstract type ArrayEvaluable{T,N} <: Evaluable{Array{T,N}} end

asarray(v::ArrayEvaluable) = v
asarray(v::Real) = Constant(v)
asarray(v::AbstractArray) = Constant(v)

Base.ndims(self::ArrayEvaluable{T,N}) where {T,N} = N :: Int
Base.size(self::ArrayEvaluable, dim::Int) = size(self)[dim]
Base.show(io::IO, self::ArrayEvaluable) = print(io, string(typeof(self).name.name), size(self))
Base.:+(self::ArrayEvaluable, rest...) = Sum(self, (asarray(v) for v in rest)...)
Base.:*(self::ArrayEvaluable, rest...) = Product(self, (asarray(v) for v in rest)...)



# Quadrature point (reference coordinates)

struct Point{N} <: ArrayEvaluable{Float64,1} end

arguments(self::Point) = ()
size(self::Point{N}) where {T,N} = (N::Int,)
codegen(self::Point) = :point



# ApplyTransform

struct ApplyTransform <: ArrayEvaluable{Float64,1}
    trans :: Evaluable{TransformChain}
    arg :: ArrayEvaluable{Float64,1}
    dims :: Int
end

arguments(self::ApplyTransform) = (self.trans, self.arg)
size(self::ApplyTransform) = (self.dims,)
codegen(self::ApplyTransform, trans, arg) = :(applytrans($arg, $trans))



# Constant

struct Constant{T,N} <: ArrayEvaluable{T,N}
    value :: Array{T,N}
end

# Scalars are wrapped as zero-dimensional arrays
Constant(v::T) where T<:Number = Constant(fill(v, ()))

arguments(::Constant) = ()
size(self::Constant) = size(self.value)
codegen(self::Constant) = self.value



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

function codegen(self::Matmul{T,L,R,N}, left, right) where {T,L,R,N}
    (result, i, j, k) = gensym("result"), gensym("i"), gensym("j"), gensym("k")
    quote
        $result = zeros($T, $(size(self)...))
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

function codegen(self::Monomials, points)
    (value, j) = gensym("value"), gensym("j")
    code = Any[:($value = zeros(Float64, $(size(self)...)))]

    loopcode = Any[:($value[1,$j] = 1.0)]
    for i in 1:self.degree
        push!(loopcode, :($value[$(i+1),$j] = $value[$i,$j] * $points[$j]))
    end
    push!(code, Expr(:for, :($j = CartesianIndices(size($points))), Expr(:block, loopcode...)))

    push!(code, :($value))
    Expr(:block, code...)
end



# Product

struct Product{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Product(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (restype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Product(terms...) = Product(terms)

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

struct Sum{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Sum(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (restype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Sum(terms...) = Sum(terms)

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

# rootcoords(ndims::Int) = ApplyTransform(trans, points{Float64,2}, ndims)

end # module
