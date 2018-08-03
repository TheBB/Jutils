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
export rootcoords


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

function _generate(func::Evaluable; show::Bool=false)
    funcindices = dependencies(func)
    tgtsymbols = Dict(func => gensym(string(index)) for (func, index) in funcindices)
    allocexprs = Dict(func => prealloc(func) for func in keys(funcindices))
    allocsymbols = Dict(func => [gensym("alloc") for _ in 1:length(exprs)] for (func, exprs) in allocexprs)

    # Create the allocating function
    alloccode = Vector{Expr}()
    evalcode = Vector{Expr}()
    for func = keys(funcindices)
        for (sym, expr) in zip(allocsymbols[func], allocexprs[func])
            push!(alloccode, :($sym = $expr))
        end

        argsyms = [tgtsymbols[arg] for arg in arguments(func)]
        tgtsym = tgtsymbols[func]
        push!(evalcode, :($tgtsym = $(codegen(func, argsyms..., allocsymbols[func]...))))
    end

    typeinfo = [(:point, Vector{Float64}), (:element, Element)]
    paramlist = Expr(:parameters, (:($sym::$tp) for (sym, tp) in typeinfo)...)
    push!(alloccode, Expr(:function, Expr(:call, :evaluate, paramlist), Expr(:block, evalcode...)))
    push!(alloccode, :(return evaluate))

    definition = Expr(:function, Expr(:call, :mkevaluate), Expr(:block, alloccode...))

    show && @show definition

    mod = Module()
    Core.eval(mod, :(import Jutils.Elements: Element))
    Core.eval(mod, :(import Jutils.Transforms: applytrans))
    Core.eval(mod, :(using EllipsisNotation))
    Core.eval(mod, definition)

    return Base.invokelatest(mod.mkevaluate)
end

function generate(func::Evaluable; show::Bool=false)
    mkeval = _generate(func; show=show)
    mkeval
end



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



# Array functions

abstract type ArrayEvaluable{T,N} <: Evaluable{Array{T,N}} end

arraytype(::ArrayEvaluable{T}) where T = T

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
# TODO: More general indexing expressions

struct GetItem{T,N} <: ArrayEvaluable{T,N}
    value :: ArrayEvaluable
    indices :: Tuple

    function GetItem(value::ArrayEvaluable{T,N}, indices...) where {T,N}
        all(isa(i, Evaluable{Int}) || isa(i, Colon) for i in indices) || error("General indexing not supported")
        indextypes = [isa(i, Evaluable) ? restype(i) : typeof(i) for i in indices]
        (_, rtype) = code_typed(getindex, (Array{T,N}, Colon, Int))[end]
        rtype <: Array || error("Result of indexing must be an array")
        rtype.isconcretetype || error("Result of indexing is not a concrete type")
        (RT, RN) = rtype.parameters
        new{RT,RN}(value, indices)
    end
end

arguments(self::GetItem) = (self.value, (i for i in self.indices if isa(i, Evaluable))...)
size(self::GetItem) = Tuple(s for (s, i) in zip(size(self.value), self.indices) if isa(i, Colon))
prealloc(self::GetItem) = []

function codegen(self::GetItem, value, varindices...)
    weaved = []
    varindices = collect(varindices)
    for s in self.indices
        isa(s, Colon) ? push!(weaved, :(:)) : push!(weaved, popfirst!(varindices))
    end
    :($value[$(weaved...)])
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
        for $i = CartesianIndices($(size(self.left)[1:end-1])), $j = CartesianIndices($(size(self.right)[2:end]))
            $result[$i,$j] = zero($T)
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
