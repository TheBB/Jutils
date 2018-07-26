module Functions

import OrderedCollections: OrderedDict
import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful
import Base: convert, size

import ..Transforms: TransformChain
import ..Elements: Element

export ast, generate, rootcoords, Constant, Sum, Product


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
    Core.eval(mod, :(using EllipsisNotation))
    Core.eval(mod, definition)
    return mod.evaluate
end



# Evaluation arguments

struct Argument <: AbstractFunction
    expression :: Union{Symbol, Expr}
end

Base.show(io::IO, self::Argument) = print(io, "Argument($(self.expression))")
arguments(::Argument) = ()
codegen(self::Argument) = self.expression

const points = Argument(:points)              # Quadrature points (reference coordinates)
const trans = Argument(:(element.transform))  # Element transformation
const elemindex = Argument(:(element.index))  # Element index



# Array functions

abstract type AbstractArrayFunction <: AbstractFunction end

asarray(v::AbstractArrayFunction) = v
asarray(v::Real) = Constant(v)
asarray(v::AbstractArray) = Constant(v)

Base.show(io::IO, self::AbstractArrayFunction) = print(io, string(typeof(self).name.name), size(self))
Base.:+(self::AbstractArrayFunction, rest...) = Sum(self, (asarray(v) for v in rest)...)
Base.:*(self::AbstractArrayFunction, rest...) = Product(self, (asarray(v) for v in rest)...)



# ApplyTransform

struct ApplyTransform <: AbstractArrayFunction
    trans :: AbstractFunction
    arg :: AbstractFunction
    dims :: Int
end

arguments(self::ApplyTransform) = (self.trans, self.arg)
size(self::ApplyTransform) = (self.dims,)
codegen(self::ApplyTransform, trans, arg) = :(applytrans($arg, $(trans)...))



# Constant

struct Constant{T} <: AbstractArrayFunction
    value :: Array{T}

    # Pad with a dummy dimension for broadcasting over quadrature points
    Constant(v::Array{T}) where T = new{T}(reshape(v, size(v)..., 1))
end

# Scalars are wrapped as zero-dimensional arrays
Constant(v::T) where T<:Number = (wrap = Array{T,0}(undef); wrap[] = v; Constant(wrap))

arguments(::Constant) = ()
size(self::Constant) = size(self.value)[1:end-1]
codegen(self::Constant) = self.value



# Elementwise

struct Elementwise <: AbstractArrayFunction
    data :: Array
    index :: AbstractFunction
end

arguments(self::Elementwise) = (self.index,)
size(self::Elementwise) = size(self.data)[1:end-1]

# Use a range for the last index to gert a dummy dimension for broadcasting
codegen(self::Elementwise, index) = :($(self.data)[.., $index:$index])



# Sum

struct Sum <: AbstractArrayFunction
    terms :: Tuple{Vararg{AbstractArrayFunction}}
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



# Product

struct Product <: AbstractArrayFunction
    terms :: Tuple{Vararg{AbstractArrayFunction}}
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



# Miscellaneous

rootcoords(ndims::Int) = ApplyTransform(trans, points, ndims)

end # module
