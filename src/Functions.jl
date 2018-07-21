module Functions

import OrderedCollections: OrderedDict
import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, isdone, repeated, Stateful
import Base: size

import ..Transforms: TransformChain

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

    typeinfo = [(func.symbol, func.type_) for func in keys(indices) if isa(func, Argument)]
    paramlist = Expr(:parameters, (:($sym::$tp) for (sym, tp) in typeinfo)...)
    prototype = Expr(:call, :evaluate, paramlist)
    definition = Expr(:function, prototype, code)

    mod = Module()
    Core.eval(mod, :(import Jutils.Transforms: applytrans))
    Core.eval(mod, definition)
    return mod.evaluate
end


struct Argument <: AbstractFunction
    symbol :: Symbol
    type_ :: DataType
end

Base.show(io::IO, self::Argument) = print(io, "Argument($(self.symbol))")
arguments(::Argument) = ()
codegen(self::Argument) = self.symbol

const points = Argument(:points, Array{Float64,2})
const trans = Argument(:trans, TransformChain)


abstract type AbstractArrayFunction <: AbstractFunction end

asarray(v::AbstractArrayFunction) = v
asarray(v::Real) = Constant(v)
asarray(v::AbstractArray) = Constant(v)

Base.show(io::IO, self::AbstractArrayFunction) = print(io, string(typeof(self).name.name), size(self))
Base.:+(self::AbstractArrayFunction, rest...) = Sum(self, (asarray(v) for v in rest)...)
Base.:*(self::AbstractArrayFunction, rest...) = Product(self, (asarray(v) for v in rest)...)


struct ApplyTransform <: AbstractArrayFunction
    trans :: AbstractFunction
    arg :: AbstractFunction
    dims :: Int
end

arguments(self::ApplyTransform) = (self.trans, self.arg)
size(self::ApplyTransform) = (self.dims,)
codegen(self::ApplyTransform, trans, arg) = :(applytrans($arg, $(trans)...))


struct Constant <: AbstractArrayFunction
    value :: Any
end

arguments(::Constant) = ()
size(self::Constant) = size(self.value)
codegen(self::Constant) = self.value


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


struct Product <: AbstractArrayFunction
    terms :: Tuple{Vararg{AbstractArrayFunction}}
end

Product(terms::AbstractArrayFunction...) = Product(terms)

arguments(self::Product) = self.terms
size(self::Product) = broadcast_shape((size(term) for term in self.terms)...)
function codegen(self::Product, args...)
    code = args[1]
    for arg in args[2:end]
        code = :($code .* $arg)
    end
    code
end


rootcoords(ndims::Int) = ApplyTransform(trans, points, ndims)

end # module
