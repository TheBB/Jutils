
# Generic evaluable

abstract type Evaluable{T} end

arguments(self::Evaluable) = ()
isconstant(::Any) = true
isconstant(self::Evaluable) = all(isconstant(arg) for arg in arguments(self))
iselconstant(::Any) = true
iselconstant(self::Evaluable) = all(iselconstant(arg) for arg in arguments(self))
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


struct CompiledFunction
    callable :: Function
    restype :: DataType
end

@inline (self::CompiledFunction)(point::Vector{Float64}, element::Element) = self.callable(point, element)


struct CompiledArrayFunction
    callable :: Function
    shape :: Tuple{Vararg{Int}}
    restype :: DataType
end

@inline (self::CompiledArrayFunction)(point::Vector{Float64}, element::Element) = self.callable(point, element)
Base.size(self::CompiledArrayFunction) = self.shape



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



# Compilation

function _generate(infunc::Evaluable, show::Bool)
    funcindices = dependencies(infunc)
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
    paramlist = (:($sym::$tp) for (sym, tp) in typeinfo)
    push!(alloccode, Expr(:function, Expr(:call, :evaluate, paramlist...), Expr(:block, evalcode...)))
    push!(alloccode, :(return evaluate))

    definition = Expr(:function, Expr(:call, :mkevaluate), Expr(:block, alloccode...))

    show && @show definition

    mod = Module()
    Core.eval(mod, :(import Jutils.Elements: Element))
    Core.eval(mod, :(import Jutils.Transforms: applytrans))
    Core.eval(mod, :(using LinearAlgebra))
    Core.eval(mod, definition)

    Base.invokelatest(mod.mkevaluate)
end

function generate(func::Evaluable{T}; show::Bool=false) where T
    return CompiledFunction(_generate(func, show), T)
end

function generate(func::ArrayEvaluable{T}; show::Bool=false) where T
    return CompiledArrayFunction(_generate(func, show), size(func), T)
end
