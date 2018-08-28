
# Generic evaluable

abstract type Evaluable{T} end

arguments(self::Evaluable) = ()
isconstant(::Any) = true
isconstant(self::Evaluable) = all(isconstant(arg) for arg in arguments(self))
iselconstant(::Any) = true
iselconstant(self::Evaluable) = all(iselconstant(arg) for arg in arguments(self))
restype(self::Evaluable{T}) where T = T
typetree(self::Evaluable) = [typeof(self).name.name, (typetree(arg) for arg in arguments(self))...]

# For internal use by the compiler
mutable struct FunctionData
    func :: Evaluable
    arginds :: Vector{Int}
    tgtsym :: Union{Symbol,Nothing}
    allocexprs :: Vector{Any}
    allocsyms :: Vector{Symbol}
    master_alloc :: Union{Symbol,Nothing}

    # Note: Many fields are left undefined
    function FunctionData(func::Evaluable)
        retval = new(func)
        retval.master_alloc = nothing
        retval
    end
end

function linearize(self::Evaluable)
    indices = OrderedDict{Evaluable,Int}()
    linearize!(indices, self)
    sequence = [FunctionData(func) for (func, index) in indices]
    for data in sequence
        data.arginds = Int[indices[arg] for arg in arguments(data.func)]
    end
    sequence
end

function linearize!(indices::OrderedDict{Evaluable,Int}, self::Evaluable)
    haskey(indices, self) && return
    for func in arguments(self)
        linearize!(indices, func)
    end
    indices[self] = length(indices) + 1
end

function ast(self::Evaluable)
    sequence = linearize(self)
    ast!(length(sequence), Set{Int}(), linearize(self))
end

function ast!(index::Int, seen::Set{Int}, sequence::Vector{FunctionData})
    index in seen && return string("%", string(index))

    s = ""
    argids = Stateful(sequence[index].arginds)
    for argid in argids
        bridge = isdone(argids) ? flatten((("└─",), repeated("  "))) : flatten((("├─",), repeated("│ ")))
        sublines = split(ast!(argid, seen, sequence), "\n")
        subtree = join((string(b, s) for (b, s) in zip(bridge, sublines)), "\n")
        s = string(s, "\n", subtree)
    end

    push!(seen, index)
    string("%", string(index), " = ", repr(sequence[index].func), s)
end



# Array functions

abstract type ArrayEvaluable{T,N} <: Evaluable{Array{T,N}} end

Base.eltype(::ArrayEvaluable{T}) where T = T

asarray(v::ArrayEvaluable) = v
asarray(v::Real) = Constant(v)
asarray(v::AbstractArray) = Constant(v)

Base.length(self::ArrayEvaluable) = prod(size(self))
Base.ndims(self::ArrayEvaluable{T,N}) where {T,N} = N :: Int
Base.size(self::ArrayEvaluable, dim::Int) = size(self)[dim]
Base.show(io::IO, self::ArrayEvaluable) = print(io, string(typeof(self).name.name), size(self))

isnormal(::ArrayEvaluable) = true
separate(self::ArrayEvaluable) = [(Tupl(Constant(collect(1:n)) for n in size(self)), self)]



# Compiled functions

struct Compiled{T}
    kernel :: Function
end

callable(self::Compiled) = Base.invokelatest(self.kernel)


struct CompiledArray{T,N}
    kernel :: Function
    shape :: Shape
end

Base.size(self::CompiledArray) = self.shape
callable(self::CompiledArray) = Base.invokelatest(self.kernel)


struct CompiledSparseArray{T,N}
    ikernel :: Compiled
    dkernel :: CompiledArray{T,N}
    shape :: Shape
    blockshape :: Shape
end

Base.size(self::CompiledSparseArray) = self.shape
callable(self::CompiledSparseArray) = (callable(self.ikernel), callable(self.dkernel))



# Compilation

function _compile(infunc::Evaluable, show::Bool)
    sequence = linearize(infunc)
    for data in sequence
        data.tgtsym = gensym()
        data.allocexprs = prealloc(data.func, [sequence[argid].master_alloc for argid in data.arginds])
        data.allocsyms = Symbol[gensym() for _ in 1:length(data.allocexprs)]
        if !isempty(data.allocsyms)
            data.master_alloc = data.allocsyms[end]
        end
    end

    # Code for allocation
    alloccode = Vector{Expr}()
    for data in sequence, (sym, expr) in zip(data.allocsyms, data.allocexprs)
        push!(alloccode, :($sym = $expr))
    end

    # Code for evaluation
    evalcode = Vector{Expr}()
    for data in sequence
        argsyms = Symbol[sequence[argid].tgtsym for argid in data.arginds]
        tgtsym = data.tgtsym
        push!(evalcode, :($(data.tgtsym) = $(codegen(data.func, argsyms, data.allocsyms))))
    end

    typeinfo = [(:point, Vector{Float64}), (:element, Element)]
    paramlist = (:($sym::$tp) for (sym, tp) in typeinfo)
    push!(alloccode, Expr(:function, Expr(:call, :evaluate, paramlist...), Expr(:block, evalcode...)))
    push!(alloccode, :(return evaluate))

    definition = Expr(:function, Expr(:call, :mkevaluate), Expr(:block, alloccode...))

    show && @show definition

    mod = Module()
    Core.eval(mod, :(import Jutils.Elements: Element))
    Core.eval(mod, :(import Jutils.Transforms: applytrans, applytrans_grad))
    Core.eval(mod, :(using LinearAlgebra))
    Core.eval(mod, :(import TensorOperations))
    Core.eval(mod, definition)

    mod.mkevaluate
end

compile(func::Evaluable{T}; show::Bool=false) where T = Compiled{T}(Base.invokelatest(_compile, func, show))

function compile(func::ArrayEvaluable{T,N}; show::Bool=false, dense::Bool=true) where {T,N}
    if dense
        CompiledArray{T,N}(Base.invokelatest(_compile, Normalize(func), show), size(func))
    else
        blocks = separate(func)
        @assert length(blocks) == 1
        inds, data = blocks[1]
        inds = Tupl(Normalize(k) for k in inds)
        ikernel = compile(inds; show=show)
        dkernel = compile(Normalize(data); show=show)
        CompiledSparseArray{T,N}(ikernel, dkernel, size(func), size(data))
    end
end
