
# Generic evaluation arguments

@autohasheq struct Argument{T} <: Evaluable{T}
    expression :: Union{Symbol, Expr}
    isconstant :: Bool
    iselconstant :: Bool
end

Base.show(io::IO, self::Argument) = print(io, "Argument($(self.expression))")
arguments(::Argument) where T = ()
isconstant(self::Argument) = self.isconstant
iselconstant(self::Argument) = self.iselconstant
prealloc(::Argument) = []
codegen(self::Argument) = self.expression


@autohasheq struct ArrayArgument{T,N} <: ArrayEvaluable{T,N}
    expression :: Union{Symbol, Expr}
    isconstant :: Bool
    iselconstant :: Bool
    shape :: Shape
end

arguments(::ArrayArgument) where T = ()
isconstant(self::ArrayArgument) = self.isconstant
iselconstant(self::ArrayArgument) = self.iselconstant
Base.size(self::ArrayArgument) = self.shape
prealloc(::ArrayArgument) = []
codegen(self::ArrayArgument) = self.expression



# Quadrature point (reference coordinates)

@autohasheq struct Point <: ArrayEvaluable{Float64,1}
    ndims :: Int
end

arguments(self::Point) = ()
isconstant(::Point) = false
iselconstant(::Point) = false
Base.size(self::Point) = (self.ndims,)
prealloc(::Point) = []
codegen(self::Point) = :point



# ApplyTransform

@autohasheq struct ApplyTransform <: ArrayEvaluable{Float64,1}
    trans :: Evaluable{TransformChain}
    arg :: ArrayEvaluable{Float64,1}
    dims :: Int
end

arguments(self::ApplyTransform) = (self.trans, self.arg)
Base.size(self::ApplyTransform) = (self.dims,)
prealloc(self::ApplyTransform) = []
codegen(self::ApplyTransform, trans, arg) = :(applytrans($arg, $trans))



# ApplyTransformGrad

@autohasheq struct ApplyTransformGrad <: ArrayEvaluable{Float64,2}
    trans :: Evaluable{TransformChain}
    arg :: ArrayEvaluable{Float64,1}
    dims :: Int
end

arguments(self::ApplyTransformGrad) = (self.trans, self.arg)
Base.size(self::ApplyTransformGrad) = (self.dims, self.dims)
prealloc(self::ApplyTransformGrad) = []
codegen(self::ApplyTransformGrad, trans, arg) = :(applytrans_grad($arg, $trans))



# Constant

@autohasheq struct Constant{T,N} <: ArrayEvaluable{T,N}
    value :: Array{T,N}
end

# Scalars are wrapped as zero-dimensional arrays
Constant(v::T) where T <: Number = Constant(fill(v, ()))

arguments(::Constant) = ()
Base.size(self::Constant) = size(self.value)
prealloc(self::Constant) = [self.value]
codegen(::Constant, alloc) = alloc



# Contract

@autohasheq struct Contract{T,N} <: ArrayEvaluable{T,N}
    left :: ArrayEvaluable
    right :: ArrayEvaluable
    linds :: Vector
    rinds :: Vector
    tinds :: Vector

    function Contract(left::ArrayEvaluable, right::ArrayEvaluable, linds::Vector, rinds::Vector, tinds::Vector)
        ndims(left) == length(linds) || error("Incorrect number of indices")
        ndims(right) == length(rinds) || error("Incorrect number of indices")
        nonrep = symdiff(Set(linds), Set(rinds))
        nonrep == Set(tinds) || error("Incorrect number of indices")

        newtype = promote_type(eltype(left), eltype(right))
        newdims = length(tinds)
        new{newtype,newdims}(left, right, linds, rinds, tinds)
    end
end

function Base.size(self::Contract)
    sizes = Dict(Iterators.flatten([zip(self.linds, size(self.left)), zip(self.rinds, size(self.right))]))
    ((sizes[ti] for ti in self.tinds)...,)
end

arguments(self::Contract) = (self.left, self.right)
prealloc(self::Contract{T}) where T = [:((Array{$T})(undef, $(size(self)...)))]

function codegen(self::Contract, left, right, target)
    cont = macroexpand(
        Functions,
        :(@tensor $target[($(self.tinds...))] = $left[($(self.linds...))] * $right[($(self.rinds...))])
    )
    :($cont; $target)
end



# GetIndex
# TODO: More general indexing expressions?

@autohasheq struct GetIndex{T,N} <: ArrayEvaluable{T,N}
    value :: ArrayEvaluable
    indices :: Indices

    function GetIndex(value::ArrayEvaluable{T,N}, indices::Indices) where {T,N}
        length(indices) == ndims(value) || error("Inconsistent indexing")
        index_dimcheck(indices, 0, 1) || error("Multidimensional indices not supported for getindex")

        # Attempt type inference. TODO: Make this more robust
        indextypes = [isa(i, Evaluable) ? restype(i) : typeof(i) for i in indices]
        (_, rtype) = code_typed(getindex, (Array{T,N}, indextypes...))[end]

        rtype <: Array || error("Result of indexing must be an array")
        rtype.isconcretetype || error("Result of indexing is not a concrete type")
        (RT, RN) = rtype.parameters
        new{RT,RN}(value, indices)
    end
end

arguments(self::GetIndex) = (self.value, (i for i in self.indices if isa(i, Evaluable))...)
Base.size(self::GetIndex) = index_resultsize(size(self.value), self.indices)
prealloc(self::GetIndex) = []

function codegen(self::GetIndex, value, varindices...)
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

@autohasheq struct Inflate{T,N} <: ArrayEvaluable{T,N}
    data :: ArrayEvaluable{T}
    indices :: Indices
    shape :: Shape

    function Inflate(data::ArrayEvaluable, shape::Shape, indices::Tuple)
        indices = Index[isa(ix, Colon) ? ix : asarray(ix) for ix in indices]
        length(indices) == length(shape) || error("Inconsistent indexing")
        index_resultsize(shape, indices) == size(data) || error("Inconsistent dimensions")
        index_dimcheck(indices, 1, 1) || error("Multidimensional indices not supported for inflate")
        new{eltype(data), length(shape)}(data, indices, shape)
    end
end

arguments(self::Inflate) = (self.data, (i for i in self.indices if isa(i, Evaluable))...)
Base.size(self::Inflate) = self.shape
separate(self::Inflate) = [
    (Tupl((isa(ix, Colon) ? ind : getindex(ix, ind)) for (ix, ind) in zip(self.indices, inds)), data)
    for (inds, data) in separate(self.data)
]
prealloc(self::Inflate{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Inflate{T}, data, indices...) where T
    target, varindices = indices[end], collect(indices[1:end-1])
    weaved = index_weave(self.indices, varindices)
    quote
        $target[:] .= $(zero(T))
        $target[$(weaved...)] = $data
        $target
    end
end



# InsertAxis

@autohasheq struct InsertAxis{T,N} <: ArrayEvaluable{T,N}
    source :: ArrayEvaluable{T}
    axes :: Vector{Int}

    function InsertAxis(source::ArrayEvaluable, axes::Vector{Int})
        isempty(axes) && return source
        all(1 <= i <= ndims(source) + 1 for i in axes) || error("Axes out of range")
        new{eltype(source), ndims(source)+length(axes)}(source, sort(axes))
    end
end

arguments(self::InsertAxis) = (self.source,)
function Base.size(self::InsertAxis)
    shape = collect(Int, size(self.source))
    insertmany!(shape, self.axes, 1)
    Tuple(shape)
end
prealloc(self::InsertAxis) = []
codegen(self::InsertAxis, source) = :(reshape($source, $(size(self)...)))



# Inv

@autohasheq struct Inv{T,N} <: ArrayEvaluable{T,2}
    source :: ArrayEvaluable{T,2}

    function Inv(source::ArrayEvaluable{T,2}) where T
        size(source, 1) == size(source, 2) || error("Expected square matrix")
        new{T, size(source,1)}(source)
    end
end

arguments(self::Inv) = (self.source,)
Base.size(self::Inv) = size(self.source)
prealloc(self::Inv{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

# Would be useful with an inv! function here
codegen(::Inv, source, target) = :($target[:] = inv($source); $target)
codegen(::Inv{T,1}, source, target) where T = :($target[1,1] = one($T) / $source[1,1]; $target)
codegen(::Inv{T,2}, source, target) where T = quote
    $target[1,1] = $source[2,2]
    $target[2,2] = $source[1,1]
    $target[1,2] = -$source[1,2]
    $target[2,1] = -$source[2,1]
    $target ./= ($source[1,1] * $source[2,2] - $source[1,2] * $source[2,1])
    $target
end
codegen(::Inv{T,3}, source, target) where T = quote
    $target[1,1] = $source[2,2] * $source[3,3] - $source[2,3] * $source[3,2]
    $target[2,1] = $source[2,3] * $source[3,1] - $source[2,1] * $source[3,3]
    $target[3,1] = $source[2,1] * $source[3,2] - $source[2,2] * $source[3,1]
    $target[1,2] = $source[1,3] * $source[3,2] - $source[1,2] * $source[3,3]
    $target[2,2] = $source[1,1] * $source[3,3] - $source[1,3] * $source[3,1]
    $target[3,2] = $source[1,2] * $source[3,1] - $source[1,1] * $source[3,2]
    $target[1,3] = $source[1,2] * $source[2,3] - $source[1,3] * $source[2,2]
    $target[2,3] = $source[1,3] * $source[2,1] - $source[1,1] * $source[2,3]
    $target[3,3] = $source[1,1] * $source[2,2] - $source[1,2] * $source[2,1]
    $target ./= $source[1,1] * $target[1,1] + $source[1,2] * $target[2,1] + $source[1,3] * $target[3,1]
    $target
end



# Monomials

@autohasheq struct Monomials{T,N} <: ArrayEvaluable{T,N}
    points :: ArrayEvaluable{T}
    degree :: Int
    padding :: Int

    Monomials(points::ArrayEvaluable{T,M}, degree::Int, padding::Int) where {T,M} = new{T,M+1}(points, degree, padding)
    Monomials(points::ArrayEvaluable, degree::Int) = Monomials(points, degree, 0)
end

arguments(self::Monomials) = (self.points,)
Base.size(self::Monomials) = (self.degree + self.padding + 1, size(self.points)...)
prealloc(self::Monomials{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Monomials{T}, points, target) where T
    j = gensym("j")

    loopcode = Any[:($target[$(1+self.padding),$j] = 1.0)]
    for i in self.padding .+ (1:self.degree)
        push!(loopcode, :($target[$(i+1),$j] = $target[$i,$j] * $points[$j]))
    end

    quote
        $target .= zero($T)
        for $j = CartesianIndices(size($points))
            $(loopcode...)
        end
        $target
    end
end



# Neg

@autohasheq struct Neg{T,N} <: ArrayEvaluable{T,N}
    source :: ArrayEvaluable{T,N}
end

arguments(self::Neg) = (self.source,)
Base.size(self::Neg) = size(self.source)
prealloc(self::Neg) = []
codegen(::Neg, source) = :(-$source)



# Product

@autohasheq struct Product{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Product(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (eltype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

arguments(self::Product) = self.terms
Base.size(self::Product) = broadcast_shape((size(term) for term in self.terms)...)
prealloc(self::Product{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

# Note: element-wise product!
function codegen(self::Product, args...)
    code = args[1]
    for arg in args[2:end-1]
        code = :($code .* $arg)
    end
    :($(args[end]) .= $code; $(args[end]))
end



# Sum

@autohasheq struct Sum{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Sum(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (eltype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Sum(terms...) = Sum(terms)

arguments(self::Sum) = self.terms
Base.size(self::Sum) = broadcast_shape((size(term) for term in self.terms)...)
prealloc(self::Sum{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Sum, args...)
    code = args[1]
    for arg in args[2:end-1]
        code = :($code .+ $arg)
    end
    :($(args[end]) .= $code; $(args[end]))
end



# Tupl(e)

@autohasheq struct Tupl{T} <: Evaluable{T}
    terms :: Tuple{Vararg{Evaluable}}

    function Tupl(terms)
        terms = Tuple(terms)
        # TODO: Is there a better way to do this?
        tupletype = eval(:(Tuple{$((restype(term) for term in terms)...)}))
        new{tupletype}(terms)
    end
end

Base.iterate(self::Tupl) = iterate(self.terms)
Base.iterate(self::Tupl, i::Int) = iterate(self.terms, i)
Base.length(self::Tupl) = length(self.terms)

arguments(self::Tupl) = self.terms
prealloc(self::Tupl) = []
codegen(self::Tupl, terms...) = :($(terms...),)



# Zeros

@autohasheq struct Zeros{T,N} <: ArrayEvaluable{T,N}
    shape :: Shape
    Zeros(T, shape::Int...) = new{T,length(shape)}(shape)
end

arguments(::Zeros) = ()
Base.size(self::Zeros) = self.shape
separate(::Zeros) = []
prealloc(self::Zeros{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]
codegen(self::Zeros{T}, target) where T = :($target[:] .= zero($T); $target)
