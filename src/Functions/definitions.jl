
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



# ApplyTransform

@autohasheq struct ApplyTransform <: ArrayEvaluable{Float64,1}
    trans :: Evaluable{TransformChain}
    arg :: Evaluable{Vector{Float64}}
    dims :: Int
end

arguments(self::ApplyTransform) = (self.trans, self.arg)
Base.size(self::ApplyTransform) = (self.dims,)
prealloc(self::ApplyTransform) = []
codegen(self::ApplyTransform, trans, arg) = :(applytrans($arg, $trans))



# ApplyTransformGrad

@autohasheq struct ApplyTransformGrad <: ArrayEvaluable{Float64,2}
    trans :: Evaluable{TransformChain}
    arg :: Evaluable{Vector{Float64}}
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
    map :: IndexMap

    function GetIndex(value::ArrayEvaluable{T,N}, map::IndexMap) where {T,N}
        dimcheck(map, 0, 1) || error("Multidimensional indices not yet supported")
        indices = Index[get(map, i, :) for i in 1:ndims(value)]
        newdims = resultndims(map, ndims(value))
        new{T,newdims}(value, indices, map)
    end
end

Base.size(self::GetIndex) = resultsize(self.map, size(self.value))

arguments(self::GetIndex) = (self.value, values(self.map)...)
prealloc(self::GetIndex{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::GetIndex, value, varindices...)
    target = varindices[end]
    varindices = varindices[1:end-1]
    quote
        $target .= $value[$(codegen(self.map, varindices, ndims(self.value))...)]
        $target
    end
end



# Inflate

@autohasheq struct Inflate{T,N} <: ArrayEvaluable{T,N}
    data :: ArrayEvaluable{T}
    shape :: Shape
    map :: IndexMap

    function Inflate(data::ArrayEvaluable, shape::Shape, map::IndexMap)
        ndims(data) == length(shape) || error("Dimension mismatch")
        resultsize(map, shape) == size(data) || error("Size mismatch")
        dimcheck(map, 1, 1) || error("Only linear indices supported in inflation")
        new{eltype(data), length(shape)}(data, shape, map)
    end
end

Inflate(data::ArrayEvaluable, shape::Shape, indices::Pair...) = Inflate(data, shape, IndexMap(indices...))

Base.size(self::Inflate) = self.shape
separate(self::Inflate) = [
    (Tupl((axis in keys(self.map) ? getindex(self.map[axis], ind) : ind)  for (axis, ind) in enumerate(inds)), data)
    for (inds, data) in separate(self.data)
]

arguments(self::Inflate) = (self.data, values(self.map)...)
prealloc(self::Inflate{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]
function codegen(self::Inflate{T}, data, indices...) where T
    target, varindices = indices[end], collect(indices[1:end-1])
    weaved = codegen(self.map, varindices, ndims(self))
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



# Reshape

@autohasheq struct Reshape{T,N} <: ArrayEvaluable{T,N}
    source :: ArrayEvaluable{T}
    newshape :: Shape

    function Reshape(source::ArrayEvaluable, newshape::Shape)
        @assert prod(newshape) == length(source)
        new{eltype(source), length(newshape)}(source, newshape)
    end
end

Base.size(self::Reshape) = self.newshape

arguments(self::Reshape) = (self.source,)
prealloc(::Reshape{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]
codegen(self::Reshape, source, target) = :($target[:] = $source[:]; $target)



# Add

@autohasheq struct Add{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Add(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (eltype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Add(terms...) = Add(terms)

arguments(self::Add) = self.terms
Base.size(self::Add) = broadcast_shape((size(term) for term in self.terms)...)
prealloc(self::Add{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Add, args...)
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
