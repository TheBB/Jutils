
# Generic evaluation arguments

@autohasheq struct Argument{T} <: Evaluable{T}
    expression :: Union{Symbol, Expr}
    isconstant :: Bool
    iselconstant :: Bool
end

Base.show(io::IO, self::Argument) = print(io, "Argument($(self.expression))")
isconstant(self::Argument) = self.isconstant
iselconstant(self::Argument) = self.iselconstant

arguments(::Argument) where T = ()
prealloc(::Argument) = []
@destructure codegen(self::Argument, [], []) = self.expression


@autohasheq struct ArrayArgument{T,N} <: ArrayEvaluable{T,N}
    expression :: Union{Symbol, Expr}
    isconstant :: Bool
    iselconstant :: Bool
    shape :: Shape
end

isconstant(self::ArrayArgument) = self.isconstant
iselconstant(self::ArrayArgument) = self.iselconstant
Base.size(self::ArrayArgument) = self.shape

arguments(::ArrayArgument) where T = ()
prealloc(::ArrayArgument) = []
@destructure codegen(self::ArrayArgument, [], []) = self.expression



# ApplyTransform

@autohasheq struct ApplyTransform <: ArrayEvaluable{Float64,1}
    trans :: Evaluable{TransformChain}
    arg :: Evaluable{Vector{Float64}}
    dims :: Int
end

arguments(self::ApplyTransform) = (self.trans, self.arg)
Base.size(self::ApplyTransform) = (self.dims,)
prealloc(::ApplyTransform) = []
@destructure codegen(self::ApplyTransform, [trans, arg], []) =
    :(applytrans($arg, $trans) :: $(restype(self)))



# ApplyTransformGrad

@autohasheq struct ApplyTransformGrad <: ArrayEvaluable{Float64,2}
    trans :: Evaluable{TransformChain}
    arg :: Evaluable{Vector{Float64}}
    dims :: Int
end

arguments(self::ApplyTransformGrad) = (self.trans, self.arg)
Base.size(self::ApplyTransformGrad) = (self.dims, self.dims)
prealloc(::ApplyTransformGrad) = []
@destructure codegen(self::ApplyTransformGrad, [trans, arg], []) =
    :(applytrans_grad($arg, $trans) :: $(restype(self)))



# Constant

@autohasheq struct Constant{T,N} <: ArrayEvaluable{T,N}
    value :: Array{T,N}
end

# Scalars are wrapped as zero-dimensional arrays
Constant(v::T) where T <: Number = Constant(fill(v, ()))

Base.size(self::Constant) = size(self.value)

arguments(::Constant) = ()
prealloc(self::Constant) = [self.value]
@destructure codegen(::Constant, [], [alloc]) = alloc



# Contract

@autohasheq struct Contract{T,N} <: ArrayEvaluable{T,N}
    terms :: Vector{ArrayEvaluable}
    indices :: Vector{Vector{Int}}
    target :: Vector{Int}
    shape :: Shape

    function Contract(terms::Vector{ArrayEvaluable}, indices::Vector{Vector{Int}}, target::Vector{Int})
        @assert length(terms) == length(indices)
        @assert all(length(ind) == ndims(term) for (ind, term) in zip(indices, terms))

        # Enumerate all index occurences and sizes
        occurences = Dict{Int,Vector{Int}}()
        for (term, inds) in zip(terms, indices)
            for (axid, sz) in zip(inds, size(term))
                push!(get!(occurences, axid, Int[]), sz)
            end
        end

        # Every occurence has consistent size, and appears exactly twice, or once + in target
        for (axid, occ) in occurences
            @assert length(occ) == 2 || length(occ) == 1 && axid in target
            @assert all(occ[1] == sz for sz in occ)
        end

        newshape = Tuple(occurences[axid][1] for axid in target)
        newtype = reduce(promote_type, (eltype(term) for term in terms))
        newdims = length(target)
        new{newtype, newdims}(ArrayEvaluable[Normalize(term) for term in terms], indices, target, newshape)
    end
end

newindex(self::Contract) = maximum(flatten((self.indices..., self.target))) + 1
Base.size(self::Contract) = self.shape

arguments(self::Contract) = Tuple(self.terms)
prealloc(self::Contract{T}) where T = [:((Array{$T})(undef, $(size(self)...)))]

# Here, we essentially roll our own einsum macro. Options are:
# - TensorOperations.jl: optimized for large arrays, which is typically not the case for us.
#   Generated code always allocates.
# - Einsum.jl: Generated code doesn't allocate, but not possible to avoid bounds checking,
#   which we can statically guarantee is not needed.
# May be changed in the future.
@destructure function codegen(self::Contract, terms, [target])
    counters = Dict{Int,Symbol}(i => gensym(string(i)) for i in flatten((self.indices..., self.target)))
    lengths = Dict{Int,Int}(i => size(term, j)
                            for (term, inds) in zip(self.terms, self.indices)
                            for (j, i) in enumerate(inds))
    caxes = Set{Int}(k for k in keys(lengths) if k ∉ self.target)
    taxes = Set{Int}(k for k in keys(lengths) if k in self.target)

    c = i -> (counters[j] for j in i)

    # Code for incrementing an element in the target array
    getcodes = (:($t[$(c(i)...)]) for (t, i) in zip(terms, self.indices))
    code = :(@inbounds $target[$(c(self.target)...)] += *($(getcodes...)))

    # Code for looping over all contracted axes
    for i in caxes
        code = :(for $(counters[i]) in 1:$(lengths[i]); $code end)
    end

    # Initialize accumulator to zero
    code = quote
        $target[$(c(self.target)...)] = $(zero(eltype(self)))
        $code
    end

    # Code for looping over all non-contracted axes
    for i in taxes
        code = :(for $(counters[i]) in 1:$(lengths[i]); $code end)
    end

    :($code; $target)
end



# DropDims

@autohasheq struct DropDims{T,N} <: ArrayEvaluable{T,N}
    source :: ArrayEvaluable
    dims :: Tuple{Vararg{Int}}

    function DropDims(source::ArrayEvaluable, dims::Tuple{Vararg{Int}})
        @assert all(size(source, d) == 1 for d in dims)
        new{eltype(source), ndims(source)-length(dims)}(source, dims)
    end
end

Base.size(self::DropDims) = Tuple(s for (i, s) in enumerate(size(self.source)) if i ∉ self.dims)

isnormal(::DropDims) = false
arguments(self::DropDims) = (self.source,)
prealloc(::DropDims) = []
@destructure codegen(self::DropDims, [source], []) = :(dropdims($source, dims=$(self.dims)))



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

isnormal(::GetIndex) = false
arguments(self::GetIndex) = (self.value, values(self.map)...)
prealloc(::GetIndex) = []

@destructure function codegen(self::GetIndex, [value, varindices...], [])
    :(view($value, $(codegen(self.map, varindices, ndims(self.value))...)))
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

@destructure function codegen(self::Inflate{T}, [data, indices...], [target]) where T
    weaved = codegen(self.map, indices, ndims(self))
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

function Base.size(self::InsertAxis)
    shape = collect(Int, size(self.source))
    insertmany!(shape, self.axes, 1)
    Tuple(shape)
end

isnormal(::InsertAxis) = false
arguments(self::InsertAxis) = (self.source,)
prealloc(::InsertAxis) = []
@destructure codegen(self::InsertAxis, [source], []) = :(reshape($source, $(size(self)...)))



# Inv

@autohasheq struct Inv{T,N} <: ArrayEvaluable{T,2}
    source :: ArrayEvaluable{T,2}

    function Inv(source::ArrayEvaluable{T,2}) where T
        size(source, 1) == size(source, 2) || error("Expected square matrix")
        new{T, size(source,1)}(source)
    end
end

Base.size(self::Inv) = size(self.source)

arguments(self::Inv) = (self.source,)
prealloc(self::Inv{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

# Would be useful with an inv! function here
@destructure codegen(::Inv, [source], [target]) = :($target .= inv($source); $target)
@destructure codegen(::Inv{T,1}, [source], [target]) where T = :($target[1,1] = one($T) / $source[1,1]; $target)
@destructure codegen(::Inv{T,2}, [source], [target]) where T = quote
    $target[1,1] = $source[2,2]
    $target[2,2] = $source[1,1]
    $target[1,2] = -$source[1,2]
    $target[2,1] = -$source[2,1]
    $target ./= ($source[1,1] * $source[2,2] - $source[1,2] * $source[2,1])
    $target
end
@destructure codegen(::Inv{T,3}, [source], [target]) where T = quote
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

Base.size(self::Monomials) = (self.degree + self.padding + 1, size(self.points)...)

arguments(self::Monomials) = (self.points,)
prealloc(self::Monomials{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

@destructure function codegen(self::Monomials{T}, [points], [target]) where T
    j = gensym("j")

    loopcode = Any[:(@inbounds $target[$(1+self.padding),$j] = 1.0)]
    for i in self.padding .+ (1:self.degree)
        push!(loopcode, :(@inbounds $target[$(i+1),$j] = $target[$i,$j] * $points[$j]))
    end

    quote
        $target .= zero($T)
        for $j = CartesianIndices($(size(self.points)))
            $(loopcode...)
        end
        $target
    end
end



# Neg

@autohasheq struct Neg{T,N} <: ArrayEvaluable{T,N}
    source :: ArrayEvaluable{T,N}
end

Base.size(self::Neg) = size(self.source)

arguments(self::Neg) = (self.source,)
prealloc(::Neg) = []
@destructure codegen(::Neg, [source], []) = :(-$source)



# Normalize

@autohasheq struct Normalize{T,N} <: ArrayEvaluable{T,N}
    source :: ArrayEvaluable{T,N}

    function Normalize(source::ArrayEvaluable)
        isnormal(source) && return source
        new{eltype(source), ndims(source)}(source)
    end
end

Base.size(self::Normalize) = size(self.source)

arguments(self::Normalize) = (self.source,)
prealloc(self::Normalize{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]
@destructure codegen(::Normalize, [source], [target]) = :($target .= $source; $target)



# Product

@autohasheq struct Product{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Product(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (eltype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Base.size(self::Product) = broadcast_shape((size(term) for term in self.terms)...)

arguments(self::Product) = self.terms
prealloc(self::Product{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

# Note: element-wise product!
@destructure function codegen(self::Product, args, [target])
    code = args[1]
    for arg in args[2:end]
        code = :($code .* $arg)
    end
    :($target .= $code; $target)
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

isnormal(::Reshape) = false
arguments(self::Reshape) = (self.source,)
prealloc(::Reshape) = []
@destructure codegen(self::Reshape, [source], []) = :(reshape($source, $(self.newshape...)))



# Sum

@autohasheq struct Sum{T,N} <: ArrayEvaluable{T,N}
    source :: ArrayEvaluable{T}
    dims :: Tuple{Vararg{Int}}

    function Sum(source::ArrayEvaluable, dims::Tuple{Vararg{Int}})
        @assert all(1 <= d <= ndims(source) for d in dims)
        new{eltype(source), ndims(source)}(source, dims)
    end
end

Base.size(self::Sum) = Tuple(i in self.dims ? 1 : k for (i, k) in enumerate(size(self.source)))

arguments(self::Sum) = (self.source,)
prealloc(self::Sum{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]
@destructure codegen(self::Sum, [source], [target]) = :($target .= sum($source, dims=$(self.dims)))



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

Base.size(self::Add) = broadcast_shape((size(term) for term in self.terms)...)

arguments(self::Add) = self.terms
prealloc(self::Add{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

@destructure function codegen(self::Add, args, [target])
    code = args[1]
    for arg in args[2:end]
        code = :($code .+ $arg)
    end
    :($target .= $code; $target)
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
@destructure codegen(self::Tupl, terms, []) = :($(terms...),)



# Zeros

@autohasheq struct Zeros{T,N} <: ArrayEvaluable{T,N}
    shape :: Shape
    Zeros(T, shape::Int...) = new{T,length(shape)}(shape)
end

Base.size(self::Zeros) = self.shape
separate(::Zeros) = []

arguments(::Zeros) = ()
prealloc(self::Zeros{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]
@destructure codegen(self::Zeros{T}, [], [target]) where T = :($target[:] .= zero($T); $target)
