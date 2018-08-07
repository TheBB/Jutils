
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

@autohasheq struct Point{N} <: ArrayEvaluable{Float64,1} end

arguments(self::Point) = ()
isconstant(::Point) = false
iselconstant(::Point) = false
Base.size(self::Point{N}) where {T,N} = (N::Int,)
optimize(self::Point) = self
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
optimize(self::ApplyTransform) = ApplyTransform(optimize(self.trans), optimize(self.arg), dims)
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
optimize(self::ApplyTransformGrad) = ApplyTransformGrad(optimize(self.trans), optimize(self.arg), dims)
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



# GetIndex
# TODO: More general indexing expressions?

@autohasheq struct GetIndex{T,N} <: ArrayEvaluable{T,N}
    value :: ArrayEvaluable
    indices :: Indices

    function GetIndex(value::ArrayEvaluable{T,N}, indices::Indices) where {T,N}
        length(indices) == ndims(value) || error("Inconsistent indexing")

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
optimize(self::GetIndex) = getindex(optimize(self.value), (optimize(i) for i in self.indices)...)
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

    function Inflate(data::ArrayEvaluable{T}, indices::Indices, shape::Shape) where T
        length(indices) == length(shape) || error("Inconsistent indexing")
        index_resultsize(shape, indices) == size(data) || error("Inconsistent dimensions")
        index_dimcheck(indices, 1, 1) || error("Multidimensional indices not supported for inflate")
        new{T,length(shape)}(data, indices, shape)
    end
end

arguments(self::Inflate) = (self.data, (i for i in self.indices if isa(i, Evaluable))...)
Base.size(self::Inflate) = self.shape
optimize(self::Inflate) = Inflate(optimize(self.data), Index[optimize(i) for i in self.indices], self.shape)
separate(self::Inflate) = [
    ((((isa(ix, Colon) ? ind : getindex(ix, ind)) for (ix, ind) in zip(self.indices, inds))...,), data)
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

    function InsertAxis(source::ArrayEvaluable{T}, axes::Vector{Int}) where T
        all(1 <= i <= ndims(source) + 1 for i in axes) || error("Axes out of range")
        new{T, ndims(source)+length(axes)}(source, sort(axes))
    end
end

arguments(self::InsertAxis) = (self.source,)
function Base.size(self::InsertAxis)
    shape = collect(Int, size(self.source))
    insertmany!(shape, self.axes, 1)
    Tuple(shape)
end
optimize(self::InsertAxis) = insertaxis(optimize(self.source), self.axes)
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
optimize(self::Inv) = inv(optimize(self.source))
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



# Matmul
# TODO: Generalize using TensorOperations.jl when compatible

@autohasheq struct Matmul{T,L,R,N} <: ArrayEvaluable{T,N}
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
Base.size(self::Matmul) = (size(self.left)[1:end-1]..., size(self.right)[2:end]...)
optimize(self::Matmul) = Matmul(optimize(self.left), optimize(self.right))
prealloc(self::Matmul{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]

function codegen(self::Matmul{T}, left, right, result) where T
    (i, j, k) = gensym("i"), gensym("j"), gensym("k")
    quote
        $result[:] .= $(zero(T))
        for $i = CartesianIndices($(size(self.left)[1:end-1])), $j = CartesianIndices($(size(self.right)[2:end]))
            @simd for $k = 1:$(size(self.right, 1))
                $result[$i,$j] += $left[$i,$k] * $right[$k,$j]
            end
        end
        $result
    end
end



# Monomials

@autohasheq struct Monomials{T,N} <: ArrayEvaluable{T,N}
    points :: ArrayEvaluable{T}
    degree :: Int

    function Monomials(points::ArrayEvaluable{T,M}, degree::Int) where {T,M}
        new{T,M+1}(points, degree)
    end
end

arguments(self::Monomials) = (self.points,)
Base.size(self::Monomials) = (self.degree + 1, size(self.points)...)
optimize(self::Monomials) = Monomials(optimize(self.points), self.degree)
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

@autohasheq struct Product{T,N} <: ArrayEvaluable{T,N}
    terms :: Tuple{Vararg{ArrayEvaluable}}

    function Product(terms::Tuple{Vararg{ArrayEvaluable}})
        newshape = broadcast_shape((size(term) for term in terms)...)
        newtype = reduce(promote_type, (arraytype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Product(terms...) = Product(terms)

arguments(self::Product) = self.terms
Base.size(self::Product) = broadcast_shape((size(term) for term in self.terms)...)
optimize(self::Product) = *(Tuple(optimize(term) for term in self.terms)...)
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
        newtype = reduce(promote_type, (arraytype(term) for term in terms))
        new{newtype, length(newshape)}(terms)
    end
end

Sum(terms...) = Sum(terms)

arguments(self::Sum) = self.terms
Base.size(self::Sum) = broadcast_shape((size(term) for term in self.terms)...)
optimize(self::Sum) = Sum(Tuple(optimize(term) for term in self.terms))
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

    function Tupl(terms::Evaluable...)
        # TODO: Is there a better way to do this?
        tupletype = eval(:(Tuple{$((restype(term) for term in terms)...)}))
        new{tupletype}(terms)
    end
end

arguments(self::Tupl) = self.terms
prealloc(self::Tupl) = []
codegen(self::Tupl, terms...) = :($(terms...),)



# Zeros

@autohasheq struct Zeros{T,N} <: ArrayEvaluable{T,N}
    shape :: Shape

    # function Zeros(shape::Shape) where T
    #     new{T,length(shape)}()
    # end
end

arguments(::Zeros) = ()
Base.size(self::Zeros) = self.shape
optimize(self::Zeros) = self
separate(::Zeros) = []
prealloc(self::Zeros{T}) where T = [:(Array{$T}(undef, $(size(self)...)))]
codegen(self::Zeros{T}, target) where T = :($target[:] .= zero($T); $target)
