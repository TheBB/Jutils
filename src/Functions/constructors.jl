
# getindex

function Base.getindex(value::ArrayEvaluable{T,N}, indices...) where {T,N}
    cleaned_indices = Index[isa(ix, Colon) ? ix : asarray(ix) for ix in indices]
    GetIndex(value, cleaned_indices)
end



# inflate

function inflate(source::ArrayEvaluable, indices, shape::Shape)
    cleaned_indices = Index[isa(ix, Colon) ? ix : asarray(ix) for ix in indices]
    Inflate(source, cleaned_indices, shape)
end



# insertaxis

insertaxis(source::ArrayEvaluable, axes::Vector{Int}) = InsertAxis(source, axes)

# Ensure that Inflate commutes past InsertAxis
function insertaxis(source::Inflate, axes::Vector{Int})
    newdata = insertaxis(source.data, axes)
    newindices = collect(Index, source.indices)
    insertmany!(newindices, axes, :)
    newshape = collect(Int, size(source))
    insertmany!(newshape, axes, 1)
    Inflate(newdata, newindices, Tuple(newshape))
end



# Multiplication

Base.:*(left::ArrayEvaluable, right::ArrayEvaluable) = Product((left, right))

# Ensure that Inflate commutes past Product
function Base.:*(left::Inflate, right::Inflate)
    any(!isa(l, Colon) && !isa(r, Colon) for (l,r) in zip(left.indices, right.indices)) &&
        error("Joining inflations with overlapping axes not supported")
    ndims(left) == ndims(right) ||
        error("Joining inflations with different number of dimensions not supported")

    newdata = left.data * right.data
    newindices = Index[isa(l, Colon) ? r : l for (l,r) in zip(left.indices, right.indices)]
    newshape = broadcast_shape(size(left), size(right))
    inflate(newdata, newindices, newshape)
end



# Miscellaneous

const elemindex = ArrayArgument{Int,0}(:(fill(element.index, ())), false, true, ())
const trans = Argument{TransformChain}(:(element.transform), false, true)

rootcoords(ndims::Int) = ApplyTransform(trans, Point{ndims}(), ndims)
