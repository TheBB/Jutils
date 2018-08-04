
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
    @show size(left.data), left.indices, size(left)
    @show size(right.data), right.indices, size(right)

    # If necessary, wrap sources in InsertAxis so they have the final number of dimensions

    Product((left, right))
end



# Miscellaneous

const elemindex = ArrayArgument{Int,0}(:(fill(element.index, ())), false, true, ())
const trans = Argument{TransformChain}(:(element.transform), false, true)

rootcoords(ndims::Int) = ApplyTransform(trans, Point{ndims}(), ndims)
