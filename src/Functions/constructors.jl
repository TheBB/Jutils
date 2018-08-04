
# getindex

Base.getindex(value::ArrayEvaluable{T,N}, indices...) where {T,N} = GetIndex(value, indices)



# insertaxis

insertaxis(source::ArrayEvaluable, axes::Vector{Int}) = InsertAxis(source, axes)

# Ensure that Inflate commutes past InsertAxis
function insertaxis(source::Inflate, axes::Vector{Int})
    indices = collect(Any, source.indices)
    shape = collect(source.size)
    insertmany!(shape, axes, 1)
    insertmany!(indices, axes, 1)
    Inflate(source.data, Tuple(indices), Tuple(shape))
end



# Miscellaneous

rootcoords(ndims::Int) = ApplyTransform(trans, Point{ndims}(), ndims)
