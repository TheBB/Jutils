
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

insertaxis(source::ArrayEvaluable, axes::Vector{Int}) = isempty(axes) ? source : InsertAxis(source, axes)

# Ensure that Inflate commutes past InsertAxis
function insertaxis(source::Inflate, axes::Vector{Int})
    isempty(axes) && return source
    newdata = insertaxis(source.data, axes)
    newindices = collect(Index, source.indices)
    insertmany!(newindices, axes, :)
    newshape = collect(Int, size(source))
    insertmany!(newshape, axes, 1)
    Inflate(newdata, newindices, Tuple(newshape))
end



# inv

Base.inv(source::ArrayEvaluable{T,2}) where T = Inv(source)



# grad

function grad(self::ApplyTransform, d::Int)
    self.dims == d || error("Inconsistent number of dimension")
    ApplyTransformGrad(self.trans, self.arg, self.dims)
end

function grad(self::Point{N}, d::Int) where N
    N == d || error("Inconsistent number of dimensions")
    Constant(Matrix(1.0I, N, N))
end

function grad(self::Product, d::Int)
    # Since the gradient dimension comes last, we need to explicitly broadcast first
    maxndims = maximum(ndims(term) for term in self.terms)
    vterms = [insertaxis(term, fill(ndims(term) + 1, maxndims - ndims(term))) for term in terms]
    gterms = [grad(term, d) for term in vterms]

    ret = gterms[1]
    for (i, (vt, gt)) in enumerate(zip(vterms, gterms))
        ret = Sum(Product(ret, vt), Product(vterms[1:i]..., gt))
    end
    ret
end

function grad(self::Sum, d::Int)
    # Since the gradient dimension comes last, we need to explicitly broadcast first
    maxndims = maximum(ndims(term) for term in self.terms)
    terms = Tuple(grad(insertaxis(term, fill(ndims(term) + 1, maxndims - ndims(term))), d) for term in self.terms)
    Sum(terms)
end

grad(self::Constant{T}, d::Int) where T = Zeros{T}((size(self)..., d))
grad(self::GetIndex, d::Int) = getindex(grad(self.value, d), (self.indices..., :))
grad(self::Inflate, d::Int) = inflate(grad(self.data, d), (self.indices..., :), (size(self)..., d))
grad(self::InsertAxis, d::Int) = insertaxis(grad(self.source, d), self.axes)
grad(self::Zeros{T}, d::Int) where T = Zeros{T}((size(self)..., d))



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
