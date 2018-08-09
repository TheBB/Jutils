
# getindex

function Base.getindex(self::ArrayEvaluable, indices...)
    cleaned_indices = Index[isa(ix, Colon) ? ix : asarray(ix) for ix in indices]
    GetIndex(self, cleaned_indices)
end

function Base.getindex(self::Constant, indices...)
    cleaned_indices = Index[isa(ix, Colon) ? ix : asarray(ix) for ix in indices]
    if all(isa(ix, Colon) || isa(ix, Constant) for ix in cleaned_indices)
        Constant(self.value[(isa(i, Colon) ? i : i.value for i in cleaned_indices)...])
    else
        GetIndex(self, cleaned_indices)
    end
end

# Ensure that Inflate commutes past GetIndex
function Base.getindex(self::Inflate, indices...)
    indices = Index[isa(ix, Colon) ? ix : asarray(ix) for ix in indices]

    @assert all(isa(x, Colon) || isa(y, Colon) for (x, y) in zip(self.indices, indices))
    @assert index_dimcheck(indices, 0, 1)

    new_iix, new_gix, new_shape = Vector{Index}(), Vector{Index}(), Vector{Int}()
    for (gix, iix, shp) in zip(indices, self.indices, size(self))
        if isa(gix, Colon) && isa(iix, Colon)
            push!(new_iix, :)
            push!(new_gix, :)
            push!(new_shape, shp)
        elseif isa(gix, Colon) && ndims(iix) == 1
            push!(new_iix, iix)
            push!(new_gix, :)
            push!(new_shape, shp)
        elseif ndims(gix) == 0 && isa(iix, Colon)
            push!(new_gix, gix)
        elseif ndims(gix) == 1 && isa(iix, Colon)
            push!(new_iix, :)
            push!(new_gix, gix)
            push!(new_shape, size(gix, 1))
        else
            @assert false
        end
    end

    new_data = all(isa(gix, Colon) for gix in new_gix) ? self.data : getindex(self.data, new_gix...)
    inflate(new_data, new_iix, Tuple(new_shape))
end



# inflate

function inflate(source::ArrayEvaluable, indices, shape::Shape)
    cleaned_indices = Index[isa(ix, Colon) ? ix : asarray(ix) for ix in indices]
    Inflate(source, cleaned_indices, shape)
end



# insertaxis

insertaxis(self::ArrayEvaluable, axes::Vector{Int}) = isempty(axes) ? self : InsertAxis(self, axes)
function insertaxis(self::Constant, axes::Vector{Int})
    isempty(axes) && return self
    newshape = collect(Int, size(self))
    insertmany!(newshape, axes, 1)
    Constant(reshape(self.value, newshape...))
end

# Ensure that Inflate commutes past InsertAxis
function insertaxis(self::Inflate, axes::Vector{Int})
    isempty(axes) && return self
    newdata = insertaxis(self.data, axes)
    newindices = collect(Index, self.indices)
    insertmany!(newindices, axes, :)
    newshape = collect(Int, size(self))
    insertmany!(newshape, axes, 1)
    Inflate(newdata, newindices, Tuple(newshape))
end

expandleft(self::ArrayEvaluable, tdims::Int) = insertaxis(self, fill(1, tdims - ndims(self)))
expandright(self::ArrayEvaluable, tdims::Int) = insertaxis(self, fill(ndims(self) + 1, tdims - ndims(self)))



# inv

Base.inv(self::ArrayEvaluable{T,2}) where T = Inv(self)



# grad

function grad(self::ApplyTransform, d::Int)
    self.dims == d || error("Inconsistent number of dimension")
    ApplyTransformGrad(self.trans, self.arg, self.dims)
end

function grad(self::Point{N}, d::Int) where N
    N == d || error("Inconsistent number of dimensions")
    Constant(Matrix(1.0I, N, N))
end

function grad(self::Contract, d::Int)
    newsym = gensym("temp")
    lgrad = Contract(grad(self.left, d), self.right, (self.linds..., newsym), self.rinds, (self.tinds..., newsym))
    rgrad = Contract(self.left, grad(self.right, d), self.linds, (self.rinds..., newsym), (self.tinds..., newsym))
    Sum(lgrad, rgrad)
end

function grad(self::Inv, d::Int)
    sgrad = grad(self.source, d)
    temp = Contract(self, sgrad, (1, 2), (2, 3, 4), (1, 3, 4))
    -Contract(temp, self, (1, 2, 3), (2, 4), (1, 4, 3))
end

function grad(self::Monomials, d::Int)
    self.degree == 0 && return Zeros(eltype(self), size(self)..., d)
    newmono = Monomials(self.points, self.degree-1, self.padding+1)
    gradpts = insertaxis(grad(self.points, d), [1])
    scale = Constant(Matrix(Diagonal(append!(fill(0.0, (self.padding+1,)), 1:self.degree))))
    Contract(scale, newmono * gradpts)
end

function grad(self::Product, d::Int)
    # Since the gradient dimension comes last, we need to explicitly broadcast first
    maxndims = maximum(ndims(term) for term in self.terms)
    vterms = [expandright(term, maxndims) for term in self.terms]
    gterms = [grad(term, d) for term in vterms]

    ret = gterms[1]
    for (i, (vt, gt)) in enumerate(zip(vterms[2:end], gterms[2:end]))
        ret = +(ret * vt, *(vterms[1:i]..., gt))
    end
    ret
end

function grad(self::Sum, d::Int)
    # Since the gradient dimension comes last, we need to explicitly broadcast first
    maxndims = maximum(ndims(term) for term in self.terms)
    +((grad(expandright(term, maxndims), d) for term in self.terms)...)
end

grad(self::Constant, d::Int) = Zeros(eltype(self), size(self)..., d)
grad(self::GetIndex, d::Int) = getindex(grad(self.value, d), self.indices..., :)
grad(self::Inflate, d::Int) = inflate(grad(self.data, d), (self.indices..., :), (size(self)..., d))
grad(self::InsertAxis, d::Int) = insertaxis(grad(self.source, d), self.axes)
grad(self::Neg, d::Int) = -grad(self.source, d)
grad(self::Zeros, d::Int) = Zeros(eltype(self), size(self)..., d)



# Multiplication

Base.:*(left::ArrayEvaluable, right::ArrayEvaluable) = Product((left, right))
Base.:*(left::ArrayEvaluable, right) = Product((left, asarray(right)))
Base.:*(left, right::ArrayEvaluable) = Product((asarray(left), right))
Base.:*(left::Constant, right::Constant) = Constant(left.value .* right.value)

function Base.:*(left::ArrayEvaluable, right::Zeros)
    newtype = promote_type(eltype(left), eltype(right))
    newshape = broadcast_shape(size(left), size(right))
    newdims = length(newshape)
    Zeros(newtype, newshape...)
end
Base.:*(left::Zeros, right::ArrayEvaluable) = right * left

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



# Summation

Base.:+(left::ArrayEvaluable, right::ArrayEvaluable) = Sum((left, right))
Base.:+(left::ArrayEvaluable, right) = Sum((left, asarray(right)))
Base.:+(left, right::ArrayEvaluable) = Sum((asarray(left), right))
Base.:+(left::Constant, right::Constant) = Constant(left.value .+ right.value)

# TODO: Apply broadcasting during optimization phase?
function Base.:+(left::ArrayEvaluable, right::Zeros)
    any(s != 1 for s in size(right)[ndims(left)+1:end]) && return Sum((left, right))
    expandright(left, ndims(right))
end
Base.:+(left::Zeros, right::ArrayEvaluable) = right + left

Base.:-(self::ArrayEvaluable) = Neg(self)
Base.:-(self::Neg) = self.source
Base.:-(left::ArrayEvaluable, right::ArrayEvaluable) = +(left, -right)
Base.:-(left::ArrayEvaluable, right) = +(left, -right)
Base.:-(left, right::ArrayEvaluable) = +(-left, right)



# Miscellaneous

const elemindex = ArrayArgument{Int,0}(:(fill(element.index, ())), false, true, ())
const trans = Argument{TransformChain}(:(element.transform), false, true)

outer(left::ArrayEvaluable, right::ArrayEvaluable) = insertaxis(left, [1]) * insertaxis(right, [2])
outer(self::ArrayEvaluable) = outer(self, self)
rootcoords(ndims::Int) = ApplyTransform(trans, Point{ndims}(), ndims)
