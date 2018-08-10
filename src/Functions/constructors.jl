
# Contraction

function Contract(left::ArrayEvaluable, right::ArrayEvaluable, laxis::Int, raxis::Int)
    (nl, nr) = ndims(left), ndims(right)
    tinds = collect(Any, 1:nl+nr-2)
    linds = collect(Any, 1:nl-1)
    rinds = collect(Any, nl:nl+nr-2)
    insert!(linds, laxis, :a)
    insert!(rinds, raxis, :a)
    Contract(left, right, linds, rinds, tinds)
end

function Contract(left::ArrayEvaluable, right::ArrayEvaluable, laxis::Symbol, raxis::Symbol)
    Contract(left, right, laxis == :first ? 1 : ndims(left), raxis == :first ? 1 : ndims(right))
end

Contract(left::ArrayEvaluable, right::ArrayEvaluable) = Contract(left, right, :last, :first)

function Contract(left::ArrayEvaluable, right::ArrayEvaluable, onto::Int; fromright::Bool=false)
    @assert ndims(left) == 2
    tinds = collect(Any, 1:ndims(right))
    rinds = collect(Any, 1:ndims(right))
    rinds[onto] = :a
    linds = fromright ? [:a, onto] : [onto, :a]
    Contract(left, right, linds, rinds, tinds)
end

Contract(left::ArrayEvaluable, right::ArrayEvaluable, onto::Symbol; fromright::Bool=false) =
    Contract(left, right, onto == :first ? 1 : ndims(right); fromright=fromright)

Base.:*(left::ArrayEvaluable, right::ArrayEvaluable) = Contract(left, right)
Base.:*(left::ArrayEvaluable, right) = Contract(left, asarray(right))
Base.:*(left, right::ArrayEvaluable) = Contract(asarray(left), right)

# Ensure that Inflate commutes past Contract
function Contract(left::Inflate, right::ArrayEvaluable, linds::Vector, rinds::Vector, tinds::Vector)
    # Find the index expressions associated with the axes that are contracted over,
    # apply a corresponding getindex followed by a contraction
    contr_indices = Dict(axid => left.indices[i] for (i, axid) in enumerate(linds) if axid in rinds)
    new_right = right[(get(contr_indices, axid, :) for axid in rinds)...]
    new_contraction = Contract(left.data, new_right, linds, rinds, tinds)

    # Inflate the output
    new_inds = [(axid in linds ? left.indices[indexin(axid, linds)[]] : (:)) for axid in tinds]
    new_shape = Tuple(
        axid in linds ? size(left, indexin(axid, linds)[]) : size(right, indexin(axid, rinds)[])
        for axid in tinds
    )
    Inflate(new_contraction, new_shape, new_inds...)
end



# GetIndex

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
    Inflate(new_data, Tuple(new_shape), new_iix)
end



# Inflate

Inflate(source::ArrayEvaluable, shape::Shape, indices::Vector{Index}) = Inflate(source, shape, Tuple(indices))
Inflate(source::ArrayEvaluable, shape::Shape, indices...) = Inflate(source, shape, indices)



# InsertAxis

InsertAxis(self::ArrayEvaluable, axes::Tuple{Vararg{Int}}) = InsertAxis(self, collect(Int, axes))
InsertAxis(self::ArrayEvaluable, axes::Int...) = InsertAxis(self, collect(Int, axes))

function InsertAxis(self::Constant, axes::Vector{Int})
    isempty(axes) && return self
    newshape = collect(Int, size(self))
    insertmany!(newshape, axes, 1)
    Constant(reshape(self.value, newshape...))
end

# Ensure that Inflate commutes past InsertAxis
function InsertAxis(self::Inflate, axes::Vector{Int})
    isempty(axes) && return self
    newdata = InsertAxis(self.data, axes)
    newindices = collect(Index, self.indices)
    insertmany!(newindices, axes, :)
    newshape = collect(Int, size(self))
    insertmany!(newshape, axes, 1)
    Inflate(newdata, Tuple(newshape), newindices)
end

expandleft(self::ArrayEvaluable, tdims::Int) = InsertAxis(self, fill(1, tdims - ndims(self)))
expandright(self::ArrayEvaluable, tdims::Int) = InsertAxis(self, fill(ndims(self) + 1, tdims - ndims(self)))



# Inv

Base.inv(self::ArrayEvaluable{T,2}) where T = Inv(self)



# Derivatives

function grad(self::ArrayEvaluable, geom::ArrayEvaluable)
    @assert ndims(geom) == 1
    dims = size(geom, 1)
    grad(self, dims) * inv(grad(geom, dims))
end

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
    lgrad = Contract(grad(self.left, d), self.right, [self.linds..., newsym], self.rinds, [self.tinds..., newsym])
    rgrad = Contract(self.left, grad(self.right, d), self.linds, [self.rinds..., newsym], [self.tinds..., newsym])
    Sum(lgrad, rgrad)
end

function grad(self::Monomials, d::Int)
    self.degree == 0 && return Zeros(eltype(self), size(self)..., d)
    newmono = Monomials(self.points, self.degree-1, self.padding+1)
    gradpts = InsertAxis(grad(self.points, d), 1)
    scale = Constant(Matrix(Diagonal(append!(fill(0.0, (self.padding+1,)), 1:self.degree))))
    scale * (newmono .* gradpts)
end

function grad(self::Product, d::Int)
    # Since the gradient dimension comes last, we need to explicitly broadcast first
    maxndims = maximum(ndims(term) for term in self.terms)
    vterms = [expandright(term, maxndims) for term in self.terms]
    gterms = [grad(term, d) for term in vterms]

    ret = gterms[1]
    for (i, (vt, gt)) in enumerate(zip(vterms[2:end], gterms[2:end]))
        ret = .+(ret .* vt, .*(vterms[1:i]..., gt))
    end
    ret
end

function grad(self::Sum, d::Int)
    # Since the gradient dimension comes last, we need to explicitly broadcast first
    maxndims = maximum(ndims(term) for term in self.terms)
    .+((grad(expandright(term, maxndims), d) for term in self.terms)...)
end

grad(self::Constant, d::Int) = Zeros(eltype(self), size(self)..., d)
grad(self::GetIndex, d::Int) = getindex(grad(self.value, d), self.indices..., :)
grad(self::Inflate, d::Int) = Inflate(grad(self.data, d), (size(self)..., d), (self.indices..., :))
grad(self::InsertAxis, d::Int) = InsertAxis(grad(self.source, d), self.axes)
grad(self::Inv, d::Int) = -Contract(self, self * grad(self.source, d), 2; fromright=true)
grad(self::Neg, d::Int) = -grad(self.source, d)
grad(self::Zeros, d::Int) = Zeros(eltype(self), size(self)..., d)



# Multiplication

Product(left::ArrayEvaluable, right::ArrayEvaluable) = Product((left, right))
Product(left::Constant, right::Constant) = Constant(left.value .* right.value)

function Product(left::ArrayEvaluable, right::Zeros)
    newtype = promote_type(eltype(left), eltype(right))
    newshape = broadcast_shape(size(left), size(right))
    newdims = length(newshape)
    Zeros(newtype, newshape...)
end
Product(left::Zeros, right::ArrayEvaluable) = Product(right, left)

# Ensure that Inflate commutes past Product
function Product(left::Inflate, right::Inflate)
    any(!isa(l, Colon) && !isa(r, Colon) for (l,r) in zip(left.indices, right.indices)) &&
        error("Joining inflations with overlapping axes not supported")
    ndims(left) == ndims(right) ||
        error("Joining inflations with different number of dimensions not supported")

    newdata = left.data .* right.data
    newindices = Index[isa(l, Colon) ? r : l for (l,r) in zip(left.indices, right.indices)]
    newshape = broadcast_shape(size(left), size(right))
    Inflate(newdata, newshape, newindices)
end

Broadcast.broadcasted(::typeof(*), left::ArrayEvaluable, right::ArrayEvaluable) = Product(left, right)
Broadcast.broadcasted(::typeof(*), left::ArrayEvaluable, right) = Product(left, asarray(right))
Broadcast.broadcasted(::typeof(*), left, right::ArrayEvaluable) = Product(asarray(left), right)



# Summation and negation

Sum(left::ArrayEvaluable, right::ArrayEvaluable) = Sum((left, right))
Sum(left::Constant, right::Constant) = Constant(left.value .+ right.value)

# TODO: Apply broadcasting during optimization phase?
function Sum(left::ArrayEvaluable, right::Zeros)
    any(s != 1 for s in size(right)[ndims(left)+1:end]) && return Sum((left, right))
    expandright(left, ndims(right))
end
Sum(left::Zeros, right::ArrayEvaluable) = Sum(right, left)

Broadcast.broadcasted(::typeof(+), left::ArrayEvaluable, right::ArrayEvaluable) = Sum(left, right)
Broadcast.broadcasted(::typeof(+), left::ArrayEvaluable, right) = Sum(left, asarray(right))
Broadcast.broadcasted(::typeof(+), left, right::ArrayEvaluable) = Sum(asarray(left), right)

Neg(self::Neg) = self.source

Base.:-(self::ArrayEvaluable) = Neg(self)
Broadcast.broadcasted(::typeof(-), self::ArrayEvaluable) = Neg(self)



# Miscellaneous

const elemindex = ArrayArgument{Int,0}(:(fill(element.index, ())), false, true, ())
const trans = Argument{TransformChain}(:(element.transform), false, true)

outer(left::ArrayEvaluable, right::ArrayEvaluable) = InsertAxis(left, 1) .* InsertAxis(right, 2)
outer(self::ArrayEvaluable) = outer(self, self)
rootcoords(ndims::Int) = ApplyTransform(trans, Point{ndims}(), ndims)
