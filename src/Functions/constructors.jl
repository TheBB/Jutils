
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
    contr_indices = Dict(axid => get(left.map, i, :) for (i, axid) in enumerate(linds))
    new_right = right[(get(contr_indices, axid, :) for axid in rinds)...]
    new_contraction = Contract(left.data, new_right, linds, rinds, tinds)

    # Mappings from name to axis index for both the left and right
    lmap = Dict(k => v for (v,k) in enumerate(linds))
    rmap = Dict(k => v for (v,k) in enumerate(rinds))

    # Inflate the output
    new_inds = IndexMap((k => left.map[lmap[k]] for k in tinds if k in keys(lmap))...)
    new_shape = Tuple(axid in keys(lmap) ? size(left, lmap[axid]) : size(right, rmap[axid]) for axid in tinds)
    Inflate(new_contraction, new_shape, new_inds)
end



# GetIndex

Base.getindex(self::ArrayEvaluable, indices...) = GetIndex(self, IndexMap(indices...))

function Base.getindex(self::Constant, indices...)
    indices = Index[isa(ix, Colon) ? ix : asarray(ix) for ix in indices]
    if all(isa(ix, Colon) || isa(ix, Constant) for ix in indices)
        Constant(self.value[(isa(i, Colon) ? i : i.value for i in indices)...])
    else
        GetIndex(self, IndexMap(indices...))
    end
end

# Ensure that Inflate commutes past GetIndex
function Base.getindex(self::Inflate, indices...)
    map = IndexMap(indices...)
    @assert isempty(keys(self.map) ∩ keys(map))
    @assert dimcheck(map, 0, 1)

    new_inflatemap = Pair[]
    new_shape = Int[]
    skipped_axes = 0
    for i in 1:ndims(self)
        i ∉ keys(self.map) && i ∉ keys(map) && (push!(new_shape, size(self, i)); continue)
        if i in keys(self.map)
            push!(new_inflatemap, (i - skipped_axes) => self.map[i])
            push!(new_shape, size(self, i))
        elseif ndims(map, i) == 1
            push!(new_shape, size(map, i, 1))
        elseif ndims(map, i) == 0
            skipped_axes += 1
        end
    end

    new_data = isempty(map) ? self.data : map[self.data]
    Inflate(new_data, Tuple(new_shape), IndexMap(new_inflatemap...))
end



# Inflate

Inflate(source::ArrayEvaluable, shape::Shape, indices::Vector{Index}) = Inflate(source, shape, IndexMap(indices...))
Inflate(source::ArrayEvaluable, shape::Shape, indices...) = Inflate(source, shape, IndexMap(indices...))



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
    newindices = IndexMap((k + sum(axes .<= k) => v for (k, v) in self.map)...)
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

function grad(self::Contract, d::Int)
    newsym = gensym("temp")
    lgrad = Contract(grad(self.left, d), self.right, [self.linds..., newsym], self.rinds, [self.tinds..., newsym])
    rgrad = Contract(self.left, grad(self.right, d), self.linds, [self.rinds..., newsym], [self.tinds..., newsym])
    Add(lgrad, rgrad)
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

function grad(self::Add, d::Int)
    # Since the gradient dimension comes last, we need to explicitly broadcast first
    maxndims = maximum(ndims(term) for term in self.terms)
    .+((grad(expandright(term, maxndims), d) for term in self.terms)...)
end

grad(self::Constant, d::Int) = Zeros(eltype(self), size(self)..., d)
grad(self::GetIndex, d::Int) = getindex(grad(self.value, d), self.indices..., :)
grad(self::Inflate, d::Int) = Inflate(grad(self.data, d), (size(self)..., d), self.map)
grad(self::InsertAxis, d::Int) = InsertAxis(grad(self.source, d), self.axes)
grad(self::Inv, d::Int) = -Contract(self, self * grad(self.source, d), 2; fromright=true)
grad(self::Neg, d::Int) = -grad(self.source, d)
grad(self::Reshape, d::Int) = reshape(grad(self.source, d), self.newshape..., d)
grad(self::Zeros, d::Int) = Zeros(eltype(self), size(self)..., d)



# Reshape

function Base.reshape(self::ArrayEvaluable, newshape::Int...)
    newshape == size(self) && return self
    Reshape(self, newshape)
end



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
    isempty(keys(left.map) ∩ keys(right.map)) || error("Joining inflations with overlapping axes not supported")
    ndims(left) == ndims(right) || error("Joining inflations with different number of dimensions not supported")

    newdata = left.data .* right.data
    newmap = IndexMap(left.map..., right.map...)
    newshape = broadcast_shape(size(left), size(right))
    Inflate(newdata, newshape, newmap)
end

Broadcast.broadcasted(::typeof(*), left::ArrayEvaluable) = left
Broadcast.broadcasted(::typeof(*), left::ArrayEvaluable, right::ArrayEvaluable) = Product(left, right)
Broadcast.broadcasted(::typeof(*), left::ArrayEvaluable, right) = Product(left, asarray(right))
Broadcast.broadcasted(::typeof(*), left, right::ArrayEvaluable) = Product(asarray(left), right)



# Summation and negation

Add(left::ArrayEvaluable, right::ArrayEvaluable) = Add((left, right))
Add(left::Constant, right::Constant) = Constant(left.value .+ right.value)

# TODO: Apply broadcasting during optimization phase?
function Add(left::ArrayEvaluable, right::Zeros)
    any(s != 1 for s in size(right)[ndims(left)+1:end]) && return Add((left, right))
    expandright(left, ndims(right))
end
Add(left::Zeros, right::ArrayEvaluable) = Add(right, left)

Broadcast.broadcasted(::typeof(+), left::ArrayEvaluable) = left
Broadcast.broadcasted(::typeof(+), left::ArrayEvaluable, right::ArrayEvaluable) = Add(left, right)
Broadcast.broadcasted(::typeof(+), left::ArrayEvaluable, right) = Add(left, asarray(right))
Broadcast.broadcasted(::typeof(+), left, right::ArrayEvaluable) = Add(asarray(left), right)

Neg(self::Neg) = self.source

Base.:-(self::ArrayEvaluable) = Neg(self)
Broadcast.broadcasted(::typeof(-), self::ArrayEvaluable) = Neg(self)



# Dimension reduction

Base.dropdims(self::ArrayEvaluable, dims) = DropDims(self, Tuple(dims))

# Ensure that Inflate commutes past DropDims
function Base.dropdims(self::Inflate, dims)
    dims = collect(dims)
    newdata = dropdims(self.data, dims)
    newmap = IndexMap((k + sum(dims .< k) => v for (k, v) in self.map if k ∉ dims)...)
    newshape = Tuple(k for (i, k) in enumerate(size(self)) if i ∉ dims)
    Inflate(newdata, newshape, newmap)
end

Base.sum(self::ArrayEvaluable, dims) = Sum(self, Tuple(dims))

# Ensure that Inflate commutes past Sum
function Base.sum(self::Inflate, dims)
    dims = collect(dims)
    newdata = sum(self.data, dims)
    newmap = IndexMap((k => v for (k, v) in self.map if k ∉ dims)...)
    newshape = Tuple(i in dims ? 1 : k for (i, k) in enumerate(size(self)))
    Inflate(newdata, newshape, newmap)
end



# Miscellaneous

const element = Argument{Element}(:element, false, true)
const elemindex = ArrayArgument{Int,0}(:(fill(element.index, ())), false, true, ())
const dimtrans = Argument{TransformChain}(:(element.dimcorr), false, true)
const fulltrans = Argument{TransformChain}(:((element.dimcorr..., element.transform...)), false, true)
const _point = Argument{Vector{Float64}}(:point, false, false)
point(n::Int) = ApplyTransform(dimtrans, _point, n)

outer(left::ArrayEvaluable, right::ArrayEvaluable) = InsertAxis(left, 1) .* InsertAxis(right, 2)
outer(self::ArrayEvaluable) = outer(self, self)
rootcoords(ndims::Int) = ApplyTransform(fulltrans, _point, ndims)
