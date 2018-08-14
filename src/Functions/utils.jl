
# Macros

"""
    autohasheq(typ)

Generate hash and equality methods for a type that are sensitive to type parameters and field
values. This macro is a modified version of AutoHashEquals.jl, where type parameters are not taken
into account.
"""
macro autohasheq(typ)
    @assert typ.head == :struct

    typename = nothing
    typeargs = []
    fieldnames = Any[]

    # Deconstruct the type declaration to get name and type parameters
    declaration = typ.args[2]
    if declaration.head == :<:
        declaration = declaration.args[1]
    end
    if isa(declaration, Symbol)
        typename = declaration
    elseif declaration.head == :curly
        typename = declaration.args[1]
        typeargs = declaration.args[2:end]
    end

    # Deconstruct field names
    for line in typ.args[3].args
        isa(line, LineNumberNode) && continue
        isa(line, Symbol) && (push!(fieldnames, line); continue)
        if line.head == :(::)
            push!(fieldnames, line.args[1])
        end
    end

    # Hash function uses type name, type parameters and field values
    hashcode = :(hash($(QuoteNode(typename)), h))
    for typearg in reverse(typeargs)
        hashcode = :(hash($typearg, $hashcode))
    end
    for fieldname in reverse(fieldnames)
        hashcode = :(hash(self.$fieldname, $hashcode))
    end

    partype = isempty(typeargs) ? typename : :($typename{$(typeargs...)})
    hashcode = :(Base.hash(self::$partype, h::UInt) where {$(typeargs...)} = $hashcode)

    # Equality function uses type parameters and field values
    eqcode = :(true)
    for fieldname in reverse(fieldnames)
        eqcode = :(isequal(l.$fieldname, r.$fieldname) && $eqcode)
    end
    eqcode = :(Base.:(==)(l::$partype, r::$partype) where {$(typeargs...)} = $eqcode)

    ineqcode = isempty(typeargs) ? :() : :(Base.:(==)(l::$typename, r::$typename) = false)

    quote
        $(esc(typ))
        $(esc(hashcode))
        $(esc(eqcode))
        $(esc(ineqcode))
    end
end



# Utility type for handling several types of indexing operations

struct IndexMap
    data :: OrderedDict{Int, ArrayEvaluable{Int}}

    IndexMap(data::OrderedDict{Int, ArrayEvaluable{Int}}) = new(data)
    function IndexMap(indices::Pair...)
        new(OrderedDict{Int, ArrayEvaluable{Int}}(k => asarray(v) for (k,v) in indices))
    end
    function IndexMap(indices...)
        new(OrderedDict{Int, ArrayEvaluable{Int}}(k => asarray(v) for (k,v) in enumerate(indices) if !isa(v, Colon)))
    end
end

Base.get(self::IndexMap, i::Int, default) = get(self.data, i, default)
Base.getindex(map::IndexMap, array) = getindex(array, (get(map, i, :) for i in 1:ndims(array))...)
Base.getindex(self::IndexMap, i::Int) = get(self.data, i, :)
Base.isempty(self::IndexMap) = isempty(self.data)
Base.iterate(self::IndexMap) = iterate(self.data)
Base.iterate(self::IndexMap, state) = iterate(self.data, state)
Base.keys(self::IndexMap) = keys(self.data)
Base.ndims(self::IndexMap, i::Int) = i in keys(self.data) ? ndims(self.data[i]) : 1
Base.size(self::IndexMap, i::Int) = size(self.data[i])
Base.size(self::IndexMap, i::Int, j::Int) = size(self.data[i], j)
Base.size(self::IndexMap, s::Shape, i::Int) = i in keys(self.data) ? size(self.data[i]) : (s[i],)
Base.values(self::IndexMap) = values(self.data)

dimcheck(self::IndexMap, low::Int=0, hi::Int=typemax(Int)) =
    all(low <= ndims(self, i) <= hi for i in keys(self))

resultndims(self::IndexMap, d::Int) = reduce(+, (ndims(v)-1 for v in values(self.data)); init=0) + d
resultsize(self::IndexMap, shape::Shape) = Tuple(flatten(size(self, shape, i) for i in 1:length(shape)))

function codegen(self::IndexMap, values, d::Int)
    subst = OrderedDict(zip(keys(self), values))
    ((get(subst, i, :) for i in 1:d)...,)
end



# Miscellaneous

function insertmany!(target::Vector{T}, axes, value::T) where T
    sortaxes = reverse(sort(collect(Int, axes)))
    for ax in sortaxes
        insert!(target, ax, value)
    end
end
