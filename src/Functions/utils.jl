
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
    fieldnames = []

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



# Indexing

legalindices(indices::Tuple) = all(
    isa(i, Int) || isa(i, Array{Int}) || isa(i, Colon) || isa(i, Evaluable{Int}) || isa(i, ArrayEvaluable{Int})
    for i in indices
)

function resultsize(origsize::Tuple{Vararg{Int}}, indices::Tuple)
    length(origsize) == length(indices) || error("Inconsistent indexing")
    Tuple(flatten(
        isa(i, Colon) ? (s,) : isa(i, Evaluable{Int}) ? () : size(i)
        for (s, i) in zip(origsize, indices)
    ))
end

function indexweave(spec::Tuple, actual)
    actual = collect(actual)
    weaved = []
    for s in spec
        if isa(s, Colon)
            push!(weaved, :(:))
        elseif isa(s, Int) || isa(s, Array{Int})
            push!(weaved, :($s))
        else
            push!(weaved, popfirst!(actual))
        end
    end
    weaved
end
