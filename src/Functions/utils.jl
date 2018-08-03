legalindices(indices::Tuple) = all(
    isa(i, Int) || isa(i, Array{Int}) || isa(i, Colon) || isa(i, Evaluable{Int}) || isa(i, ArrayEvaluable{Int})
    for i in indices
)

function resultsize(origsize::Tuple{Vararg{Int}}, indices::Tuple)
    length(origsize) == length(indices) || error("Inconsistent indexing")
    Tuple(Iterators.flatten(
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
