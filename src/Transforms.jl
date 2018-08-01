module Transforms

using AutoHashEquals

abstract type AbstractTransform end

export Shift, applytrans

TransformChain = Tuple{Vararg{AbstractTransform}}


@auto_hash_equals struct Shift <: AbstractTransform
    offset :: Array{Float64,1}
end

fromdims(self::Shift) = length(self.offset)
todims(self::Shift) = length(self.offset)
codegen(::Type{Shift}, transform::Expr, points::Symbol) = :($points .+ $transform.offset)


@generated function applytrans(points::Array{Float64}, transforms::TransformChain)
    symbols = [gensym(string(index)) for index in 1:fieldcount(transforms)+1]

    ret = :($(symbols[1]) = points)
    for (i, fromsym, tosym) in zip(1:fieldcount(transforms), symbols[1:end-1], symbols[2:end])
        code = codegen(fieldtype(transforms, i), :(transforms[$i]), fromsym)
        ret = :($ret; $tosym = $code)
    end
    ret
end

end # module
