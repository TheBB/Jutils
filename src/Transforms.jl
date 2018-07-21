module Transforms

abstract type AbstractTransform end

export Shift, applytrans

TransformChain = Tuple{Vararg{AbstractTransform}}


struct Shift <: AbstractTransform
    offset :: Array{Float64,1}
end

fromdims(self::Shift) = length(self.offset)
todims(self::Shift) = length(self.offset)
codegen(::Type{Shift}, transform::Expr, points::Symbol) = :($points .+ $transform.offset)


apply(points::Array{Float64}, transforms::TransformChain) = apply(points, transforms...)

@generated function applytrans(points::Array{Float64}, transforms::AbstractTransform...)
    symbols = [gensym(string(index)) for index in 1:length(transforms)+1]

    ret = :($(symbols[1]) = points)
    for ((i, trf), fromsym, tosym) in zip(enumerate(transforms), symbols[1:end-1], symbols[2:end])
        code = codegen(trf, :(transforms[$i]), fromsym)
        ret = :($ret; $tosym = $code)
    end
    ret
end

end # module
