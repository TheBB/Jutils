module Transforms

using AutoHashEquals
using LinearAlgebra

export applytrans, applytrans_grad
export AbstractTransform
export Shift, Updim


abstract type AbstractTransform end

TransformChain = Tuple{Vararg{AbstractTransform}}


@auto_hash_equals struct Shift <: AbstractTransform
    offset :: Array{Float64,1}
end

fromdims(self::Shift) = length(self.offset)
todims(self::Shift) = length(self.offset)
codegen(::Type{Shift}, transform::Expr, points::Symbol) = :($points .+ $transform.offset)
codegen_grad(::Type{Shift}, ::Expr, ::Symbol, jac::Symbol) = jac


@auto_hash_equals struct Updim{N,D} <: AbstractTransform
    value :: Float64
end

fromdims(self::Updim{N}) where N = N
todims(self::Updim{N}) where N = N + 1

function codegen(::Type{Updim{N,D}}, transform::Expr, points::Symbol) where {N,D}
    exprs = [:($points[$i]) for i in 1:N]
    insert!(exprs, D, :($transform.value))
    :([$(exprs...)])
end

function codegen_grad(::Type{Updim{1,D}}, transform::Expr, points::Symbol, jac::Symbol) where D
    z = gensym()
    quote
        $z = [$jac[1:$(D-1),:]; zeros(1); $jac[$D:end,:]]
        [$z[1,:] $z[2,1]; $z[2,:] -$z[1,1]]
    end
end


@generated function applytrans(points::Array{Float64}, transforms::TransformChain)
    symbols = [gensym(string(index)) for index in 1:fieldcount(transforms)+1]

    ret = :($(symbols[1]) = points)
    for (i, fromsym, tosym) in zip(1:fieldcount(transforms), symbols[1:end-1], symbols[2:end])
        code = codegen(fieldtype(transforms, i), :(transforms[$i]), fromsym)
        ret = :($ret; $tosym = $code)
    end
    :($ret; return $(symbols[end]))
end

@generated function applytrans_grad(points::Array{Float64}, transforms::TransformChain)
    ptsyms = [gensym(string(index)) for index in 1:fieldcount(transforms)+1]
    mxsyms = [gensym(string(index)) for index in 1:fieldcount(transforms)+1]

    ret = :($(ptsyms[1]) = points; $(mxsyms[1]) = Matrix(1.0I, length(points), length(points)))
    iter = zip(1:fieldcount(transforms), ptsyms[1:end-1], ptsyms[2:end], mxsyms[1:end-1], mxsyms[2:end])
    for (i, fromptsym, toptsym, frommxsym, tomxsym) in iter
        mxcode = codegen_grad(fieldtype(transforms, i), :(transforms[$i]), fromptsym, frommxsym)
        ptcode = codegen(fieldtype(transforms, i), :(transforms[$i]), fromptsym)
        ret = :($ret; $tomxsym = $mxcode; $toptsym = $ptcode)
    end
    :($ret; return $(mxsyms[end]))
end

end # module
