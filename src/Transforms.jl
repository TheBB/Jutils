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
codegen(::Type{Shift}, transform::Expr, out::Symbol, ::Symbol) = :($out .+= $transform.offset)
codegen_grad(::Type{Shift}, ::Expr, ::Symbol, ::Symbol, ::Symbol) = :()


@auto_hash_equals struct Updim{N,D} <: AbstractTransform
    value :: Float64
end

fromdims(self::Updim{N}) where N = N
todims(self::Updim{N}) where N = N + 1

function codegen(::Type{Updim{N,D}}, transform::Expr, out::Symbol, _) where {N,D}
    quote
        $out[$(D+1):$(N+1)] .= $out[$D:$N]
        $out[$D] = $transform.value
    end
end

function codegen_grad(::Type{Updim{1,D}}, transform::Expr, pt::Symbol, out::Symbol, ::Symbol) where D
    quote
        $out[$(D+1):2, :] .= $out[$D:1, :]
        $out[$D, :] .= 0.0
        $out[1,2] = $out[2,1]
        $out[2,2] = -$out[1,1]
    end
end


@generated function applytrans(transforms::TransformChain, points::Vector{Float64},
                               workspace::Vector{Float64}, output::Vector{Float64})
    trfcode = [
        codegen(fieldtype(transforms, i), :(transforms[$i]), :output, :workspace)
        for i in 1:fieldcount(transforms)
    ]

    quote
        output .= 0.0
        output[1:length(points)] = points
        $(trfcode...)
        output
    end
end

@generated function applytrans_grad(transforms::TransformChain, points::Vector{Float64},
                                    ptworkspace::Vector{Float64}, ptoutput::Vector{Float64},
                                    mxworkspace::Matrix{Float64}, mxoutput::Matrix{Float64})
    pt_trfcode = [
        codegen(fieldtype(transforms, i), :(transforms[$i]), :ptoutput, :ptworkspace)
        for i in 1:fieldcount(transforms)
    ]
    mx_trfcode = [
        codegen_grad(fieldtype(transforms, i), :(transforms[$i]), :ptoutput, :mxoutput, :mxworkspace)
        for i in 1:fieldcount(transforms)
    ]

    trfcode = [:($mx; $pt) for (pt, mx) in zip(pt_trfcode, mx_trfcode)]

    quote
        ptoutput .= 0.0
        ptoutput[1:length(points)] = points
        copyto!(mxoutput, I)
        $(trfcode...)
        mxoutput
    end
end

end # module
