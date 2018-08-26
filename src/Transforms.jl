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


function flatten_transforms!(ret, transforms, rootexpr)
    for (i, trf) in enumerate(transforms.parameters)
        if trf <: Tuple
            flatten_transforms!(ret, trf, :($rootexpr[$i]))
        elseif trf <: AbstractTransform
            push!(ret, (trf, :($rootexpr[$i])))
        end
    end
end

function flatten_transforms(transforms)
    ret = Tuple{DataType,Expr}[]
    flatten_transforms!(ret, transforms, :transforms)
    ret
end

@generated function applytrans(transforms::Tuple, points::Vector{Float64},
                               workspace::Vector{Float64}, output::Vector{Float64})
    flat = flatten_transforms(transforms)
    trfcode = [codegen(dtype, code, :output, :workspace) for (dtype, code) in flat]

    quote
        output .= 0.0
        output[1:length(points)] = points
        $(trfcode...)
        output
    end
end

@generated function applytrans_grad(transforms::Tuple, points::Vector{Float64},
                                    ptworkspace::Vector{Float64}, ptoutput::Vector{Float64},
                                    mxworkspace::Matrix{Float64}, mxoutput::Matrix{Float64})
    flat = flatten_transforms(transforms)
    pt_trfcode = [codegen(dtype, code, :ptoutput, :ptworkspace) for (dtype, code) in flat]
    mx_trfcode = [codegen_grad(dtype, code, :ptoutput, :mxoutput, :mxworkspace) for (dtype, code) in flat]
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
