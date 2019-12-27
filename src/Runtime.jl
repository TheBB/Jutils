module Runtime

export UnsafeView, legal_unsafe_index


struct UnsafeView{T,N} <: AbstractArray{T,N}
    offset :: Int
    stride :: Int
    length :: Int
    ptr :: Ptr{T}
    shape :: NTuple{N,Int}
end

@generated function UnsafeView(master::K, I...) where {T, N, K <: StridedArray{T,N}}
    varinds = [i for (i, k) in enumerate(I) if k == Colon]

    @assert all(k == Colon || ndims(k) == 0 for k in I)
    if !isempty(varinds)
        @assert all(k != Colon for k in I[1:varinds[1]-1])
        @assert all(k != Colon for k in I[varinds[end]+1:end])
    end

    firstindex = Any[k == Colon ? 1 : :(I[$i][]) for (i, k) in enumerate(I)]

    if isempty(varinds)
        quote
            offset = LinearIndices(size(master))[$(firstindex...)] - 1
            UnsafeView(offset, 0, 1, pointer(master), ())
        end
    else
        quote
            offset = LinearIndices(size(master))[$(firstindex...)] - 1
            stride = strides(master)[$(varinds[1])]
            shape = size(master)
            newshape = ($((:(shape[$i]) for i in varinds)...),)
            UnsafeView(offset, stride, prod(newshape), pointer(master), newshape)
        end
    end
end

function legal_unsafe_index(I...)
    all(isa(k, Colon) || ndims(k) == 0 for k in I) || return false
    all(!isa(k, Colon) for k in I) && return true

    varinds = [i for (i, k) in enumerate(I) if isa(k, Colon)]
    varinds == varinds[1]:varinds[end] || return false

    true
end

@inline Base.length(self::UnsafeView) = self.length
@inline Base.size(self::UnsafeView) = self.shape

@inline Base.IndexStyle(::Type{V}) where V <: UnsafeView = Base.IndexLinear()
@inline Base.getindex(self::UnsafeView, idx) = unsafe_load(self.ptr, self.offset + self.stride * (idx - 1) + 1)
@inline Base.setindex(self::UnsafeView, value, idx) = unsafe_store!(self.ptr, value, self.offset + self.stride * (idx - 1) + 1)

end # module
