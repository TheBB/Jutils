using Jutils.Functions
using Jutils.Integration
using Jutils.Mesh
using Jutils.Topologies


function bench_overhead!(suite)
    func = Constant(1.0)
    suite["cmp"]["constant"] = @benchmarkable compile($func)
end

function bench_1d_lag_mass!(suite)
    key = ("1d", "lag", "mass")
    domain, geom = line(1000)

    suite["lpe"]["1d"] = @benchmarkable collect($domain)

    mass = outer(mkbasis(domain, Lagrange, 1))

    suite["cmp"][(key..., "den")] = @benchmarkable compile($mass; dense=true)
    suite["cmp"][(key..., "spr")] = @benchmarkable compile($mass; dense=false)

    dmass = compile(mass; dense=true)
    smass = compile(mass; dense=false)

    suite["mke"][(key..., "den")] = @benchmarkable callable($dmass)
    suite["mke"][(key..., "spr")] = @benchmarkable callable($smass)

    suite["itg"][(key..., "den")] = @benchmarkable integrate($dmass, $domain, 5)
    suite["itg"][(key..., "spr")] = @benchmarkable integrate($smass, $domain, 5)
end

function bench_1d_lag_lapl!(suite)
    key = ("1d", "lag", "lapl")
    domain, geom = line(1000)

    mass = outer(grad(mkbasis(domain, Lagrange, 1), geom)[:,1])

    suite["cmp"][(key..., "spr")] = @benchmarkable compile($mass; dense=false)

    smass = compile(mass; dense=false)

    suite["mke"][(key..., "spr")] = @benchmarkable callable($smass)
    suite["itg"][(key..., "spr")] = @benchmarkable integrate($smass, $domain, 5)
end

function bench_2d_lag_mass!(suite)
    key = ("2d", "lag", "mass")
    domain, geom = rectilinear(30, 30)

    suite["lpe"]["2d"] = @benchmarkable collect($domain)

    mass = outer(mkbasis(domain, Lagrange, 1))

    suite["cmp"][(key..., "den")] = @benchmarkable compile($mass; dense=true)
    suite["cmp"][(key..., "spr")] = @benchmarkable compile($mass; dense=false)

    dmass = compile(mass; dense=true)
    smass = compile(mass; dense=false)

    suite["mke"][(key..., "den")] = @benchmarkable callable($dmass)
    suite["mke"][(key..., "spr")] = @benchmarkable callable($smass)

    suite["itg"][(key..., "den")] = @benchmarkable integrate($dmass, $domain, 5)
    suite["itg"][(key..., "spr")] = @benchmarkable integrate($smass, $domain, 5)
end

function bench_2d_lag_lapl!(suite)
    key = ("2d", "lag", "lapl")
    domain, geom = rectilinear(30, 30)

    mass = outer(grad(mkbasis(domain, Lagrange, 1), geom))
    mass = dropdims(sum(mass, (3,)), (3,))

    suite["cmp"][(key..., "spr")] = @benchmarkable compile($mass; dense=false)

    smass = compile(mass; dense=false)

    suite["mke"][(key..., "spr")] = @benchmarkable callable($smass)
    suite["itg"][(key..., "spr")] = @benchmarkable integrate($smass, $domain, 5)
end

function runbench()
    suite = BenchmarkGroup()
    suite["cmp"] = BenchmarkGroup()
    suite["itg"] = BenchmarkGroup()
    suite["lpe"] = BenchmarkGroup()
    suite["mke"] = BenchmarkGroup()

    bench_overhead!(suite)
    bench_1d_lag_mass!(suite)
    bench_1d_lag_lapl!(suite)
    bench_2d_lag_mass!(suite)
    bench_2d_lag_lapl!(suite)

    tune!(suite)
    return run(suite)
end
