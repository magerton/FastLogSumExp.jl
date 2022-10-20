using ForwardDiff, Test, LogExpFunctions, LoopVectorization
using FastLogSumExp
using BenchmarkTools

const FD = ForwardDiff
const flse = FastLogSumExp

# -------------------------------------------
# setup
# -------------------------------------------

n,k = 1000, 3
randX = rand(n,k)
theta0 = randn(k)

cfg = ForwardDiff.GradientConfig(Nothing, theta0)
thetad = cfg.duals
ForwardDiff.seed!(thetad, theta0, cfg.seeds)

XF = randX.*theta0'
XD = randX.*thetad'

XFtmp = similar(XF)
XDtmp = similar(XD)

VbarF = Vector{eltype(XF)}(undef, n)
VbarD = Vector{eltype(XD)}(undef, n)

tmp_maxF = similar(VbarF)
tmp_maxD = similar(VbarD)

tmp_cart = Vector{CartesianIndex{2}}(undef, n)


X1D = XD[:,1]
X1F = XF[:,1]
tmp = zeros(n)

# -------------------------------------------
# vector versions
# -------------------------------------------

@testset "check vectors" begin
	@test flse.vec_logsumexp_float_turbo(X1F)         ≈ logsumexp(X1F)
	@test flse.vec_logsumexp_dual_reinterp(X1D)       ≈ logsumexp(X1D)
	@test flse.vec_logsumexp_dual_reinterp!(tmp, X1D) ≈ logsumexp(X1D)
end

# -------------------------------------------
# matrix versions
# -------------------------------------------

@testset "Check matrix" begin
	@testset "Floats" begin
		                         logsumexp!(        VbarF,           XF)
        bmark_F = copy(VbarF)
		@test bmark_F ≈ flse.mat_logsumexp_vexp_log_fast!(VbarF, tmp_maxF, XF)
		@test bmark_F ≈ flse.mat_logsumexp_float_turbo!(  VbarF, tmp_maxF, XF)
	end
	
	@testset "Duals" begin
                                 logsumexp!(              VbarD,           XD)
        bmark_D = copy(VbarD)
		@test bmark_D ≈ flse.mat_logsumexp_vexp_log_fast!(VbarD, tmp_maxD,        XD)
		@test bmark_D ≈ flse.mat_logsumexp_dual_reinterp!(VbarD, tmp_maxF, XFtmp, XD)
	end
end

# -------------------------------------------
# Benchmarking
# -------------------------------------------

bg = BenchmarkGroup()

bg["V"] = BenchmarkGroup(["V", "Vector" ])
bg["V"]["Float64"] = BenchmarkGroup(["Float64"])
bg["V"]["Dual"]    = BenchmarkGroup(["Dual"])

bg["V"]["Float64"]["LogExpFunctions"] = @benchmarkable logsumexp($X1F)
bg["V"]["Float64"]["Turbo"]           = @benchmarkable flse.vec_logsumexp_float_turbo($X1F)

bg["V"]["Dual"]["LogExpFunctions"] = @benchmarkable logsumexp(X1D)
bg["V"]["Dual"]["Reinterp"]        = @benchmarkable flse.vec_logsumexp_dual_reinterp!($tmp, $X1D)
bg["V"]["Dual"]["Reinterp no tmp"] = @benchmarkable flse.vec_logsumexp_dual_reinterp($X1D)


bg["M"] = BenchmarkGroup(["M" ,"Matrix"])
bg["M"]["Float64"] = BenchmarkGroup(["Float64"])
bg["M"]["Dual"]    = BenchmarkGroup(["Dual"])

bg["M"]["Float64"]["LogExpFunctions"] = @benchmarkable                        logsumexp!($VbarF,            $XF);
bg["M"]["Float64"]["Fast LogExp"]     = @benchmarkable flse.mat_logsumexp_vexp_log_fast!($VbarF, $tmp_maxF, $XF);
bg["M"]["Float64"]["Turbo"]           = @benchmarkable flse.mat_logsumexp_float_turbo!(  $VbarF, $tmp_maxF, $XF);

bg["M"]["Dual"]["LogExpFunctions"] = @benchmarkable                        logsumexp!($VbarD,                    $XD);
bg["M"]["Dual"]["Fast LogExp"]     = @benchmarkable flse.mat_logsumexp_vexp_log_fast!($VbarD, $tmp_maxD,         $XD);
bg["M"]["Dual"]["Reinterp"]        = @benchmarkable flse.mat_logsumexp_dual_reinterp!($VbarD, $tmp_maxF, $XFtmp, $XD);

results = run(bg, verbose=true)

println(results)
