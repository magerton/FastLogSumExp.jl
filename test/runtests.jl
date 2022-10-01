using ForwardDiff, Test, LogExpFunctions, LoopVectorization
using FastLogSumExp

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

# -------------------------------------------
# versions of stuff
# -------------------------------------------

@testset "Check fcts are correct" begin
	@testset "Floats" begin
		                     logsumexp!(        VbarF,           XF)
        bmark_F = copy(VbarF)
		@test bmark_F ≈ flse.logsumexp_simd!(   VbarF, tmp_maxF, XF)
		@test bmark_F ≈ flse.logsumexp_tricks!( VbarF, tmp_maxF, XF)
		@test bmark_F ≈ flse.logsumexp_vanilla!(VbarF, tmp_maxF, XF)
		@test bmark_F ≈ flse.logsumexp_turbo!(  VbarF, tmp_maxF, XF)
		@test bmark_F ≈ flse.logsumexp_vmap!(   VbarF, tmp_maxF, XF, XFtmp)
		@test bmark_F ≈ flse.logsumexp_tullio!( VbarF, tmp_maxF, XF)
	end
	
	@testset "Duals" begin
                             logsumexp!(        VbarD,           XD)
        bmark_D = copy(VbarD)
		@test bmark_D ≈ flse.logsumexp_simd!(   VbarD, tmp_maxD, XD)
		@test bmark_D ≈ flse.logsumexp_tricks!( VbarD, tmp_maxD, XD)
		@test bmark_D ≈ flse.logsumexp_vanilla!(VbarD, tmp_maxD, XD)
		# @test bmark_D ≈ flse.logsumexp_turbo!(  VbarD, tmp_maxD, XD)
		@test bmark_D ≈ flse.logsumexp_vmap!(   VbarD, tmp_maxD, XD, XDtmp)
		@test bmark_D ≈ flse.logsumexp_tullio!( VbarD, tmp_maxD, XD)
	end
end

VbarDbmark = logsumexp!(VbarD, XD)
flse.logsumexp_turbo2!(VbarD, tmp_maxD, XD)
@test VbarD ≈ VbarDbmark

@btime logsumexp!($VbarD, $XD);
@btime flse.logsumexp_simd!(    $VbarD, $tmp_maxD, $XD); # 37

@btime flse.logsumexp_tricks!(  $VbarD, $tmp_maxD, $XD); # 24.6
@btime flse.logsumexp_specials!($VbarD, $tmp_maxD, $XD); # 24.6
@btime flse.logsumexp_turbo2!(  $VbarD, $tmp_maxD, $XD); # 26.6

@btime flse.logsumexp_tricks!(  $VbarF, $tmp_maxF, $XF); # 12.5
@btime flse.logsumexp_specials!($VbarF, $tmp_maxF, $XF); # 12.3
@btime flse.logsumexp_turbo2!(  $VbarF, $tmp_maxF, $XF); # 2.7


@profview [flse.logsumexp_tricks!(VbarD, tmp_maxD, XD) for i in 1:10_000]