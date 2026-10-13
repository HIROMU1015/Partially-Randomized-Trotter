# H8保存rawからのstate事前検証

最新SOURCE `2ff3fc9f6e1091cbd9915507acfa74112c7d3055`、26限定検査。保存stateまで実完了、追加integral/DF0、successful solver matvec61、exit0/全reap/cleanup0。[統合結果](../../track_a_h48_preflight_result_v1/2026-10-13/README.md)、[完了receipt](completion_receipt.json)。未記録の停止state02 matvec正確値はnull、予約10000を保持する。以下1d6/24は前段履歴。

2026-10-13 JST。SOURCE `1d6b277d2ce92529118a0163a994b86b052f2088`、[plan/auth](auth01/manifest.json)、[24限定tests](local_tests.json)、[保存rawの診断](saved_raw_projection_diagnostic.json)。H8本格比較/trajectory/compileなし。利用者の準備後開始許可・対象1（小規模事前）に基づく。

H8_INPUTのSCF1/DF1を再実行せず、停止時に保存したintegrals/raw28 fragmentsのbytes/SHAを受理する。旧stopはValueError:HERMITIZATION_POLICY:fragment_21、source d5900669、native coordinator274999/worker275247/exit1、全reap/cleanup0。rawから一度のweighted Hermitian projectionで、1e-10 Hartree budget/9.9e-11 decision limitを変更せず、conservative primaryNを全16modes、secondary8とする。評価値eta16≈3.65950447e-12 Hartreeで基準内。binary64 Frobenius/60-digit Decimal評価は数学boundの工学的評価で、interval certificateではない。

全28 signed weights/生成順/cutoff0/定数を保持し、exact cross-spin zero/alpha-beta equalityを検査する。独立vectorized正常順係数と、旧明示pair-index再構成を照合し、source/input receiptへbindingする。SCF/rawは旧pathへwriteせず、新exclusive outputにreceiptと状態だけを保存する。旧actualIntegral1/DF1を保持し、追加Integral0/DF0、solver matvec上限10000、compile/trajectory0、返却/reset0。

state：linearH8/1Å/STO-3G/16modes/電子4+4/sector4900、HF初期eigsh(k1/SA/tol1e-12/ncv40/maxiter1000)、numba/thread1、residual1e-9+1e-10 max(1,|E|)、一度normalize/phase固定。ground-state/総数値誤差はcertificateにしない。H8 signal・16-mode compile feasibility・resource-mapは未完了。

CPU30 worker1/CPU31 coordinator、own affinity/thread1/workerAS32/RSS9/coAS8/hard32/RSS2/aggregate16GiB。H4補完の20/21/24–29・co22/23と交差しない。H4aggregate76＋H8worker9/co2/余白2＝89GiBで96GiBの共同headroom内。fresh CPU/memory/pressure/OOM/filesystem/inode/quotaとnative318file subsetを照合する。共有/venv/他job変更0、GPU query/use0。別worktreeを使い、稼働H4のsource closureを変えない。

[前段原稿・計画](../../track_a_h6_manuscript_h8_plan_v1/2026-10-13/README.md)、[H4補完準備](../../track_a_h48_preflight_preparation_v1/2026-10-13/README.md)。新SOURCEの6file bytesをGitHubから取得して照合した。今回のlocal24 testsは旧20＋新weighted fixture4、real input/solver/科学結果のPASSではない。開始結果は[receipt](launch_receipt.json)へ保存する。
