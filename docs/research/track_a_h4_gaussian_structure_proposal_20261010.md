# H4 Gaussian構造化候補：現実的なcompile時間へ向けた限定修正

run09の実測は約42分で4/74784wrapper、signal2/1308。単純全件外挿は約548日であり、後半の処理量/cache再利用が未実測なので確定ETAではない。
72hは完了予測ではなく停止上限。72h内完了にはその時点の平均速度から約184倍の増速が必要で、通常のqueue/JSON/worker微調整だけで達成済みとは扱えない。
実保存q1/q2のcompiled circuit_sizeは約273万/507万、run09はこのsourceのまま動作中。今回は稼働中source・worker・observerを変更しない。

## 原因と具体的な変更候補

circuits.gaussian_basisは8x8 orbital Uをlog/expmで256x256 full JW行列へ持ち上げ、8qUnitaryGateとして包む。
installed Qiskit1.3のDefaultUnitarySynthesisは>2qubitをgeneric qs_decompositionへ渡すため、Gaussian構造を利用しない数百万operationの回路を繰り返し作る。
GeometryPreparationのbasis再利用と隣接basis境界共有、4worker boundedqueueは既に実装されている。単なるprepare cache不足を今回の主因とはしない。
既存libraryのGivens cost helperはqiskit_natureに依存するが候補venvで使えず、現在H4 portはdense fallbackを固定している。install/upgradeは行わない。

候補moduleはadjacent complex Givens QRを直接実装し、n<=8で最大n(n-1)/2=28の2mode回転と8number phaseへ分ける。
full Gaussianを小さい4x4 fermionic two-qubit matrixとPhaseGateへ表すため、generic8qUnitaryGateのQSDを避ける。
これは同じΓ(U)を実装する異なる回路構成であり、compiled RZ/CX/depthは旧dense回路と等しくならない。旧cost系列と混ぜず、新resource series/semantics/profileへ固定する必要がある。
この構造化は[OpenFermionのGivens分解](https://quantumai.google/reference/python/openfermion/linalg/givens_decomposition_square)、[Qiskit NatureのBogoliubov回路](https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.circuit.library.BogoliubovTransform.html)にも対応する。実装は新規で、外部package/sourceをコピー・installしていない。
28はGaussian変換1回あたりの2mode回転上限であり、full wrapperやtranspile後のCX/RZ総数ではない。
例えばB0/L_D3/q1のbasis列は最大14回のGaussian適用となり、Gaussian部分の上限は392two-mode rotations+112number phases（別途controlled diagonalとwrapper gates）。最終metricやruntime speedは未実測。
大幅な回路縮小が期待されるが、必要184倍の実throughput・全map72h以内の完走・他template/cellsの性能はまだ保証しない。

## 限定実装・検証の証拠

SOURCE `0cfdca65ed4a19d05e0cd886c9bbfe2bc26bee57`、closure74。旧production72paths全byte不変、新inactive candidate/test2件のみ。
phaseはRZ単独で置き換えずnumber PhaseGateとして表し、vacuum=1を保持する。隣接modeのsingle-particle2x2を4x4へliftし、double occupancyにはdet(G)を入れてfermionic signを保持する。
QRはcomplex SU2で下三角をzero化、Dから逆回転をreverse elimination順に適用しΓ(U)を再構成する。exactzeroだけをskipし、approximate pruningを使わない。
nonfinite/nonunitary/shape/n>8はSTOP、unitarity/QR residual/reconstruction tolerance1e-12。input Uを変更しない。

12人工numerical tests PASS（fail/error/skip0）、wall0.538088秒、peakRSS78,319,616B。
1process/thread1、AS768MiB/RSS256MiB/wall30s/output1MiB、NumPy/SciPyだけ。実NPZ・分子/SCF/DF/input生成・Qiskit import/build/transpile・実worker/affinity/GPU0。
independent exterior-power minorsと、旧fallbackの式を独立JW演算で再実装したexpm(dΓ(antihermitian(logU)))を人工1/2/3/4/5/8modeで比較し、256x256 full Fock matrixまでabsolute1e-12で一致した。
complex phase/real negative determinant/swap/π branch/vacuum/double-occupancy sign、3mode controlled cosine/sine branch-relative phase、28rotations upperbound、入力不変を確認した。
これを実分子U・Qiskit builder・compiler output・9qubit実wrapper・全map成功の検証済みとは扱わない。build_basisのQiskit接続は未実行、人工compileは利用者指示で省略する。
初回tests-01でnumpy.testingのCPU検出subprocess10試行をguardが全拒否、実child0。assertだけをchild不要の有限abs-error比較に替えtests-02 PASS、初回ログ保持。

候補はproductionからimportされず、approved=false/runtime_authorization=false/wiring=false、allowed[]、未seal、commandnull。
H4 linear/STO3G/DF12/6凍結入力/T0.8/218templates/32trajectories・PF/RTE式・compiler optionsは採用時も保持する案。
worker4/32GiB・driver8GiB・observer256MiB/64MiB・CPU2/4/5/6・driver16・observer18・host猶予1/5/30/152.25・5秒監視・17GiB/74805/72h・carry0を維持する。
変更が必要なのはGaussian回路構成とresource cost系列の結合だけ。承認後はown-run09だけの正常停止・native残存確認、newSOURCE/seed/semantics/profile/input/carry/plan/auth/review/fresh未使用runへ結合して本計算で初期速度を評価する。
旧partial/random/checkpoint/scientificcacheを混ぜない。次stage未認可、共有環境/既存venv/他job/GPU変更なし、home内だけ。
[候補・検証・採用範囲入口](../../artifacts/resource_applicability/track_a_h4_gaussian_structure_proposal/2026-10-10/README.md)。

新2filesだけの独立数学差分reviewはPASS_INACTIVE_GAUSSIAN_MATH_PROPOSAL_DELTA_ONLY、blocking指摘0。
review SHA256 `8e336df69b976eb4e52375792a1e3514fdeefff52d3092ffb9f347b8d4a2dbb6`。Qiskitbuilder/compiledcounts/実速度は未検証、採用は新cost系列の明示承認後。既存environment/input/proof/campaignの全面review反復0。
