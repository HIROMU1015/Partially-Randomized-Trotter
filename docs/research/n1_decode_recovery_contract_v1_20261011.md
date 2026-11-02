# N1 入力decode復旧・一回再実行契約 v1

状態：`RECOVERY_SCOPE_PREPARED_REPAIR_AND_RERUN_PENDING`。
今回は修正境界・実形式preflight条件・再実行scopeを準備した。decoder修正sourceと科学結果はまだない。
N1の再実行は未認可で、元のTECHNICAL_STOPとfailureを保持する。
[機械契約](n1_decode_recovery_contract_v1_20261011.json)、[共通入口](n3_generator_n1_recovery_preparation_20261011.md)を参照。
前回GPT科学レビューは完了済み。同じ結果の再レビューや元batch全体のretryを行わない。

## 1. 技術停止の原因と修正範囲

元source `bbcb0176d43bfd1f7a10a87fa8f38e200d61ec46` の`prepare_factors`は
`ungauged_eta`を数値listとして読むが、実packでは`{"real": [...], "imag": [...]}`である。
科学selector・PF・shot/費用行へ到達していないので、性能失敗とは解釈しない。
既存source、run1/failure、N3/N2結果は上書きしない。

新versionの修正を次に限定する。

- 境界でvectorとmatrixをdecodeする。eta/approx_etaの実list、ungauged_etaのreal/imag objectを明示的に扱う。
- 全vector/matrixにshape、長さ、finite、数値型、spectrum虚部zeroの検査を追加する。
  bool、NaN/Infinity、ragged、unknown/oracle fieldは拒否する。非zero虚部を捨てない。
- vector/matrixのreal/imagを成分ごとに読み、元floatの`as_integer_ratio`を保持する。
  JSON再保存・丸め・decimalへの置換・原gのsymmetrizeをしない。
- 新しいN1専用runnerを作る。N3/N2関数を呼ばず、三context全件のpreflight後にだけN1 phaseへ進む。
- 旧N1 numerical関数は入力adapter以外を変えない。変更差分を機械監査し、sourceを新commitへ固定する。

frameのunitarityやraw gのHermiticityをschemaで勝手に修復しない。
元gに微小な非Hermitian成分があるという既定解釈と、そのgrowth allowanceを保つ。
入力変更、非zero spectrum虚部、旧条件との不一致が必要なら、decode修正を越えるのでSTOP。

## 2. 固定入力と全context検査

selectorへ渡すpackは
`artifacts/n13_redesign/2026-10-11/preparation/n1_candidate_input.json`、SHA-256
`70b2631b114983572cfc92527b2d569a4b48b03a6d88790b75dcf89f13614438`。
旧full resultは真値・旧費用情報を含むため新selectorへ渡さない。
新packを作り直さず、このbyte identityとtop/context/candidate whitelistを照合する。

三contextはplanted_local_gauge 9×9、dense_group_frame 9×9、separated_spectrum 4×4。
計44 factor候補・178 pair、gauged/ungauged計356構成を変えない。
candidate ID、順序、family、groups、係数、frame、元factorをすべて保持する。
全44個のungauged_etaが長さ4のreal/imag object、その他eta fieldsが長さ4の実listであることを確認する。
imag zero検査は許容toleranceなし。frame/g/one_body/input_factorsは4×4 complex objectとして検査する。
groupsはindices範囲・重複・partition、rank/weights/factor個数、ν/modes、q/time/model budgetも確認する。
実行前に一部contextだけ成功して他contextが後からdecode失敗する設計にしない。

今回は保存packの形状・finite・field・個数のread-only metadata確認だけを行う。
それは実装後のdecoder regression合格や科学的検証の代わりではない。

## 3. 修正後に必須となる回帰・preflight

後の修正作業では実packの全contextを新decoderに通す回帰testを追加する。
各real/imagのbinary64整数比が元成分と一致し、decode後にも44/178と候補順序が一致することを確認する。
object/list双方、長さ/shape不正、bool/NaN/非zero spectrum虚部、unknown/oracle field拒否を含む。
旧plain-listだけの合成fixtureがPASSしたことでは実pack対応を確認したとしない。

N1だけのnumerical synthetic検証を行い、frozen N1 referenceと同じ係数/誤差式・compiler条件を確認する。
N3/N2科学testsや旧付録replayは起動しない。新testsは最大24小fixture。
修正source固定→all-context実形式preflight保存→新source-bound planの順にする。
source固定後の実pack preflightはmetadataだけで、PF/cost/compile/真値を評価しない。
科学launchには別の利用者指示が必要。単なるschema修正ごとにGPT科学レビューは要求しない。

## 4. 科学条件を変えない

JW4 modes・ν2・sector dimension6、synthetic signed DF rank2、weights(.7,−.4)、
H1 diagonal(.08,−.04,.03,−.02)、scalar .11、geometry/化学basisなし、L_D該当なし、決定論S2。
T=.6、q=1/2/4/8、delta=.6/.3/.15/.075、model budget .02、εcomplex .04、αtotal .05、
X/Y二軸、normalization1、X0X1状態準備をfull5-qubit measured wrapper費用へ含める。
seed20261011、Qiskit1.3.0、opt1、rz/sx/x/cx、ordinary control、scalar controlled phase、
`qubits_initially_zero=False`、transpile num_processes1を保つ。

model/PF/phase上界、128-bit外向きinterval（既定の一回256-bit refinementだけ）、
Pauli cancellation、nested-commutator係数、safe整数Hoeffding shot規則、tie-breakは変更しない。
詳細は[元実行数学契約](n13_redesign_execution_20261011.md)と元sourceのN1関数を優先する。
未知の真の誤差をselectorへ使わず、ledgerを閉じてから真値診断する。
RZ/CXを別々にshot×full-wrapper費用で選び、exact fallbackを残す。

対照はnative exact、finite shifted cutoff、finite zero cutoff、同一近似ungauged、
whole-Pauli元H、whole-Pauli selected N1 approximation。family/q/精度/対照の追加・削除はしない。
後者whole-Pauliは選択後の同一近似対照であり、whole-Pauli全候補のglobal最適化ではない。
native operator checkは数値semantic検査であり、rigorous native rounding証明に置き換えない。
保証scopeは`ideal_gate_certificate_with_numerical_compile_validation`。

## 5. 将来の一回実行とSTOP

N1だけの新authorization、source/plan/input hash、新空run directoryを用意する。
旧混合runnerを再実行しない。N3/N2実行回数はzero。
CPU/memoryをlaunch前に再確認、最大2 spawn workers・各BLAS/OMP/Numba/Rayon thread1。
context内basis/cacheを保ち、scientific comparison・precision・compiler条件を変えず独立contextを並列にする。
一回限り、resume/retry/上書きなし。失敗checkpointとtracebackも保存する。

wrapper requestsは候補2848＋元Pauli24＋選択後Pauli48=2920。
既定hard cap3072 wrapper、44追加basis、total3116 compilesを維持する。
worker memory4 GiB・aggregate8 GiB、JSON/IR shard32 MiB、output512 MiB、log16 MiB、
native回路20,000 operations、Pauli dictionary256、rational4096 bitsを保つ。
wall-time/CPU-time capは置かない。tool待機timeoutは科学上限にしない。

全qが係数PF保証で不適格なら`UNRESOLVED_PF_CERTIFICATE`、exactへ戻る/費用差なしも正常な科学結果。
改善なしでもqや候補を増やさず保存してSTOP。技術失敗時も無断retryしない。
N1終了状態・一次結果・監査はN3の新生成法の状態と分ける。
前batchのN1未取得という履歴を成功時にも書き換えず、新runとして記録する。
中心テーマはnull、次科学stageはfalse。N2は完了扱い、N3の実行認可をN1から推論しない。
