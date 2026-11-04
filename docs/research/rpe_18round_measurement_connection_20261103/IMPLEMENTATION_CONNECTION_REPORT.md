# RPE 18 round：測定・実回路への接続調査と技術準備

2026-11-03 JST / LOCAL。主エージェント自身によるSOURCE調査・技術検証。
独立詳細科学レビュー、論文原稿、新しい科学的数値評価は実施していない。

**保存平均→既存decoderの接続と、shotごとのtrajectory要求→仮想±1結果の
interfaceを人工fixtureで確認した。実H4で保存平均と実回路の測定平均が対応すること、
最終18 roundのfresh-IID測定、実状態準備は今回未検証である。**
次の第一候補は[計画B：少数の固定trajectoryの条件付き測定平均照合](MINIMAL_SCIENTIFIC_VALIDATION_PLAN.md)。
大量samplingより先に、同一入力・同一requestに対する符号・scalar位相・回路意味論を切り分ける。

## 固定条件と証拠の版

対象はlinear H4、隣接1.00 Å、STO-3G、保存legacy DF rank12、8 system qubit、
4電子sector（dimension70）。元H物理基底状態を主入力とする。
Partial L_D=3 / Full L_D=12、δ=.02、m=0..17、q=1..131072のdyadic schedule、
最終coherent opt2のr/K/N/α/βを維持する。Fullはtail-free r0/K0。
費用側の条件はQiskit1.3.0 / opt2 / rz,sx,x,cx / seed17、状態準備費用除外である。
今回、価格の数値再評価はしていない。

| 既存資料 | 固定commitと読む範囲 |
|---|---|
| [V1](../pr_rpe_v1_e0_state_radius_20261103/REPORT.md) | e520e6e396f7e0def9e522ea1cff58eb10ddfa8f。physical/control、半径、共通E₀ |
| [V2](../pr_rpe_v2_branch_success_20261103/REPORT.md) | 78c49d0b5f1d71e63b5959e78813237923bbc648。REPORT、CSV、BRANCH/SUCCESS/VALIDATION JSON、runner/testを閲覧。再実行なし |
| [V3](../pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md) | cc60beb51930d4311d2ded84440682e583ccafc7。直接価格domainと外挿 |
| [R1](../pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md)、[次段判断](../pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/MINIMAL_NEXT_ACTION.md) | 8fde197c64da671f1d428f0d3e8fdb762dc21930。本branchの親 |
| 元SOURCE・保存入力 | 以下のSOURCEリンクはcarrier 4591b8d75a2a6a55c74a51225f0c61648023c8eeに固定 |

carrierは成果物の所在であり、元科学実行のclean SOURCEではない。
M06-F initialは26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c、fresh32は
7753eb71844e429facdb6f32c2def6e37379541fのdirty provenanceを持つ（V3の記録を引用）。
今回それらを再現・再監査したとはしない。

元checkoutのPROJECT_MAP、研究概要・現状、関連規範、VALIDATION_STATUS / manifest、
固定後続V1–R1を区別して参照した。LOCALのAGENTSと共通指示2026-10-25
§8.1・§8.4・§10を適用した。出力親のAGENTSは別版で、原文を変更していない。
本準備は科学結果を変更しないため、研究概要・科学manifestのstatusを書き換えない。

## A1. fresh-IIDとevent-meanの対応

| 処理 | SOURCEと接続 | 直接確認できる範囲 |
|---|---|---|
| 1 event | [rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rte.py)：finite_rte_distribution、_make_event、iter_sample_rte_events | 偶数orderを重み/B_Kで選び、各componentを係数絶対値の正規化分布から選ぶ。PCG64。event.phase=(-1)^(order/2)、rotation_angle=atan(τ/(order+1)) |
| 1 trajectory | [repeated request](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_partial_s2_repeated.py)：make_df_partial_s2_repeated_request | q個のoccurrence、各r event。master→step seedを使う。q/r/K/δと分布の対応を保つ必要がある |
| controlled回路 | 同moduleのQiskitDFPartialS2RepeatedCircuitBuilder、[step builder](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_partial_s2.py) | 与えられたrequestをbuildする。backend shot loopではない |
| cosine/sine | [wrapper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_hadamard_interrogation.py) | 渡された同じcontrolled evolutionを包むだけ。再抽選せず、fresh_iid verified=Falseを返す |
| shotごと再抽選 | [failure validation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_hadamard_failure_validation.py)：_explicit_fresh_iid_validation、_seed_for_shot、_trajectory_signal | (q,axis,shot)ごとにseed、q*r個のevent、sector state-action、条件付きBernoulliを生成。実backend測定ではない |
| 価格のpaired標本 | [full scope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_scope.py)：_compile_paired_requests | 一つのrequestから両axis・両basis policyの価格を取得。量子shotの独立性の証拠にはならない |
| M06-F | [full opt2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2.py)：compute_full_opt2_randomized_cell | trajectory stream→paired compile。N quantum shotsを実行しない |
| fresh置換と再会計 | [extension analysis](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2_extension_analysis.py) | 価格較正のfresh置換であり、測定shotのfresh-IID確認とは別。全解析はH準備と再探索を含むため起動しない |

理論上の対応は次のとおり。τ=λ_R δ/r、D_f/D_rを決定論halfとし、
T_Kをdegree K+1までのpaired Taylor numeratorとする。
独立eventの積について期待値を取れる場合、

\[
\mathbb E[V_e]=T_K(-i\tau h_R)/B_K,\quad
\mathbb E[V_{\cal T}]=[D_r\{T_K/B_K\}^rD_f]^q
=B_K^{-rq}A_{\rm corr}.
\]

物理入力ψに対するz_att=〈ψ|E[V_T]|ψ〉は、
[finite signal SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/finite_rte_signal_validation.py)
のcorrected_step→matrix_power→attenuation×corrected_roundという保存定義に対応する。
これはSOURCE定義からの代数的対応であり、新しいH4数値照合ではない。
同じtail・component符号・順序・scalar・基底規約、独立なoccurrence/event、同じ入力ψ、
正しいcontrolled/Hadamard測定が必要である。

Y∈{±1}、E[Y|T,axis]=Re/Im〈ψ|V_T|ψ〉なら、
P(Y=+1)=(1+Re/Im z_att)/2。shotごとの独立trajectoryと独立測定乱数により
各axisのIID Bernoulli marginalへ接続する。cos/sinの二項productには両実験の独立性も要る。
round間独立はunion boundそのものには不要である。

一つのrandom trajectoryをN shotへ固定すると、条件付きには同じμ_Tの二項分布でも、
無条件では共通latent Tを持つ混合分布となる。一般に保存平均μの単一二項分布ではない。
cos/sinで同じTを共有しても一般に独立productにはならない。
Fullはdeterministic PFなので同じ回路の反復利用が可能だが、測定・状態resetの独立性は別条件。
seed非重複だけを数学的独立性の証明としない。

旧RPE4はH4/1Å/STO-3G/DF12/L_D3、δ=.1/r4/K2、q=1,2,4,8でfresh-IID
trajectoryの局所検査を保存している。V2 REPORTの引用であり、今回再実行していない。
最終δ=.02・選択r・18roundで同じ測定を実行済みとは読まない。

## A2. controlled-U・測定・位相

ordinary controlledは|0〉〈0|⊗I+|1〉〈1|⊗V。
回路はancilla H→controlled V→H→ZでRe z、
H→controlled V→S†→H→ZでIm zを読む。bit0→+1、bit1→−1。
したがって推定complex値はmean_cos+i mean_sinで、U(t)=exp(-itH)なら
base phase φ=-Eδ、energy=-φ/δである。

constantおよび抽出identityはstepの_append_step_phasesで−δ×coefficient。
uncontrolledではglobal phase、controlledではancilla PhaseGateへ載せる。
RTE eventの符号位相も[df_rte_qiskit.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_rte_qiskit.py)
でrelative ancilla phaseとして保持する。追加のenergy shiftやidentity二重補正はしない。
uncontrolledのglobal phaseを捨てる操作はHadamard平均を変え得る。

基底の逆変換は実際のgate.inverse()。固定trajectoryのadjointは**全積を逆順にし、
各gateとscalar位相を共役反転したもの**である。eventが非可換なら単にrotation角だけを
反転して同順に適用したものとは限らない。D_rもD_fの同じ負時間adjointではなく、
同符号half-timeの逆sweepである。平均A_attは一般に非unitaryなのでA_att†とA_att⁻¹も混同しない。

Partialの測定平均はphysical_df_ground_state / attenuated_event_mean_signal。
B_K^(rq)で補正したcomplex値を±1測定平均へ入れない。正normalizationは位相を変えないが、
可視度と分散・shot成功条件を変える。Fullはtail-freeのphysical PF signalで、RTE補正を加えない。

通常wrapperの公開build domainはq=1,2,4。
[QiskitRPEHadamardBenchmarkCircuitBuilder](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py)
は明示maximumのdyadic domainを検査し同じ_build_validatedを再利用する。
qを許容できるAPIと、長qのcorrectness・価格・実行可能性の検証済みdomainは別である。

既存testsはsmall synthetic DFで全eventのdense対応、diag(I,V)、符号・identity、
wrapper Re/Imと負sine、反復・basis共有を検査している。tests/test_df_rte_qiskit_builder.py、
test_df_partial_s2.py、test_df_partial_s2_repeated.py、test_rpe_hadamard_interrogation.pyを静的参照した。
別研究のnative_df_gate_model / test_native_df_qiskit_connectionも存在するが、別constructionの
correctnessをlegacy H4最終18roundへ転用しない。既存回路testは今回実行していない。

## A3. decoderの接続

既存reconstruct_rpe_phaseはarg z_mからq個のbranch候補を列挙し、前round推定位相に
円周上で最も近い候補を選ぶ。初期q1はbranch0、phase∈[-π,π)。
対応energy windowは(-π/δ,π/δ]であり、物理的なalias除外は外部条件である。
decoderにはE₀を渡さない。V2のoracleは復元後の評価専用である。

元scalar decoderはtieでargminの最初の候補を返す。新adapterは選択を変更せず、
runner-up距離と1e-12以内で同距離ならAMBIGUOUS_BRANCH、energy=nullとする。
zero signalはArg(0)へ進めずZERO_SIGNAL。初期／最終が±π境界近傍ならALIAS_BOUNDARY。
これらは技術上の不定・境界表示で、別のbranch規則や科学的保証を導入しない。
aliasの内部にある測定推定でも真のenergyのalias所属を確認したとはしない。

V2の両候補CONDITIONAL_SUCCESSは保存平均と固定Nのモデルに対する既存判定。
全round位相十分事象のunion boundは、全decoderの厳密joint失敗確率ではない。
今回の人工測定結果にV2の成功率を流用せず、新しい成功率も評価しない。

## B. 入力状態・半径

元accountingはPF effective eigenstateと単位unattenuated radiusを仮定する。
V1/V2の主評価は元H物理ground stateであり、高overlapでも両者を同一と証明していない。
Full δ=.02のPF controlは未発見、Full診断自身のstate fingerprintも未記録を保持する。
Fullの半径はrawでは18点とも1未満、既存matrix atol=1e-10を超える7点
（m6,7,8,9,10,11,14）のV1不一致を維持する。新しい物理的破綻と断定しない。

V2は保存physical μから固定Nの分布を評価したので、単位半径仮定を新たに証明せずとも
その条件付き確率評価を持てる。一方、任意の実装入力へのN保証には、実際のmarginalと
正当な半径／位相条件の対応が必要。near-one半径を厳密下界へ昇格しない。

保存Hamiltonian snapshotのwriterはHのarrayを保存し、state vectorを保存しない。
保存fingerprintはvectorそのものではない。今回byte取得・NPZ decode・状態生成はしていない。
将来、同じHから同じsector ground stateを再構成するなら、state solveも明示実行範囲に含める。
原Full診断の欠測fingerprintを新stateで事後補完しない。

実state preparationへの接続には、実際のsystem state/density、8qubitへのsector埋込みと
endian、reset、準備誤差・visibility、ancillaとの初期積状態、shotごとの安定性が必要。
理想statevector注入は準備回路の実装・価格・backend validationではない。

## C. 新adapterと技術検査

[measurement_adapter.py](measurement_adapter.py)はこのtaskに閉じた薄い接続である。
load_fixed_roundsはV2/元opt2から36roundの設定を転記するだけで、保存signalのbranch再解析や
確率・費用評価を行わない。RoundSettingsはN/α/β/ε、q/r/K、state/signal labelをimmutableに保持する。
E₀と保存signalを設定objectへ入れない。

collect_virtual_roundは明示callbackだけを受け、Partialは(axis,shot)ごとにfactoryを呼ぶ。
reuse_within_axisは別policyとして表示し、FullではRTE factoryを呼ばない。
callbackはtrajectory request、controlled回路、測定に将来接続する場所であり、
今回H4 backendは付属しない。factory自身のcacheやRNG独立性まで保証するinterfaceではない。
decode_measurementsは既存3 FunctionDefをAST変更なしで固定carrierから読み、
科学module全体とtrotterlib.__init__の広い依存importを避ける。既存V2と同じ限定再利用方式である。
重複decoder、RTE sampler、state-preparation、compilerは新設していない。

[test_measurement_adapter.py](test_measurement_adapter.py)は15件PASS。
人工phaseの18round復元、sine/energy符号、fresh/reuse factory回数、Full分離、
範囲外μ／bit値拒否、低半径、zero、tie、alias境界、immutable設定と36roundの転記を検査した。
有限sampleのcomplex推定値は各座標が±1以内でも半径>1になり得るためclampしない。
真のmarginalに対するunit disk検査とは分ける。

実行方法（既存NumPyがあるPythonで、このdirectoryから）：

```bash
python -m unittest -v test_measurement_adapter.py
```

Python3.11.0rc1 / NumPy1.26.4の既存環境をread-only使用。人工H4・RNG生成はなく、
科学的なstate-action、sampling、build/compile、V1–R1の再実行は0。
記録は[VALIDATION.json](VALIDATION.json)。本PASSはH4成功率・compiled correctnessの保証ではない。

## 保全・費用研究・STOP

V1/V2/V3/R1、元SOURCE/input/raw/QPY、grant/ledger/marker、旧STOPは変更しない。
元dirty/untracked tree、SERVER2のH8 worktree、他研究をcleanup/resetしない。
本branchはR1を親とする独立worktreeであり、公開対象は今回の文書・adapter・人工tests・索引のみ。

V2両候補CONDITIONAL_SUCCESS、V3長q価格外挿、R1 Partial約4.858%点選好と総合UNDETERMINED、
順位決定目的でR2/R3非推奨を維持。共通準備単価もNが異なるので一般に相殺しない。
新科学SOURCE/backendは未実装・未実行。次の承認対象と不足は[判断](NEXT_STAGE_DECISION.md)に明記する。
専用branchのcommit/push・remote SHA/成果物byte照合後、mandatory STOP。
