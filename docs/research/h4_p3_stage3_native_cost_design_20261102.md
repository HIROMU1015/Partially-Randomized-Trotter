# P3 Stage 3：H4 matched native回路・価格比較の非実行設計

2026-11-02 JST。**設計案のみ。Stage 3の科学実行・回路生成・sampling・compileは未承認。**
ユーザーはStage 3準備を優先し、Stage 2 finite biasの追加高精度化を現時点の必須条件にしないと判断した。
その範囲でSOURCEを静的に調査した。[技術preflight・再利用計画](h4_p3_stage3_preflight_readiness_20261102.md)と一組で読む。
本案は新しい主要claimの採択や詳細科学レビューではない。旧STOPを解除せず、今回はMarkdownだけを作成する。

## 1. 根拠・目的・固定条件

Stage 2結果commitは`8bcf5851f99f980dc08c5885ace54874b554c815`、科学SOURCEは
`4bfbbaad1f8178077489bf06b085377256d6fb80`。
[結果報告][REPORT2]、[辞書・固有basis][DICT2]、[信号/normalization][RESULT2]、[STOP][STOP2]を継承する。
独立operator preflight57項目、nominal/共通対照/raw-B照合5項目はengineering PASS。
保存BはPauli `39.125909965750068`、DF `34.527999187486905`。
同じ物理分割でDFのBが小さいという固定条件の観測であり、以前のPauli純RTE対DF B3の比較と同一ではない。
Stage 2 finite biasの両方式・両軸はUNRESOLVEDを維持する。

目的は**同じGaussian保持programを共有したときの、異なる残差辞書とそのnative実装のcompiled価格**を調べることである。
同精度総資源、母平均価格、全law保証を初期pilotの目的に含めない。

| 項目 | Stage 3初期案 |
|---|---|
| 系・入力 | linear H4、隣接1.00 Å、STO-3G、中性singlet、system8/full256、保存DF tol-only1e-8/rank10 |
| 分割・方式 | 案A `T_eff / V_square`、M_B=Pauli残差、M_C=DF rotated Z/ZZ残差 |
| 時間 | `T=Fraction.from_float(0.8)`、S2、q8、δ=T/8、r4、h=δ/4、K4（event次数0/2/4） |
| 共通保持 | 同じDF-native Gaussian T_eff half-block、同じbasis・命令列・half時間・制御方式 |
| 残差 | Stage 2保存Pauli184成分、DF360成分・10 basis。identityなし、係数cutoff0 |
| 測定境界 | 8 system＋ancilla8、同じcosine／+Im sine full wrapper、P=0 |
| 主指標 | 実compile後のRZ、CX、total depth。size、RZ/CX depth、全histogramも保存 |
| 初期標本案 | 各方式4 trajectory、合計8本。各本32 event位置、全体256 event位置 |
| 後続 | 母平均費用・追加標本・実効law平均・Stage 4は別scopeとして判断 |

共通raw state/ν/referenceは来歴として保全する。価格pilotではstate準備も新しいphysical信号評価も行わない。
`P=0`は状態準備費用を除くという評価境界であり、保存φを|0…0⟩へ置換して科学信号を比較する意味ではない。

## 2. 共通Gaussian保持blockとscalar

保存h_effのHermitian対角化を1回だけ行い、`h_eff=U_T diag(ε) U_T†`とする案。
`Γ(U_T)`を既存JW Gaussian native loweringで構成する。system unitaryは

\[
 E_T=\exp(-i\delta T_*/2)
    =\Gamma(U_T)\exp[-i(\delta/2)\sum_i\epsilon_i n_i]\Gamma(U_T)^\dagger.
\]

命令の時間順は`Γ(U_T)† → D_T → Γ(U_T)`。
逆変換には**実際のgate列の逆順＋各gate.inverse()**を使う。
`U.conj()`、符号変更だけ、転置の取り違えで代用しない。
同じ作成済みhalf-blockを両方式へcopy/composeし、方式ごとの再対角化・再合成をしない。

native basisは`df_trotter.ops.U_to_qiskit_ops_jw_givens`、内部の`BogoliubovTransform(U.T)`経路を第一案とする。
これは既存DF-native Gaussian経路の利用であり、dense UnitaryGateの汎用合成価格を採用する案ではない。
汎用`U_to_qiskit_ops_jw`にはdense fallbackがあるため、初期価格案ではfallbackを無効として扱う。
qiskit-natureが使えない場合はSTOPし、実装familyを黙って変更しない。

対角一体作用では`n_i=(I−Z_i)/2`なので、既存`one_body_diagonal_primitives`は
`RZ_i(−τε_i)`とphase `−τΣ_i ε_i/2`を返す。
τ=δ/2。このphaseはT内部identityの一部であり削除しない。
controlled保持は無制御の前後basisとcontrolled対角primitiveを組み合わせ、ancilla=0側でbasisが打ち消され、
ancilla=1側でE_Tになる`diag(I,E_T)`を要求する。
basis自身の全体phaseが表現上省略されても、同じbasisとactual adjointで打ち消されることを検査する。

外側scalarはStage 2と同じ`c_*=c_N+v̄`、`c_N=2.29310124732`、`v̄=4.679666433618258`。
exact binary64 operandの有理数和にTを掛けた角度を基準に、一度だけ`PhaseGate(−c_*T)`をancillaへ置く。
scalarは保持側Rが所有する。Pauli full-H constant、各辞書のidentity、T traceを別途追加しない。
Stage 2のidentity defect（Pauli約5.33e-15、DF約−8.88e-16）は旧結果として保持し、別scalarで補正し直さない。

**最初のH4 correctnessはGaussian保持作用・scalar・actual adjointの照合。**
Stage 2の共通T配列を読み、同じcomplex128で必要なE_Tを1回構成して参照cacheする計画である。
Stage 2 artifactにはE_T行列そのものは保存されていないため、保存済みE_Tを再利用できると装わない。
この再指数計算もStage 3科学実行許可に明示して含め、今回起動しない。full-H/referenceの再生成は不要。

## 3. 残差native接続とevent意味

Stage 2の成分ID・順序・係数、DF U/ηをそのまま読む。DFの再対角化・新factorizationをしない。
各係数の符号をinvolutionへ含め、samplingは係数絶対値に比例する既存規約を使う。

| 箇所 | M_B：Pauli | M_C：DF |
|---|---|---|
| 辞書 | 保存global Pauli label順 | 保存fragment/basis/support順、basisを越えた統合なし |
| basis | X/YのPauli basis-change＋parity回路 | 保存U_lをGivens経路でlowerし`DFBasisRegistry`へ登録 |
| product | signed Pauli積を時間順に実装、負符号はrelative phase | rotated Z/ZZのproduct、basisは無制御、central Z/ZZは制御 |
| rotation | `exp(−i θ s_j P_j)`、D2のcentral RZ制御 | `exp(−i θ s_j Z/ZZ)`、central RZ/RZZだけ制御 |
| event phase | `(-1)^(k/2)`を一度 | 同左、負product符号を別途一度 |

SOURCEの`_make_event`では抽出index列の最初がrotation、残りがproductsだが、**作用順はproducts→rotation**。
`RTEEvent.selected_component_ids/application_sequence`の保存順に従い、抽出順で回路へ並べない。
`rotation_angle=atan(λ_R h/(k+1))`、DFの`unsigned_rotation_angle`という既存propertyは実際には
`coefficient_sign * rotation_angle`を返す。この符号をさらに掛けて二重反転しない。

Pauli残差だけを`append_pauli_event`で作り、`h4_d2_pauli_control.central_control_body`へ渡す。
Gaussian保持を混ぜたbody全体をD2へ渡さない。DFは`QiskitDFRTEEventCircuitBuilder`を再利用する。
初期案は両方式ともbasisを無制御、central作用だけ制御する規約を採る。
一般的なwhole-body `.control()`との価格比較を追加しない。

DF basis reuseは既存`raw_adjacent_equal_basis`、`registered_full_basis`に限定する。
同じbasis ID/hashが連続する残差application間だけでactual adjointの対を省く。
各outerのr=4 event sequence内で使用し、保持halfとの境界を越えるreuseやsupport-restricted最適化は行わない。
残差sequenceは各outerの終わりにsystemの元basisへ戻す。
input programを両armで揃えるためのbarrierは挿入せず、compilerによる境界間の簡約はFull価格の一部とする。

## 4. Trajectoryの事前選定案

初期案は**各arm4本を取得前に固定し、採用・再抽出・標本追加を価格観測後に行わない**。

| trajectory index | Pauli seed | DF seed |
|---|---:|---:|
| 0 | 310201 | 310301 |
| 1 | 310202 | 310302 |
| 2 | 310203 | 310303 |
| 3 | 310204 | 310304 |

これは新規Stage 3専用seed案。今回、RNGを呼び出したりeventを選んだりしていない。
各seedから既存`sample_rte_events`を1回、`sample_count=32`で呼ぶ。
PCG64、方式別の保存成分順・係数確率、保存λ_Rを使ったfinite paired分布を固定し、
flat streamを`[outer0 inner0…3, …, outer7 inner0…3]`へ割り当てる。
`rte_steps=4/evolution_time=δ/K4`のlawを32位置で使い、T全体をもう一度rで割らない。
DF repeated runnerの別のstep-seed階層を片方だけに適用しない。

同じindexの両armは独立した別標本。cross-armのpaired差・paired SE・共分散低減を主張しない。
一方、同じarm/trajectoryのcosineとsine、およびFull/残差のみには同じ保存event列を再利用する。
出現したorderの偏りやorder4未出現を理由に再抽出せず、coverageを結果へ記す。
order0/2/4、負係数、Y、basis切替のcorrectnessは別の小規模人工fixtureで確保する。
forced人工eventをH4費用cohortへ混ぜて「無条件平均」にしない。

binary64確率・角度・PRNG/choiceの実際のレシピと全eventを将来保存する。
Stage 2はideal nominal平均であり、Stage 3の具体的sampling/angle bindingとは同一保証ではない。
操作上必要なlaw構成・samplingと、53-bit CDF/finite-bit実効**平均の新評価**を区別する。
後者、`Δ_grid`、全law包含保証をこの初期価格案に追加しない。Bの新評価やStage 4への転用も行わない。

## 5. Full・部品と評価境界

各trajectoryのcontrolled system bodyは、時間順に各outerで

```text
common T_half → residual event[0] → … → residual event[3] → common T_half
```

を8回並べ、共通scalarを全体で一度置く。
その同じbodyへ`hadamard_wrapper(..., already_controlled=True)`を適用する。
ancillaはqubit8、cosineはH/body/H/measure、sineはH/body/Sdg/H/measureで+Imを測る。
measure込みでcompileし、P=0。新しいstate preparation回路は作らない。

| program | 操作・phase所有 | input wrapper数＝予定compile数 |
|---|---|---:|
| F：Full | T-half 16位置＋保存32 event＋共通scalar | 2arm×4trajectory×2軸＝16 |
| E：残差のみ | 同じevent列とevent phase。Tと共通scalarを削除 | 16 |
| R：共通保持のみ | FのT-half列と共通scalar。残差を削除、half列は手動結合しない | 2（両arm共通） |
| Empty | identity controlled body、scalarもevent phaseもなし | 2（共通） |
| 合計 | Full16が主価格、E/R/Empty20が診断 | **36** |

この件数は提案であり、今回の実行件数ではない。
F/Eは同じeventから作る削除介入、Rは単一の共有program。
Eでは各outerのresidual sequence境界を保存し、保持削除後の手動basis再最適化は行わない。
新しいcost frameworkは不要。旧P2のF/R/E/Emptyという操作的定義とmetricsを再利用し、
旧H6のK2制限・Pauli-only R builder・科学契約全体は使わない。

RZ/CX/sizeについて`Δ=C_F−C_R−C_E+C_Empty`を同じ軸で報告可能とする。
これは固定program・compilerにおける非加法性であり、一意な因果配賦ではない。
depthは各programで報告し、部品depthの加算を回路構造の恒等式としない。
初期cohortは各行と観測min/maxまでを主表示とし、新しい母平均・tail・方式順位・N/Gを算出しない。
追加平均費用評価の要否・標本数・探索と独立確認は後続判断へ残す。

## 6. Compilerとprovenance

第一案は既存比較と同じQiskit1.3.0、basis `rz,sx,x,cx`、opt1、transpiler seed17、
backend/coupling_map/layout/routingの追加指定なし、数値パラメータbinding済み。
qiskit-nature0.7.2/Givens経路を記録する。現地METADATAの静的読取では両versionが存在するが、
import・回路loweringの成功は今回確認していない。
project専用runtimeを使い、共有venv/package/cacheを変更しない。

各price targetはinput QPY・semantic命令/phase fingerprint・compiler設定、compiled QPY、
RZ/CX/depth/size/histogramを対応づける。
cosine/sineのtargetを分け、program fingerprintが同一と確認できるR/Emptyだけ共有する。
旧費用JSON・別prefix・別basisのcacheを新価格として流用しない。
同じtargetの重複compileは避けるが、成功していないcompileをcache hitや価格0として扱わない。

## 7. Correctness・STOP・主張範囲

[技術preflight](h4_p3_stage3_preflight_readiness_20261102.md)を先に完了する。
共通保持の作用/actual adjoint/scalar、保存辞書とnative basis、選定event順・phase、
controlled bodyのancilla0/1、wrapperの+Imが非PASSなら、価格compileへ進まない。
global phaseを事後alignmentして差を消さない。full256/controlled512で評価し、無断sector射影をしない。

compile済みであることは科学的作用保証ではない。
初期案はartificial compiler回帰と、H4の固定input native programの数値correctnessを要求する。
全36 compiled targetのfull operator残差ρ、全lawρ、厳密なuniform compiler保証を取得したと主張しない。
それらが必要な科学判断は別の明示scopeへ分離する。

将来の実行で入力/辞書/basis不一致、共通保持programの方式差、phase欠落、
反転/制御/bit順の不整合、非有限値、native lowering失敗、価格取得失敗があれば原因と途中出力を保持してSTOP。
自動fallback・別seed・追加trajectory・別compiler・高精度solver・科学retryは行わない。
一時的なCPU競合や無関係worktreeの通常更新だけで自分の正常workerを中断しない。

完了後も固定program/cohortに条件付けた価格診断で止める。
Stage 2 finite biasをゼロ認定せず、Stage 4の精度余裕が必要になる時点でその分解能を再検討する。
現在の`STOP_STAGE2_UNRESOLVED_COMPONENTS`を保全し、今回の準備完了後も科学実行へ自動移行しない。

## 8. 承認対象・残る判断

本案の推奨は、**各arm4本／8 trajectory／36 wrapper compileと、必要なnative correctness preflightを一括で許可する**範囲。
これはCPU/RAMの人工上限ではなく、取得する科学対象の指定である。
ユーザーはこの科学範囲・seed・compiler・preflight精度政策を確認後、希望する実行範囲を明示許可する。
4本は十分な母平均推定標本数を意味しない。Full16だけに縮小する場合は残差単独価格と非加法性の診断を失う。
その変更は実行前に範囲を明記し、結果を見て対象を選び直さない。

承認後にCodexが最小adapter/runner・人工testsを実装し、実際の科学SOURCE commitとruntimeを固定する。
新たな承認証明書・grant発行基盤は不要。許可された固定scope内の通常実装/debugはまとめて行う。
今回の成果物は文書だけで、SOURCEの未来SHA、event選定結果、native一致値、費用値は未作成。
**今回mandatory STOP。次の担当はユーザー：Stage 3の実行範囲を判断。Codexは明示許可後だけ実装・科学実行する。**

[REPORT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/REPORT.md
[DICT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/dictionaries.json
[RESULT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/results.json
[STOP2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/STOP.json
