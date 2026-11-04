# 研究C：現PR候補のIID-grid reference bridge・Stage A実行結果

2026-11-03。固定したStage Aだけをfresh `stageA-run01`で一度実行し、**38/38 calls COMPLETE、correctness FAIL/retry/resume 0**で完了した。
結果は`IID_GRID_REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE`。全13 roundの数値対応と、IID-gridモデル内の正確なcategorical TVを得た。
Whole-kernel P0一様誤差、reference数値不確かさ、実PCG64 joint stream、厳密CA、actual CAは未資格のままである。
Stage B/C・native compile・sampling・RPE/shots評価は開始していない。

## 固定条件・出典

対象は直鎖H4・隣接1.00 Å/STO-3G、Nα=Nβ=2、sector36、number CDF rank8、元frame順LD2、partial ordered symmetric S2、
s1/r32/K0、δ=h=.1、m0..12。CA=.00159 Ha、DF reserve=CA/8、own-DF ground oracle、状態準備費用除外。
保存native仕様はP0（Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/MPFR160 declared phase/既存Gaussian・relative fusion）、
controlled full512、head/X/Y shell込みfenced serial。今回はsector referenceだけを数値評価し、full512 native回路は作っていない。

- Endpoint SHA256：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
- 元PR科学SOURCE：`4df8eca29c59ffd55e2dd898f05da16bb246ddbe`。保存下流結果は`beb8de5cd8b71f80bb69556f92299435ff431c3e`。
- 今回の実行SOURCE commit：`ddd8333488e07a3b3af9095b5cb193ef9f90b3c3`、実装固定commit：`55fc3f3015e95bf181d4dcf155d74c90a19abc3e`。
- [固定契約](research_c_iid_grid_all_law_contract_unapproved_20261103.json)のSHA256：`f9328123c2655ec36ba590556231b9f19ab17bee68f065c0cb422d2d09f2bee2`。
- [準備SOURCE](../../artifacts/research_c_iid_grid_all_law/2026-11-03/technical/SOURCE.json)のSHA256：`95897050fb5a36ede0f60126da9fee3d4458fcb83d9c7afcb08b4f3a0cccd38a`。
- [準備報告・数式・誤差会計・Stage A〜D](research_c_iid_grid_all_law_readiness_20261103.md)。旧未承認proposalは原本のまま保存し、今回の実行に合わせて書き換えていない。

実行時のSOURCE154 file・保存input19 file・package/identity/compiler/target・旧STOP・fresh条件はpreflightで一致した。
新grant・承認証明・承認原文保存を要件にしていない。ユーザーの今回の実行許可は固定Stage Aに限定して扱った。

## 実行範囲と資源

既存Stage B/C2/Bridgeのfactor・signed atom・K0平均・CAR照合、既存categorical格子計数、Calls/resource_planを再利用した。
追加adapterは現在のh/2=.05、r32、p/q両平均、保存corrected kernel、13 roundへの接続だけで、実行SOURCEに変更はない。

| 登録単位 | COMPLETE |
|---|---:|
| Metadata/identity/input/sector/arrays/model replay | 6 |
| Factor binding | 9 |
| Law/atom context/choice-grid/216 atom audit | 4 |
| Ideal p・grid q event mean | 2 |
| Head/kernel/共有round powers | 3 |
| Round bridge | 13 |
| Summary | 1 |
| **合計** | **38** |

[STOP](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/STOP.json)と
[call journal](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/science_calls.jsonl)は38 attempt/38 COMPLETE、例外0を記録する。
Stage A wall timeは2.5686秒。CPU affinity32、実行内観測available memory約55.7 GB、独立13 roundに13 workers/worker当たりBLAS1を使った。
4本のbinary-power chainとsector演算子を共有した。任意のCPU/RAM/wall capや共有環境の設定変更はない。比較実測のないspeedupは主張しない。
Import時のMatplotlibは書込み不能な既定config先を報告し、`/tmp`の一時cacheを使用した。科学callは正常完了した。

## 正確なcategorical距離と条件付き伝播

数学的モデルは各event・coherent occurrence・measurement shotで独立な一様53-bit整数draw。
NumPy1.26.4の既存binary64 normalize/cumsum/normalize、`searchsorted_right`を維持した。
保存PCG64は再現用実装であり、このモデルとのjoint-stream一致・実shot独立性を証明したものではない。

[Categorical table](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/categorical_table.json)は216カテゴリの整数格子計数を保存する。
正規化前後のp、CDF、2^53格子のqを区別し、乱数drawや全格子列挙は行っていない。正のideal atomの消失は0件。

`TV(p,q) = 74656384577585465687339807 / 21779998563114881419643461642781944971264`。

外向き表示値は**3.4277497476064313e-15**。これは現在の保存atom係数から定まるpと選択写像のqとの正確な距離で、
元integralからの係数生成誤差・CAR bridge・native・実PCG64の誤差を含まない。

同じexact unitary trajectory map、ideal shellで`|z|≤1`、独立抽出の下で、既存coupling式`min(2, 2 N_m TV)`を使う。
`N_m=32×2^m`。独立kernel再抽出の平均のべきと、同じwordを再利用する`E[U^n]`を同一視していない。

| Round | Kernel occurrences | Events | 条件付きraw complex signal差上界（外向き値） |
|---|---:|---:|---:|
| m0 | 1 | 32 | 2.193759838468116e-13 |
| m12 | 4096 | 131072 | 8.985640298365403e-10 |

これはraw信号単位で、attenuationを重ねて乗除しない。PF/corrected-RTEやDFをcategorical差へ加えていない。
最終compiled P0への保証・actual implemented-law errorとしては使っていない。

## Reference bridgeの数値観測

9 factorの保存CAR対応、全216 signed atom、λ/τ/identity/K0、event unitaryと重み付き平均、head/scalar/normalizationの検査を通過した。
Exact atom massのbinary64表示は.5000442435909978、event angleは.0015626369893203945、m0 attenuationは.9999609313975324。
元g補正κと核scalarを含む保存G0、抽出tail identity、frame/spin/bit順を保持し、phase fitを使っていない。
保存PF/corrected-RTE kernelやground stateを再計算せず、同candidateの保存corrected kernelとown-groundを読んだ。

[Kernel照合](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/kernel.json)のideal-atom対saved raw CARは
最大要素差3.130879420609716e-14、Frobenius差1.7563870500806382e-13（登録要素allowance1e-10）。
保存raw complex信号の再生差は全13 roundで最大6.021104987719889e-15（登録allowance2e-10）。

以下は[m12出力](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/m12.json)の**数値観測**である。

| 差の対象 | Operator 2-norm観測 | Frobenius数値pilot | Own-ground raw complex差観測 |
|---|---:|---:|---:|
| Ideal atom − saved CAR | 1.0867657355169296e-10 | 6.169816075972338e-10 | 1.086765502209345e-10 |
| Grid − ideal atom | 1.1542424662292711e-10 | 6.746867779568235e-10 | 1.1258488197740313e-10 |
| Grid − saved CAR | 1.58961020298557e-11 | 6.050586939661043e-11 | 3.909136558063065e-12 |
| 二つのnormalization power経路 | 3.987450533635986e-13 | 1.3453405226579224e-12 | 2.3363133811002854e-13 |

全m0..12の値・complex X/Y parts・条件付きcategorical上界は[result.json](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/result.json)に保存した。
行列の生成・減算・べき・normを厳密包囲した値ではなく、Frobenius field名の`upper_pilot`もrepresented matrixに対する数値pilotである。
小さいgrid対CAR残差をモデル改善や相殺保証と解釈しない。
`reference_numerical_uncertainty`・`rigorous_bridge_bound`はnullを保持し、観測距離を厳密上界としてaccuracyへ自動注入していない。
PF/RTE誤差は保存値の引用だけで、新評価・二重加算はない。DF評価・加算も今回は行っていない。

## 証拠の資格・保全・次の段階

Stage Aはsector36の**ローカル数値bridge pilot＋明示したIIDモデル内のcategorical評価**であり、CIや外部独立再現ではない。
実行前の182件人工/回帰PASSは準備時の記録で、実H4数値のfixture再計算として再実行していない。
[実行後の静的確認](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA_static_verification.json)は
SOURCE/input/契約・38-call内訳・13round・null資格・26出力・旧記録保全を検査し、新しい科学評価を行わない。
全体manifest checkerは旧`relevant_commits[37]`のキー不整合でFAILする既知状態を保持しており、global validation PASSとはしない。

- Stage A：完了。fresh run01を消費済み。retry/resume0、再起動不可。
- Stage B：`P0_UNIFORM_NOT_ESTABLISHED`。全合法wordの最終P0・control/actual inverse/shell/full512・丸めを覆う方法が未成立、calls未確定。
- Stage C：`NOT_READY_BOUND_INPUTS`。最大16-call adapterは未実行。Native一様boundとreferenceの未包含数値不確かさ等が必要。
- Stage D：実PCG64 joint stream、厳密包囲、shot-level simulation、actual CAは対象外。

全出力で実PCG64差を0に置かず、`error_evidence/all_law_synthesis/actual_CA_achieved/rigorous_CA`の未資格を維持した。
新native build/compile・trajectory sampling・PF/RTE生成・physical/own-ground eigensolve・RPE/shots評価は0件。
既存SOURCE/raw/契約/grant/ledger/marker/STOP、元dirty checkoutのHEAD/index/変更状態、共有venv/OS設定/他jobを保全した。
今回のmodel replayに含まれる小さいone-body基底対角化とhead/event/round計算は、固定38-call Stage Aの実行範囲に含む。

研究概要・当日ノート・validation status/manifestにはこの限定結果と未資格を追記し、旧結果・旧Evidence MapのClaimを変更しない。
公開後はremote SHA/tree、報告本文、主要blobを照合する。

**Mandatory STOP。** 次の候補はCodexによるWhole-P0一様評価法とreference数値不確かさの技術準備。
ユーザーが次の対象範囲を判断する。Stage B/Cの新科学実行や重要科学レビューは今回の許可に含まず、自動開始しない。
