# H4 Pauli central R0：binary64経験的実装残差診断

PF・PR index0のcosine **2論理診断を完了**。新状態作用は**成功5／総試行5**、科学的な失敗・retry・回復実行は0。
直接差dはPFで`1.8722085654184194e-14`、PR0で`7.860978615986437e-16`となり、引用した保存ρ予算より小さい。
判定は両方とも **`EMPIRICAL_ONLY_NO_RHO_UPPER`**。
`numerical_error_upper=null`、`rho_upper=null`、`LOCAL_BUDGET_MET=false`、正式certificate未受理を維持する。
今回のworkerはnative exit=0、reap・保存検証完了、**mandatory STOP**。

## 1. 固定条件と来歴

Linear H4／隣接1.00 Å／STO-3G／system8＋ancilla wire8、Pauli-JW非identity184項／PR prefix27。
S2、T=0.8、q=8、r=4、K=4、元event law・保存trajectory・raw rayを維持した。
対象はD2保存central-control PF singletonとPR index0のcosine。DF、H2、別系列S2/Yoshida4解析には接続しない。
Qiskit 1.3.0、basis `rz,sx,x,cx`、optimization1、seed17、backend/coupling/layout/routing指定なしの保存出力を使用。新compileは0。

| 来歴 | commit |
|---|---|
| 基点 | `2734b97c2dd5f017e93cd81619ff52a6a0c6f16b` |
| 今回の科学SOURCE（作用前にcommit） | `84a326598b006384d13c7a354a3eec6881ee841c` |
| 元H4 | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| C | `e80d7510880f8db993b777c627bb8e95b05a070d` |
| D2 | `74515d8eb4f073c40fc783681a3775dbca89e9b1` |
| A_AXES | `a37232f85812688c3b9ab957c5073c5ce143b17e` |
| 条件付きC/N/G | `d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34` |

専用branchは`codex/non-df-h4-central-r0-empirical-20261024`。
基点と主要SOURCEの6commitをGitHubから実取得し、targets・C実装・保存C結果・PR0 price_joinの4blobを照合した。
[GitHub取得記録](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/github_verification.json)と
[承認原文](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/approval.txt)を保存した。

| 対象 | trajectory ID | C-target |
|---|---|---|
| PF | `d761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d` | 保存C PF区間の同一rayへのscalar変換 |
| PR0 | `93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97` | 保存32eventの選択trajectoryをbinary64で1作用 |

## 2. 三信号と残差

元raw状態ψは再正規化せず、`ν=〈ψ|ψ〉`を全input/outputで共有した。
νは保存binary64対の二乗和のexact rational、作用時のbinary64表示は`1.0000000000000002`。
state hashは`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。
[state_identity.json](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/state_identity.json)にexact νを保存した。
最終measurementのみ除いた実保存QPYを`Statevector.evolve`で作用させ、
ancilla=0/1 sectorの振幅二乗和から`m=(p0-p1)/ν`を求めた。
確率APIによるclippingと出力norm除算は使わず、global/relative phaseを保持した。

| 保存signal（表示値） | PF cosine | PR index0 cosine |
|---|---:|---:|
| C-target mC | -0.16151585491295273 | -0.10648829323047102 |
| 未compiled input m_in | -0.16151585491251422 | -0.10648829323046691 |
| compiled output m_out | -0.161515854912934 | -0.1064882932304704 |
| B_saved | 1 | 1.2873709223899619 |

定義は`δ=B_saved |m_out−m_in|`、`η=B_saved |m_in−mC|`、`d=B_saved |m_out−mC|`。
B_savedを一度だけ掛け、B_upを代用しない。
保存点値をexact rationalとして照合し、両方で`d≤δ+η`を確認した。この恒等式検証は伝播誤差の保証ではない。

| 保存結果 | PF cosine | PR index0 cosine |
|---|---:|---:|
| compiler-only差 δ | 4.197753256107717e-13 | 4.502196843701322e-15 |
| C-target→input差 η | 4.384974112649559e-13 | 5.2882947052999665e-15 |
| **C-target→compiled直接差 d** | **1.8722085654184194e-14** | **7.860978615986437e-16** |
| 保存ρ予算（ε_sig=.0005, cosine） | 2.6490038338323596e-6 | 2.593611368819805e-6 |
| d／保存ρ予算 | 7.067594774711455e-9 | 3.030900739598273e-10 |
| 保存ρ予算−d | 2.649003815110274e-6 | 2.593611368033707e-6 |
| 経験的大小 | BELOW_SAVED_BUDGET | BELOW_SAVED_BUDGET |
| 保証判定 | EMPIRICAL_ONLY_NO_RHO_UPPER | EMPIRICAL_ONLY_NO_RHO_UPPER |

予算のexact値はPF `10107730069649565297/3815672118158673332666368`、
PR `19792741170690161541/7631344236317346665332736`。
[guard_inputs_quoted.json](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json)の保存行を引用した。
新しいheadroom、shot数N、総費用Gは計算していない。
全exact値は[results.json](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/results.json)、転記表は[results.csv](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/results.csv)。

### C-targetの違いと数値精度

PFはCの`implemented_law.PF.corrected_encoded_mean.real`の保存端点をexact νで除し、
その区間内のexact rational midpointをmC代表値とした。
保存C norm区間がexact raw νを含むことも照合した。
表のmCは表示用binary64であり、残差会計には丸めた表示値を使わない。
元区間・変換端点・代表値は[PF_C_target_saved.json](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/PF_C_target_saved.json)。
保存区間を新しいcompiled信号の包含証拠へ転用していない。

PR0は元32event（q8×r4、今回の記録はorder0）とCのretained blocks／符号／rotation angle／native scalarを使用した。
既存`event_unitary`、Pauli operand、C・trajectory-referenceのforward/events/reverse/scalar順をvector作用へ接続した。
保持項angleは保存係数とstep timeのexact operandからbinary64三角関数へ変換し、その誤差は保証していない。
新Hamiltonian、generic回路、sampling、full C/Lは生成していない。
[PR0_C_target.json](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/PR0_C_target.json)の
`PR_law_mean_used=false`、`normalization_B_applied_to_target=false`を確認した。
PR0は選択trajectoryの点評価であり、PR law mean・母平均残差・一様上界へ外挿しない。

## 3. 位相・scale・control対応と限界

本番前と本番内で、source body／controlled body／input／compiledのPF・PR0、合計8 QPYを実decodeした。
元SOURCEのPauli recipe token、D2 central rotation/productの静的token mapping、sidecar、
Hadamard wrapper、wire8→clbit0、fingerprint・SHA、compiler profile・保存histogramを照合した。
未対応gate・中間measurement・reset・condition・未解決parameter・layoutは検出していない。
本番前と本番内の[preflight](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/preflight.json)は同一。

両bodyの保存aggregate phaseは`0.2651822507334482`、controlled全体phaseは0。
branch0は恒等、branch1は元bodyに対応するSOURCE token列を確認し、aggregate phaseはancilla Pとして一度だけ扱った。
compiled全体phaseはPF `1.703387452164006`／PR0 `1.7033874521616523`であり、そのまま保持した。
**限定照合と2信号では位相・scale・controlの技術的不一致を検出していない。**
新whole-operator検証や一般compiler certificateを実施したという意味ではない。

PFはsource body 2944／compiled 48967 instructions、PR0は464／3140 instructions。
両方でm_inがmCより大きく、m_outがmCへ近づく向きとなり、δとηが打ち消し合ってdが小さくなった。
PFのδ・ηがPR0より大きい背景には異なるprogram長や演算順があるが、
この差がroundoffだけで生じたという因果確定も、誤差スケーリングの推定も行わない。

保存された作用後normはPF input `0.999999999997284`、PF output `0.9999999999998872`、
PR0 input `0.9999999999999707`、PR0 output `0.9999999999999821`。
これらで信号を再正規化しない。伝播・三角関数・matvec・確率縮約・νのbinary64変換には未保証の数値誤差が残る。
特にPR0のdは浮動小数点丸めの影響を受け得る桁であり、真の非零残差やその桁数を確定しない。
PFも保存C区間だけではcompiled/input伝播の数値誤差を包含できない。
将来の区間幅目標`1e-8`を今回の不確かさへ流用していない。

## 4. 実装・人工検証・実行監査

[production adapter](../src/trotterlib/h4_r0_empirical.py)と[runner](../scripts/run_h4_r0_empirical.py)を最小追加した。
既存`h4_central_rho_plan`のscalar／mock検証資産、`h4_central_cng`系の保存会計値、D2 control規則、
`h4_a_axes` signature、元fingerprint、Pauli recipe、`event_unitary`、CallBudget、native witnessを再利用した。
toy-only endpointのflag変更ではなく、実保存QPY・raw state・選択C-targetへの明示接続である。
`fixed_signal_evaluation`を確認し、generic信号再生成・sampling経路は呼んでいない。

新[人工10テスト](../tests/test_h4_r0_empirical.py)と既存準備15テスト、合計**25 PASS**。
raw norm／clipping／位相・control／誤った測定配置／C event符号・angle／B一回／保証非昇格／有限callを確認した。
これは主担当Codex自身の検査であり、別reviewerによる独立査読ではない。
[新10最終log](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/artificial10_final_PASS.log)、
[再利用15log](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/reused15_PASS.log)。

SOURCE commit後にcode17、runtime6999、入力89、旧証拠保全192、承認・出力schemaを封印した。
[seal](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/seal.json)と[active contract](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/active_contract.json)を保存。
新one-shot grantは`eb41a2201a4d3d791cd2144711fe2ab91ceb3a2753db45c9c29f9686805ac25b`。
2026-10-24 10:32:21 UTCに一度消費し、全callを実行前予約した。

| 新状態作用 | 予約／成功／総試行 |
|---|---|
| PF input | 1／1／1 |
| PF output | 1／1／1 |
| PR0 input | 1／1／1 |
| PR0 output | 1／1／1 |
| PR0 selected C-target | 1／1／1 |

[call ledger](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/call_ledger.json)には
これに加えてpreflight1と論理診断各1があり、合計8予約。PF C-target読取は新状態作用ではない。
worker PID615052は2026-10-24 10:32:59 UTCにexit0でreap、10:33:00 UTCに[STOP](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/STOP.json)を保存した。
stderr空、`all_owned_reaped=true`、`automatic_next_stage=false`。旧H4 native PASSを補完する証拠ではない。

[保存検証](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/verification.json)はPASS。
code17／runtime6999／入力89／保全192のSHA、旧manifest182行のdigest不変、grant消費、全receipt、
保存scalar恒等式・三角不等式・quoted guard・native終了を照合した。
後検証の新状態作用0、QPY decode0。科学再実行や保存vectorからの信号再縮約は行っていない。
[出力inventory](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/output_inventory.json)はrun01の30ファイルをSHAで固定する。

## 5. 技術回復履歴・運用版

SOURCE前のmetadata準備でruntime.jsonがA_AXES直下にあると仮定し、パス読取が失敗した。
既存A_AXES sealの`runtime_manifest`参照を辿るよう修正してmetadataだけ作成し直した。
実参照先は`artifacts/non_df_h4_d1_execution/2026-10-24/runtime.json`。
初回logと[回復記録](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/pre_SOURCE_recovery.json)を保持した。
当時の科学grant0／状態作用0であり、成功5／総試行5に隠れた科学失敗はない。
SOURCE固定後のコード変更、科学的中断、retry、新grant再発行は0。

最新版`research_project_instructions_unified_2026-10-23.md`の§8.4・10.1・10.2と共通AGENTSを適用した。
§10.3は利用可能版に存在せず版不一致として記録し、未取得本文を読んだとは扱わない。
直接指示の再利用・研究目的優先・過剰管理回避を適用した。
固定5program walkは256/512 amplitudesと短いため単一workerで入力を共有し、再読込を抑えた。
CPU/RAM/thread/affinityの独自上限、共有環境変更、他processの停止・変更は加えていない。

## 6. 保持する未解決状態

| 項目 | 今回終了時 |
|---|---|
| Pauli-PR historical fingerprint | FAIL／原因UNKNOWNを保持 |
| 元H4 native終了 | UNKNOWNを保持 |
| 旧20精度行のN/G | null・UNDETERMINEDを保持 |
| C/L誤差会計 | COMBINATION_BLOCKEDを保持 |
| 旧compiler証拠 | MISSING_EVIDENCEを保持 |
| 今回の局所ρ上界 | null、経験値のみ |
| PR law全体のρ | UNKNOWN／null |
| PR母平均費用tail | UNKNOWN |
| 正式certificate | 未受理 |

未観測order4を含むevent lawはPR0にも保存256 trajectoryにも網羅されない。
PR0の局所結果から全law転送や母平均費用tailを解決しない。
Arb／precision escalation／R1／R2／sine／新compile／sampling／Hamiltonian-state生成／full C-L／N-G再解析／正式受理はすべて0。

## 7. 次段階の提案（未実行・明示承認が必要）

| 候補 | 担当と最小作業 | 判断・優先度 |
|---|---|---|
| A：R1 order2 cosine | ユーザーが対象・call上限を承認後、Codexが保存PR index1のinput/outputと選択C-targetを限定診断 | 経験的範囲拡張の第1候補。非可換product・sign・phaseを今回のorder0から分離して検査 |
| B：R1 sine | ユーザー承認後、Codexが実保存sine QPYのPF/PR0を診断。今回PR0 C-targetの既存複素点値は来歴確認のうえ再利用候補 | 経験的範囲拡張の第2候補。測定軸をcosineから外挿しない |
| C：Arb区間評価 | ユーザー承認後、Codexが固定C-targetとcompiled期待値のvalidated enclosureを実装・限定実行 | 局所ρ上界が目的なら必要。binary64の小さなdだけでは代替不可。原因修正のために今すぐ必須とは判定しない |
| D：R2全law転送 | ユーザー承認後、Codexが未観測eventまで含むPR全law上界の転送設計 | PR全law保証には別途必要。A/B/Cの局所結果だけでは解決しない |

大きなSOURCE不一致は検出されなかったため、再compileや入力の選び直しは提案しない。
経験的な実装範囲確認を優先するならA→B、局所保証の確立を優先するならCを先に承認する。
新SOURCE／対象call上限／未使用grantを別に固定して実施する。今回は提案までで停止する。

## 8. 成果物と公開状態

[artifact入口](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/README.md)、
[実行結果](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/result_summary.json)、
[保存検証](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/verification.json)、
[実行STOP](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/STOP.json)、
[完了監査](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/completion_audit.json)、
[公開状態](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/publication_status.json)を保存する。
成果物commit・push試行・確認可能なremote SHAは公開状態を参照する。SOURCEと結果commitは区別する。
過去のpush失敗履歴は変更しない。今回の認証失敗時はlocal保全し、最終チャットへ再実行コマンドを提示する。
**mandatory STOP。追加数値診断・Arb・R1/R2・N/G再解析へ自動移行しない。**

## 付記：SOURCE固定前の準備記録

基点 `2734b97c2dd5f017e93cd81619ff52a6a0c6f16b`、専用branch `codex/non-df-h4-central-r0-empirical-20261024`。
固定Linear H4/1Å/STO-3G/n8+ancilla8、Pauli184/prefix27、S2/T=.8/q8/r4/K4。
D2保存PF・PR index0のcosineに限定し、新状態作用最大5。
新10＋既存15人工PASS。実QPY8個をdecodeしSOURCE token・phase・wrapper・input/output fingerprint・nativebasis・measurementを照合、状態作用0。
PF C-targetは保存intervalの同一ray scalar変換、PR0は保存event列のC整合vector作用。
新SOURCEをcommitし、runtime/input/output schema/承認と未使用grantを結合した後にだけ状態作用を行う。

runtime metadataの参照先をA_AXES直下と仮定した処理が失敗したため、既存sealのruntime_manifest参照を使用するよう修正。
SOURCE前・science grant0・状態作用0であり、失敗logを保存してmetadataだけ再実行した。

出力はEMPIRICAL_ONLY_NO_RHO_UPPER、numerical_error_upper/rho_upper=null。
PF保存区間からexact scalar midpointを取り、PR0 C-targetはbinary64点評価とする。
Arb/precision escalation/R1/R2/sine/新compile/sampling/Hamiltonian-state生成/C-Lフル計算/N_G/正式受理0。
旧hash FAIL/原因UNKNOWN・元native UNKNOWN・旧N/G null・C/L/旧compiler/PR全law/tailを保持。

[承認原文](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/approval.txt)、
[GitHub照合](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/github_verification.json)、
[構造照合](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/structural_preflight.json)。

最新版規約§8.4/10.1/10.2を適用。§10.3は利用可能版に存在せず、今回の直接指示を適用する。
単一workerで固定5program walkを行う。入力QPY・state・runtimeを共有して再読込を抑え、CPU/RAM/threads/affinity上限や共有環境変更は加えない。
完了後mandatory STOP。
