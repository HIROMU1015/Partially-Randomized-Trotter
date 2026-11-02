# H4 Pauli PR0 cosine：個別C-target Arb包含接続の準備

**選択済みPR0の個別C-targetとcompiled回路を包含比較する接続を実装し、人工38テストを通した。準備完了、mandatory STOP。**
実H4のArb状態作用、新科学grant発行/消費、compile/sampling/Hamiltonian/state生成、N/G再解析は0。
実H4のC-target/compiled signal/ρ区間は今回未取得であり、旧経験値やlaw meanで補完しない。
次回のPR0 cosine・最大6状態作用案は**実行無効**のまま保存した。

## 1. 承認範囲・来歴

2026-10-25の[利用者指示](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/REQUEST.md)に従う技術準備。
専用branchは `codex/non-df-h4-central-arb-pr0-preparation-20261025`。
基点commit `78d7f24c80b500fc63c908efc768e683406571b3` をGitHub connectorから実取得した。
[GitHub記録](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/github_base_verification.json)。

| 参照 | 固定SOURCE |
|---|---|
| 元H4・RTE event | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| C | `e80d7510880f8db993b777c627bb8e95b05a070d` |
| D2保存PR0 QPY | `74515d8eb4f073c40fc783681a3775dbca89e9b1` |
| R0 selected-target schedule | `84a326598b006384d13c7a354a3eec6881ee841c` |
| PF Arb compiled bridge | `92a5f104a2e9962353870e87f1d910d236adba41` |

固定対象はlinear H4、隣接1.00 Å、STO-3G、Pauli-JW非identity184項/prefix27、S2、T=.8/q8/r4/K4。
system8＋ancilla wire8。PR index0 trajectory IDは
`93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97`。
PF cosineの既存局所上界9.565e-15以下は過去の別対象の結果として保持し、PR0には外挿しない。
DFや別系列H2解析の科学結果・条件・実行承認とは統合していない。

## 2. 実保存証拠への静的接続

[静的検査CLI](../scripts/inspect_h4_arb_pr0.py)は、保存source body/controlled body/input/compiledの**4 QPYをdecode**した。
新しい回路・control・wrapperの生成や状態作用は行わない。
R0のrecipe/token照合・control/phase/wire検査・fingerprintを再利用し、実model・join・state hashも確認した。
join自体のtrajectory/event hashと位置は既存pure `verify_join`を再利用する。
[検査結果](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/structural_inspection.json)。

- 保存32 eventは**すべてorder0**。順序はstep0〜7、各inner position0〜3。
- retained blockは27、各step forward27→selected event4→reverse27→native scalar。
  retained rotationは全体432、selected rotation32、native scalar8。
- compiled nativeはRZ1,145/SX138/CX1,856＝3,139 gate＋最終measure1。
- QPY SHA：`f68965bde6a001bc3a9dd6f0d55a5a76682b868738360cef620e926c0a99fa2c`。
- fingerprint：`f7e855e7560ccd9c64c188075c86a7a4c1b2f023272c9bf7f6fa9cf3147c169a`。
- raw state hash：`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。

[C exact operand plan](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/PR0_selected_C_plan_NO_EXECUTION.json)と
[compiled descriptor](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/PR0_compiled_program_NO_EXECUTION.json)を保存した。
これらは既存科学入力のliteral metadata接続であり、新しいH4信号・量子compile・状態計算ではない。

## 3. 個別C-targetの包含アルゴリズム

[新接続](../src/trotterlib/h4_arb_pr0_preparation.py)は、Cの実装primitiveとR0の個別trajectory順序を使う。
Cのlaw mean EngineやR0のbinary64 `selected_C_action/propagate`は起動しない。

1. Retained角度は保存binary64係数aとstep_time δの**exact積 aδ/2**。
   その積をfloatへ丸め直した値や十進意図値を代入しない。
2. Event角度はCの保存law angleと一致する**保存binary64値**。
   ここはretained exact積と異なる規約であり、両者のkindをplanに明記した。
3. 各eventは保存product列のP_l1、P_l2…を先に作用し、その後にsigned rotationを作用する。
   演算子表記は `(-1)^(k/2) V_rotation P_lk ... P_l1`。
4. Signed Pauliはbit permutationとexact ±1/±iで作用する。
   負productの符号をすでにPauliへ含めるため、別のπ位相を重ねて加算しない。
   order phase ±1は一度だけ作用する。
5. 各outer stepのreverse retained後、Cの保存native scalar operandsをexactに加算したexp(iφ)を作用する。
   compiled側の保存aggregate phaseをC側のexact operandへ代入しない。

Pauli labelは右端がqubit0のlittle endian。Cではunitary全体のphaseもraw braとの内積に効き、
Hadamard側ではcontrol=1 branchのrelative phaseへ対応する。compiled側のglobal phase・RZ/SX/CX列も既存bridgeのまま保持する。
PR0の実eventにproductはないが、汎用primitiveのorder2/4 product/sign/phaseは**人工fixtureだけ**で検証した。
実PR1データの読み出し・数値評価には拡張していない。

### 包含半径と共通norm

compiled bridgeの `_recenter` を再利用し、exact dyadic center mとglobal L2半径eについて`||x−m||₂≤e`を保持する。
unitary作用GのArb成分区間からfresh半径rを求め、`e′≥e+r`を外向きに更新する。
signed Pauli/±1のexact permutationでは半径を増やす丸め操作はない。
Rotationではcosθ I−i sinθ PをArbで包含し、exact operandの変換・trig・vector演算の誤差もrへ回収する。

初期raw braはexact binary64 dyadicであり、νはその成分のexact二乗和。
最終内積に対して **`|〈raw|x−m〉| ≤ sqrt(ν)e`** を用いる。
Arbによるcenter内積区間へこの半径を加え、同じνで除してcosine targetを包含する。
入力/出力state再正規化、出力norm除算、probability clipping、ν=1の仮定は行わない。

compiled側は同じraw ray＋ancilla0、最後のZ_anc/νを既存kernelで包含する。
両者のexact ν一致を確認してから、
`ρ_local = B_saved × |m_compiled − m_C_selected|` を**区間全体の直接差**で評価する。
**C-targetにBは掛けず、差の尺度で一度だけ適用する。**
保存Bは `1449450801590761/1125899906842624`、約1.2873709223899619。
B_upやlaw mean、経験的点値を代用しない。log_B/normalizationの正式証拠gateまで成立したとはしない。

## 4. 人工検証と実行境界

新13＋既存Arb10＋QPY6＋sine9＝**38テストPASS**。
[Arbログ一覧](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/arb_artificial_tests.json)、
[QPY/measurementログ一覧](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/qpy_artificial_tests.json)。

- 個別targetは2/3 system qubitで、既存C.pauli/acb_mat/signalの256bit dense参照区間を64/96/128bitの新vector区間が包含した。
- 非可換Y×Xとorder phase、負productについて既知の±sin(1/4)を包含した。
- exact retained積と丸めたfloat、native scalarのexact和と逐次binary64加算、eventのbinary64値と十進意図値の混同を拒否した。
- 共通normとraw不変、C-target未scale、B一回だけを確認した。
- system2＋ancilla1の3qubit人工列では、controlled RZ・relative phase・Z測定とC-targetを接続し、dense参照からのρを包含した。
  relative phaseを逆にした人工列はtoy予算超過と判定し、位相の欠落を見逃さないことを確認した。
- law mean/経験的点の代用、event順序/符号/位相破損、layout不一致、zero normを拒否する。

Arb/private環境とQPY環境は既存のものを使用し、packageの追加・共有設定変更は0。
新C-target数値入口は**private kernelを含め人工fixtureだけ**を受け入れ、保存H4/n8をArbロード前に拒否する。
`execution_request`は常に拒否し、CLIには科学実行/grant発行機能がない。
実H4のnumeric intervalは今回すべてnullである。

準備中にscalar counterexampleの人工fixture1件を修正して再検証した。
数値bridgeは変更せず、意図した逐次binary64加算を明示した。
[技術回復](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/technical_recovery.json)。
全検証は主agent本人によるもので、別reviewerによる独立reviewとは呼ばない。

## 5. 次回pilotの無効契約

[非実行契約](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/pilot_contract_DRAFT_DISABLED.json)は
`execution_authorized=false / dispatch_enabled=false / grant_valid=false`。
grant・科学runner SOURCE・launch commandはnull。現在は科学grant発行/消費0。

| 項目 | 次回案（未承認・未実行） |
|---|---|
| 対象 | 保存PR index0 cosineだけ、1論理対象 |
| 精度 | 64→96→128bit、必要時のみ |
| 各精度 | C-target状態作用最大1＋compiled状態作用最大1 |
| 最大状態作用 | **6**、途中失敗も非返還で数える |
| 局所差scalar縮約 | 最大3、B_saved一度だけ |
| 保存ρ予算 | `19792741170690161541/7631344236317346665332736` ≈2.593611368819805e-6 |
| 早期終了 | 上端≤予算または下端>予算が決まった精度でSTOP |
| 昇精度 | straddle/nonfiniteの場合だけ。128bitでも未確定ならINCONCLUSIVE STOP |
| 幅1e-8 | 参考目標。判定後の不要な昇精度やallowance変更には使わない |

同じprecisionのC/compiledは独立であり、将来の制御実装では別call予約の下で並列化や入力共有を選べる。
precision間は早期停止判断への依存がある。恣意的なCPU/RAM/worker上限は置かない。
非有限で残る片方が無用となる場合は未使用callを省略し、費やしたcallは保存する案を明記した。
これは無効の契約案であり、実行前にユーザーの明示承認と新production SOURCE/入力/runtime/契約の固定、
新しい未使用one-shot grant・namespace・native終了/STOP制御が必要。
科学的retry/回復が必要な場合も成功済み作用を無断で再実行せず、旧SOURCE/grant/失敗call/STOPを保存する。

## 6. 保全・残る事項・担当

旧証拠およびPF科学コード481ファイル、保存入力11 SHA、旧manifest187行を保全した。
PF科学SOURCE・既存結果・grant/ledger/marker/STOPを変更していない。
[保全/無効契約の検証](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/verification.json)、
[再利用と固定SOURCE照合](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/reuse_and_source_checks.json)。
提供された最新版2026-10-23 policyには§10.3が存在せず、版不一致を記録した。
実在する§8.4/10/10.1/10.2と今回の明示指示を適用し、共有環境・他ユーザーのprocessに変更していない。

今回の[技術STOP](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/STOP.json)は準備だけを閉じ、元H4 native証拠の代用にはしない。
旧N/G=null、C/L COMBINATION_BLOCKED、元native UNKNOWN、旧compiler証拠不足、PR全lawρ/費用tailなどの未解決状態は維持する。
今回の接続は個別selected trajectoryだけの設計であり、PR全lawや母平均の保証ではない。

次は**ユーザー**がPR0 cosine・64→96→128bit・最大6状態作用とscalar縮約の範囲を明示承認する。
**Codex**は承認後だけ新しいproduction制御/SOURCE/runtime/入力/契約/未使用grantを固定し、限定数値評価と終了監査を行う。
現在は実H4評価0。PR1/sine/R2/N_G/正式certificate受理には別の承認が必要。

## 7. Commit・公開状態

技術準備commit、push試行、remote SHAは[publication_status.json](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/publication_status.json)に保存する。
新しい科学SOURCE/grantではない。認証失敗なら全成果物をlocalに保全し、再pushコマンドを最終回答へ示す。
公開処理後mandatory STOP、数値実行へ自動移行しない。

技術準備commitは `4e97841b87b6cc032639f0e616fdd2c01c48f244`。
push1回はHTTPS Username取得不能でexit128。
[失敗log](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/push_attempt01.log)、
[remote再確認](../artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25/remote_after_push.json)を保存。
専用branch検索0件、remote SHA=null。認証可能な環境で同branchをpushする必要がある。
後続local metadata commitは公開状態だけを記録し、接続code・人工結果・無効契約・旧証拠・STOPは変更しない。
実H4状態作用/新科学grant0、mandatory STOPを維持する。
