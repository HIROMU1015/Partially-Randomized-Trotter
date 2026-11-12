# H4 PR m1保存RMSE診断の準備とFull PFのN/RMSE条件

2026-11-13 JST。**m1一候補の保存scalar診断を実行できるadapterを準備し、正式B V2との一致を確認した。実m1の数値再解析は未実行。** Full PFはDの既存物理request・保存carrier・A評価器の差分を整理した段階で、公式/幅/Mや科学的RMSE policyを採択していない。

[metadata preflight](PREFLIGHT.json)／[Full PF SOURCE/保存field索引](FULL_PF_SOURCES.json)／[HANDOFF](HANDOFF.md)／[人工・回帰検証](TESTS.txt)。実分子の新signal/normalization/N/RMSE、保存値の数値再解析、array load、sampling/decoder simulation、回路/価格/Gはすべて0。A既存branch `codex/pr-r-dependent-n-model-20261111`を継続し、B/C/D worktreeを変更していない。

## 固定SOURCEと既存成果の再利用

| SOURCE | 今回の利用 |
|---|---|
| A V2 `75b5c89bddf8a1ffa4a8ac99b2c06d6e05fa012d` | 元V2 configuration/packet、13round coverageを変更せず参照 |
| A S0公開 `7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75` / 科学 `7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e` | m0/m6はnominal64/38のcopyのみ。S0 scalar gate・allowance式の固定SOURCEを再利用 |
| 正式B V2 `030f3f548a35d98a88e62e92a35f132e0243a293` | 正本CONFIGURATION_V2とA正本の完全一致、B `validate_A_v2_packet`でS0 packet受入を確認 |
| D `eecaa8030280f8957ad85555bcc89e2daa1a29fa` | `full_baseline_v2_requests`のcatalog/spec builderを変更せずmetadata限定で再利用。標準Yoshida4をAで再展開しない |
| m1 carrier `89c91c692e1d35de94d2e7afbe890aabe6d80b7e` | H4 batch `run01/result.json:records[5]`。元科学SOURCE90dcc3ed20f30dfab6b22a68cc4adb9461485a98 |
| P1 `71ba49bb28b295aa28e51101bc3b1bf07d500767` | `checked_reference`の二経路gateだけを固定SOURCEから再利用 |
| Full PF carrier `155529e20a8fac39e00786c40b57830b71203c4f` | 旧long_timeのprimary/extension gzip、scientific provenance、保存field inventory。数値は評価しない |

B030のcontroller/integrationはA adapterが参照するaf284d853とbyte一致。Bのplanner/controller/共通identityを重複実装していない。元S0/P1/旧N/結果/SOURCE/input/raw/grant/ledger/marker/KEEP/STOPを変更せず、既存の数値moduleにも変更はない。

## m1：固定条件と技術接続

Linear H4・1.00Å・STO-3G・保存tol-only DF10、sector Nα=Nβ=2/dimension36、保存数値state条件付き。S2/L_D1/ΔPF=.1/K2/h=.1/M12、P=0、無shift、元H/state/snapshot/λ_R/prior/decoder/profile/lawを保持。εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski/finite_B_diagnostic。κは元S0のexact有理数文字列を参照し、表示小数から再生成しない。

- Canonical V2 ID：`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。
- m1/T=.2、qRPE=2、qPF=2、R_outer=4、S2一piece/8 event位置。
- physical ID：`c00e00dc1e9c414fb7d9`。qRPEはRPE multiplier、qPFはPF outer数で、同じ数値でも意味は異なる。

[新adapter](../../../src/trottertracks/h4_pr_m1_rmse.py)の`prepare`は正本B V2・旧A packet・保存candidate/structure/binding/profile/law/SOURCE・field存在だけを照合する。mean/reference/allowanceの数値gateや算術は呼ばない。m0/m6の64/38を表示し、**他11round nominalはnull、全十分N/global boundはnull**のまま。B正式S0-only validatorによる受入は完了した。

`diagnose`は将来の科学leafであり、今回は人工信号にだけ使用した。実保存値に対しては別承認が必要。

1. 既存R-aware `analyze`の**point作成より前のgate/allowance部分だけ**を固定ASTから取り出す。旧`analyze`本体・Hoeffding solver・shot_recordは起動しない。
2. `stages=q[2(prefix+1)J+J]+2 events`、既存roundoff/u/logB margin式を維持。過去のu/marginが大きければそれも保持し、`max(old, R-aware)`を使う。元allowanceを上書きせず`previous_allowance`と実carrier SOURCEを併記する。empiricalでありcertified=false。
3. 元S0 `numeric_diagnostics`の固定関数本体、P1 `checked_reference`、元`supplied_disk_geometry`を再利用。独立reference差≤1e−10、norm/vector closure≤1e−9等の元gateを緩和しない。rawをcorrected/Bから事後定義しない。
4. S0関数のlegacy locator変数をm1の実carrierへ明示bindingする。geometryの`error_source`は**89c91c692…のH4 batch＋records[5]**で、元P1 locatorの誤流用はない。
5. scalar geometryの後、meanだけを`A.mean_evidence/evaluate`へ渡し、finite_B経験的nominalを得る。旧HP NやplanをRMSE evaluatorへ渡さない。actual law/実装/ground誤差はunknown/nullのまま。

### 将来packetのscope

将来のm1結果は同じcanonical V2へ、m1だけ新診断として接続する。m0/m6のmodel recordは元S0 resultからbyte由来の値をcopyし、再計算しない。新packetでは組立SOURCEと各roundの`diagnostic_source`、`preserved_S0_packet`を分離する。m0/m6は元SOURCE、m1は新科学SOURCE、他roundはnull。

**B030の`validate_A_v2_packet`はS0のm0/m6専用で、m1の値を受け入れるvalidatorではない。** 新scopeには`diagnostic_scope.newly_reanalyzed_rounds=[1]`を付け、S0 packetへ偽装しない。B既存`n_from_a`でfull identity/lawを保持して変換できるが、m1公表後の専用受入条件の更新はB担当である。共通C record API・価格処理は変更しない。

## 次のm1科学実行案（未実行）

[CLI](../../../scripts/run_h4_pr_m1_rmse.py)の既定はmetadata preflightのみ。

```bash
python3 -B scripts/run_h4_pr_m1_rmse.py --out <新規preflight.json>
```

実行を承認する際の固定範囲は、上記m1一候補の保存値について、reference pair再確認、独立raw/corrected・exact-tail/norm/closure・bias分解、SOURCE付きR-aware allowance、phase/radius、経験的nominal N_X/Y一組とcanonical packetへの接続まで。**新finite mean/内部対照/reference action/finite normalizer/H-state array loadはすべて0**。保存logBのexp表示・局所geometry・nominal算術は、この別承認scopeに含める。

将来は公開した今回のmodule/runnerを科学SOURCEとして固定し、`--execute --source <固定commit SHA> --out <新規run directory>`を明示する。これは科学SOURCE指定であり承認証明書ではない。今回`--execute`は実データで一度も呼んでいない。

不足field、SOURCE/candidate/law不一致、保存FAIL、reference/norm/closure gate不成立、非有限値、丸め未解決なら結果をSTOP_FAILED_NO_FALLBACKとして保存し停止。新mean取得・normalizer再計数・科学retry・候補/閾値/κ/Mの変更へ進まない。m0/m6再解析、m2–12、新decoder統計、価格/Gは範囲外。

## Full PF：SOURCE対応とA評価器との差分

| 確認対象 | 再利用経路・条件 |
|---|---|
| Fullとtruncated | D `full_pf_spec/requests`のFull PFはone-body＋全DF fragmentを元順で保持。H4 rank10/H6 rank19。H4 B0/prefix7は別物。旧`h46_m5_plan.SYSTEMS.H4.rank12`を現在DF10へ使わず、Dの保存`h46_long_time_signal.RANK`とbindingを使う |
| H/state/sector | 旧`scientific_provenance.inputs`とD request bindingを照合。H4 sector(2,2)/36、H6(3,3)/400。保存DF-H/保存数値stateを用い、厳密groundや有効PF固有stateを保証済みとはしない |
| S2 | D formula `2nd`→`weights=(1)`、B V2生成可。A `structure`・D2 actionで同じ順序を扱える |
| 標準Yoshida4 | D formula `4th`→元`h46_m5_plan.weights`の3 signed piece。signal/actionは既存対応。固定B V2生成とA common.structureは未対応なのでB/D接続待ち。m5_bestへaliasせずAで展開実装しない |
| m5_best | D formula `4th(m5_best)`、既存11 signed pieceとbinary64 formula ID。B V2/A structure/action対応。負時間・積順を維持 |
| T/qPF/幅 | D requestのphysical candidate、T、qPF、actual_delta_pfを使用。B nativeはceil gridを許すため、宣言幅と実T/qPFが異なり得る。A旧fixed-grid scheduleのexact割切りを無断適用しない。Full PFの公式/幅/Mは今回未採択 |
| scalar | `evolve_batch(method=FULL)`/D2 `evolve`は全blockのsigned半step列後、ham.constantをtotal Tで一回。PR tailのextracted scalarやidentity normalizationを加えない |
| normalization | Full PFはRTE eventなし、R_outer/K/κ/λ_R不要、設計上B=1/logB=0。raw=corrected=deterministic PF signal。PR finite-K overheadを乗じない。ただし元H ground stateは有効PFの固有stateとは限らず、unitaryでも信号振幅が1とは限らない |
| mean/reference | D SOURCE `h46_long_time_signal.context/evolve_batch`、A `existing_finite_mean`→D2 actionが物理kernel候補。新科学実行では保存H/stateを復元し、既存half-step/action/reference pathを使用。contextにはspectral処理もあるので、単なるmetadata readerとして実行しない |
| N/RMSE入口 | D `requests()['rounds'][m]['A']`のidentity/law/physical IDを正本にする。A既存scheduleはPRと旧PF B0を対で作りκ/tail入力も要求するため、D Full PFへそのまま使わない。PF-only schedule/packet接続は将来の最小adapter課題で、偽κ/Kを補って既存APIを満たさない |

### 保存carrierの存在と不足field

[FULL_PF_SOURCES.json](FULL_PF_SOURCES.json)はprimary/extensionの**136個の異なるFull PF candidate**をmetadataで索引化した。H4/H6それぞれS2 24、標準Yoshida4 22、m5_best 22。全候補にsignal recordはあるが、これは新しいgate合格ではない。H6のS2/標準Yoshida4各1件はallowance evidence自体が欠測。他にもobserved norm scalarの欠測等があり、SOURCE gateと観測証拠を分けている。人工null-signalの境界テストも行い、欠測を0にしない。

保存gridはT=.8/1.2/1.6/2.0と元Q、追加q32の一部。DF10/19・formula/prefix/T/q/実幅/physical ID・H/state/snapshot/profile/law/scalarが一致する場合だけ再利用候補になる。旧HP Nは採用しない。grid hintだけで保存meanを取得済みとせず、非一意な近傍や別formulaを使わない。reference pairは同binding・同Tの既存A catalogとfieldで照合した。H4 T=.8/1.6の独立pair不足は従前どおりであり、単一full-H referenceと区別する。

### Full PF用RMSE policyで未解決の事項

PR原論文型の経験的shot則をB=1と置けば形式上は各軸`ceil[11+4(M−m)]`となるが、**今回実候補のNを算出していない**。既存A policyのPF Cmax=.08も経験的定数で、現在Full PF/state/decoderへ科学的に適用済みとはしない。D specが共通比較用にPR policyをcopyできることは科学的採択ではない。

Full PFでもTrotter誤差・RPE統計誤差・誤branchが残る。元DF-H ground energyと有効PF Hamiltonianの固有値差、保存stateの有効PF固有stateへのoverlap、signed公式と幅の誤差、prior/alias/全round decoderの分布を確認する必要がある。局所signal biasをenergy RMSEやεtrot上界へ直接換算しない。Minkowskiを用いるなら、各成分のL2誤差契約とtargetが整合する証拠が必要。

将来ユーザーがFull PFの対象系・公式・幅・M・h/prior/decoder・RMSE target/予算・policyを選び、D/Bのrequestを固定してから、保存一致mean/referenceと不足証拠を判定する。保存再解析又は不足actionの具体的scopeを別承認する。8次PF・Full-H RTE・価格評価は今回の実装/科学対象外。

## 13roundとの関係、検証、STOP

元coverageは不変：m0/m6だけnominal済み、m1は今回準備だけ、m3/m4は保存meanがあるが独立pair・backend/数値allowance bridgeに課題、m2/m5/m7–12は一致mean欠測。m1実行後でも新しく評価されるのはm1一候補で、残る10roundは未評価。nominalと十分N、局所biasと全energy RMSE、数値state条件付きとactual sampling/実装誤差を区別する。全体RMSE boundはnullのまま。

新11件の人工/metadataと旧S0 7・RMSE17・V2 10、計45件のテストがPASSした。初回人工テストでsigned metadata受渡し不足を検出して修正し、旧archiveのnull allowanceも欠測として保持した。最終結果は[TESTS](TESTS.txt)参照。作者自身の技術確認で、独立科学レビューではない。

**ユーザー**が次にm1一候補の保存scalar再解析を承認するか判断する。**A**は承認後だけ固定SOURCEで一括実行し、**B**へ新m1 scopeとS0由来を分離した未資格packetを渡す。**B/D**は必要に応じてFull PFのPF-only identity/標準Yoshida4接続を担当し、Full PF科学条件はユーザーが別途選ぶ。今回はcommit/push・remote照合後**mandatory STOP**。
