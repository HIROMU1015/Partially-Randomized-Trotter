# 新A V6-P：限定数値screenの実行準備

2026-11-09、SERVER 2。**準備完了・科学campaign未実行。** 添付指示§8の「別途の明示承認後にのみ実行する作業」と§11の「実行準備完了時点でSTOP」に従い、今回の作業はadapter・人工テスト・入力照合・公開までで終了する。

固定計画は[e5d6e388のNEXT_VALIDATION_PLAN.md §1〜2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5d6e388ee2543b47ebcc1dfd9e19862d08ac327/docs/research/pr_reference_v6_static_20261108/NEXT_VALIDATION_PLAN.md)。科学的入力／再利用SOURCEの正本は **f2638a7845685a119619382b850441f5d8d33713**。この準備をB1実装、新しい成功資格、科学実行許可とは扱わない。

## 1. 固定scopeと未実行の評価

Linear H4、1.00 Å、STO-3G、DF12、元H厳密ground供給・準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit 1.3.0／compiler A／rz,sx,x,cxを保持する。B′4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event相対位相・X/Yも変更しない。

対象は保存m17、q=131072、r=32、bits=88。保存B′626／Reference630 shots/axis、他17roundのq/r/bits/N・単価を保持する。精度1.5936e-4 Ha、failure=.05、phase cap=.4 rad、αaxis=1/720を固定する。E0・gap・groundの古典記述は追加利用しない。

将来の単一campaignは、(1)保存MとB_normからδ_B1=B_norm+2·2⁻⁵³Mを外向き評価、(2)Reference m17の条件付き十分N、(3)保存単価で条件付き全18round総native RZを再会計する範囲だけ。**今回はこの三つの数値を一つも評価していない。** 新しい位相S／Φ、F/G、trajectory、回路、価格は生成しない。

## 2. 準備したSOURCEと既存処理の再利用

- [実行adapter](../../../scripts/diagnose_pr_reference_v6p.py)：固定保存blobだけを読む。`inspect-inputs`はbyte provenanceだけ、`run`は別途許可が必要な科学campaignである。import時に保存科学データを読まない。
- [人工テスト](../../../tests/test_pr_reference_v6p.py)：toy disk／toy価格とmockのshot値だけを使用。科学入力のcoverage・旧N・RPE成立を検証した証拠ではない。
- [SOURCEとtarget](../../../artifacts/pr_reference_v6p/2026-11-09/preparation/SOURCE.json)、[入力blob／SHA256](../../../artifacts/pr_reference_v6p/2026-11-09/preparation/INPUTS.json)、[技術検算](../../../artifacts/pr_reference_v6p/2026-11-09/preparation/TECHNICAL_CHECKS.json)、[準備STOP](../../../artifacts/pr_reference_v6p/2026-11-09/preparation/STOP.json)。

既存`track_c_compiler_a_18round.py`から**I／upper／lowerの3定義だけ**をpinned SOURCEのASTから取り出して再利用する。元module全体のimportはQiskitやbuilderを読み込むため避けた。192-bit directed intervalとnextafterによる外向き端点を保ち、新しい区間演算器は作っていない。

`diagnose_pr_reference_cost_structure.py`のinterval／index／split／JSON helper、`check_pr_reference_cost_structure_d0.py`のpair／sum／端点独立検算をpinned版のまま再利用する。両者のmain、D0 runner、phase_bound、qualify、common、量子builderは呼ばない。新しいRPE estimator・compiler・phase provider・承認管理機構を作っていない。

薄いadapter固有の処理は保存schema接続、指定された一つのB1式、既存shot算術への成分代入、m17 N差分会計、STOP／fresh出力である。新しい一般的なpolicy／最適化frameworkは含まない。

## 3. 入力確認と数値規約

正本commitの6入力（STAGE3、原ROUNDS、STAGE4、B′ERROR_RPE、B′TOTALS、D0 RESULTS.json.xz）と3再利用SOURCEをread-onlyで確認し、blob・byte hashを保存した。m17のq/r/bits、X/Yと9roleのschema、成功規約のfield名・文字列を照合した。原本の値から新しい和・差・N・価格順位は計算していない。

実行時には次を検算するよう準備した。

1. 欠損・重複・schedule／bits／α不一致を拒否。roleを0補完しない。
2. 旧Referenceは保存したcharged phase上界とshared/native raw成分から再構成し、旧630を検算。B′は保存ρ/βから626との整合だけを確認し、scheduleを変更しない。**実入力による旧N検算も今回未実行。**
3. D_otherはsharedとnative rawの別成分から作る。丸め済みtotalからphaseをfloat減算しない。normalization、finite-bit proposal、native joint由来diskを保持し、旧compilerを二重課金しない。
4. nominal radiusは原ROUNDSのraw_radius_lower＋raw_extra_disk_upper。ρ/βを旧規約で外向き保存してから、192-bit区間の上端にceilを取る。m0〜m16のNは再計算しない。
5. 成分の保存端点を使うため旧内部intervalのbitwise再現とは呼ばない。旧630／626が再現しなければ候補評価・再会計の前にFAIL／STOPする。
6. m17の両axisに同じ保存worst-case保証を適用。価格は変更せず、9roleとwhole価格の合計を照合する。負のshot差を価格区間へ掛ける際は、同じ端点を旧費用・新費用へ対応させる。旧総区間に負の差分区間を独立に足して、不要に広い新総計を作らない。

B1を実現でき、最終Pが同じcompiler contextで1 RZへlowerされ他の単価を変えないことが、将来の費用比較の条件。現行B1 providerのcorrectnessを今回認定したという意味ではない。

## 4. 技術検算

**人工11テストPASS**。importの科学I/O不在、旧interval helperの外向き性、normalization保持、raw成分組立、整数ceil、人工diskに対する単調性、異常値拒否、負のshot差と端点相関、signed D融合、coverage／N不一致、toy schema接続、旧N不一致時の即停止を確認した。実科学入力を使うcampaign、B1上界の新評価、m17 N算定、全18round RZ再会計は0回。

既存project venvのPython／mpmath 1.3.0を`-B`でread-only利用。host GPUA100-gleap02、affinity 64 CPU、観測MemAvailable 515444248 kB、CPU／memory PSI avg10は0.00だった。小さい保存値／人工算術なので重複する大規模workerを起動せず、資源上限・共有設定・他job・他worktreeは変更していない。著者自身の技術検算であり、別の独立科学レビューではない。

## 5. 別途の明示承認後に使う実行方法

**次のcommandは今回実行していない。** このadapterを含む公開commitを`--source-commit`へ指定し、一つのfresh出力領域で一回だけ実行する。SOURCEのbyte一致を確認し、既存の失敗／完了出力を上書き・resumeしない。別候補や別pathでの自動再試行はしない。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/pr-reference-v6p-prep-20261109
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-h4-parallel-pilots-20261106/.venv/bin/python -B \
  scripts/diagnose_pr_reference_v6p.py run \
  --source-commit <この準備を公開したcommitの完全SHA> \
  --output artifacts/pr_reference_v6p/2026-11-09/P_run01
```

出力はSOURCE.json／RESULTS.json／STOP.json。RESULTSには旧B′、旧Referenceの再構成と保存端点、条件付きB1上界、残るdisk、ρ/β、m17 N、保存単価不変を仮定した全18round総RZ、9roleと検算を含める。例外時はFAILのSTOPを残す。

固定判定は`NO_PHASE_BOUND_IMPROVEMENT`、`NO_SUFFICIENT_SHOT_REDUCTION`、`SHOT_REDUCTION_WITHOUT_TOTAL_RZ_ADVANTAGE`、`CONDITIONAL_B1_TOTAL_RZ_ADVANTAGE`、`INCONCLUSIVE`、入力／旧N等の不整合時`FAIL`。総RZの差区間がReference有利でも、B1を実現・資格化できた場合の条件付き結果に限る。実装済みReferenceの優越・実機成功・総T削減とはしない。

一つのcampaign内で指定上界・m17 N・保存価格によるRZ再会計と必要な検算を完了し公開してSTOP。新しいB1 phase provider、V6-Q、A1未完成上界、fsum／別precision、量子simulation／build／compile、sampling、新guard／合成、CT系列、CX／depth／size／qubit／総Tの新会計、他Trackへ拡張しない。

## 6. 今回の完了・次担当

追加は専用branchのadapter、人工test、この短いREPORT、準備証跡4ファイルだけ。固定SOURCE・raw・結果・grant・ledger・marker・STOP・科学契約は変更していない。

**commit/pushとremote SHA／内容照合後、準備完了のmandatory STOP。** 次担当はユーザー／GPT：この公開SOURCEと固定scopeのV6-P科学campaignを一回実行するか判断し、実行する場合は明示承認する。shot削減余地、総RZ優越の可能性、V6-Qへ進む価値は、現時点では未判定である。
