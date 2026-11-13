# 標準Yoshida4 Full PF・Full-H RTE明示layoutの物理request接続（Codex D、2026-11-13）

**更新（2026-11-13）**：B f0c4bb55への最終接続と保存価格17件のreceipt照合は[d_full_pf_full_h_v2_final_receipts_20261113](../d_full_pf_full_h_v2_final_receipts_20261113/REPORT.md)。

前段のD adapter（`eecaa803`）を拡張した。

- **物理schedule層**を先に作り、公開済みのB V2が生成できる範囲だけをV2 configurationへ結合する。
- 標準Yoshida4のFull PFと、κなしの明示layoutのFull-H RTEは、物理requestとして完成している。ただしV2 IDは、Bが対応interfaceを公開するまで**null**のまま（IDを作って補っていない）。
- **D 16＋B回帰80＝96技術test PASS。**
- 8次PFは実装・接続・検証していない。
- 新しい科学計算は0件：作用、N/RMSE、sampling、compile、価格、fit、G、数値探索。

成果：
- [adapter](../../../src/trottertracks/full_baseline_v2_requests.py)／[test](../../../tests/check_full_baseline_v2_requests.py)／[出力CLI](../../../scripts/report_full_baseline_v2_requests.py)
- [SOURCE_CATALOG](SOURCE_CATALOG.json)／[REQUEST_EXAMPLE_ARTIFICIAL](REQUEST_EXAMPLE_ARTIFICIAL.json)（人工のpolicy・layoutのschema例）
- [HANDOFF](HANDOFF.md)／[TESTS](TESTS.txt)
- 前段の資料：[REPORT](../d_full_pf_full_h_rte_v2_paths_20261113/REPORT.md)

## 1. B interfaceの確認

B最新`030f3f548a35d98a88e62e92a35f132e0243a293`（H4 S0のV2結合）を確認した。controller、evidence planner、pipeline、integrationは、`af284d85`から変更されていない。

| Bの予定interface | 公開状況 | Dの対応 |
|---|---|---|
| 標準`4th`の公式hook | 未公開 | 物理requestは完成。V2 IDはnull（`PENDING_B_FORMULA_INTERFACE`） |
| κなしの明示layout | 未公開（`FULL_H_RTE`はκ必須） | 物理requestは完成。V2 IDはnull（`PENDING_B_KAPPA_FREE_EXPLICIT_LAYOUT_INTERFACE`） |
| physical candidate hook | 未公開 | SOURCE付きの外部対応表（`identity_map`）を保持 |

`bind()`は、Bが対応した時点で同じ物理scheduleを`B.generate`へ渡し、全roundの照合（T、q、実幅、signed weights、scalar一回）を経てV2 IDを対応表へ入れる。2nd/m5_bestではすでにこの経路で動作しており、照合も成立している。

## 2. 標準Yoshida4 Full PFの接続状況

- **係数と順序**：
  - `h46_m5_plan.weights('4th')`を、そのbinary64値と順序のまま使う。値は(a, −u/(2−u), a)、u=2^{1/3}、a=1/(2−u)。
  - 中央のpieceは負時間で、各pieceの時間は`T·w_j/q`。
  - pinned SOURCEのtextから実行した値と一致する。既存`product_formula.yoshida_4th_list`（s3odr4）とも同じ数値である（あちらは中央が先の表現）。
- **m5_bestとの区別**：
  - formula_idは`YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64`。m5_bestは`PF_M5_BEST_20D64C2_BINARY64_V1`（11 piece）で、別のIDになる。
  - 物理候補ID（h46の`cid`）も全roundで異なる。
- **Full Hの保持**：
  - one-body＋全DF fragment（H4 rank10、H6 rank19）を、保存snapshot順で保持する。
  - B0/prefix7とtruncated scopeは拒否する。
- **共通の物理条件**：scalarは全T一回、symmetric-directional control、測定付きfull Hadamard wrapper、P0。κ・K・layoutはなし。
- **既存の経路**：作用は`h46_long_time_signal`（`FORMULAS`に`4th`を含む）、価格は`h46_long_time_native.compose`（`core.weights('4th')`）。保存grid（T∈{0.8,1.2,1.6,2.0}）に一致するroundは、carrier確認の対象として表示する。
- **B側の残課題**：B.generateでのblockerは公式の検査だけである。testで、公式をm5_bestに替えた同じspecがB.generateを通ることを確認した。

## 3. Full-H RTE明示layoutの接続状況

- **preparation**：`prepare_full_h`（`9b83b96b`）の意味を保持する。
  - one-bodyもrandomized、決定論block 0、全DF fragmentを保存順でrandomized。
  - identity scalarを分離して一回適用、全H λ（H4 21.0279、H6 41.2279。転記）。
  - PRのtail-only λ（2.8630）は拒否する。B3も拒否する。
- **roundごとのlayout**：次を保持する。
  - T、q、R_outer、event_count=qR、grouping `S2_OUTER_Q_TIMES_R`、`allocation=[R]`
  - K（偶数、paired finite Taylor）、`event_tau=λ(T/q)/R`、outer stepの幅T/q
  - scalar一回、component law（SOURCE付き）、compose SOURCE
- **κの扱い**：spec・mappingともnull（`kappa_mode=NOT_DEFINED_EXPLICIT_LAYOUT`）。架空のκを入れていない。`paper_kappa_optimality_claim=false`、`priced_equivalence_claim=false`。
- **groupingの区別**：同じevent数でもq/Rの分け方が違えば別の物理ID（`FULL_H_`＋cid）になる。
  - 公開layout（H4 q16/R16/K4、T=0.8）は`FULL_H_8d6f23210529d68452dd`となり、light studyの候補と一致した。
  - layoutを与えないroundは物理候補がnullで、`LAYOUT_NOT_SUPPLIED`とする。
- **B側の残課題**：blockerはκだけである。test専用のprobe κを与えると、同じspecとDのlayout mappingがB.generateを通り、mappingの`physical_layout`がそのまま保持されることを確認した。probe κはD出力には含まれない。

## 4. A/C requestとphysical candidateの対応

`requests()`は、全roundでA向け1件とC向けX/Y 2件を出す。これらは同じ値を参照する：
- `physical_candidate_id`、`physical_round`
- `v2_configuration_id`／`v2_candidate_id`（未対応時は両方null）
- identity、law_binding、binding、T、qPF

C側には、profile（compiler、symmetric_directional、full wrapper、P0）と測定軸・符号・単位も入る。`identity_map`の行は`{m, physical_candidate, V2_configuration_id, V2_candidate_id, source}`で、Bの`IDENTITY_MAP`と同じ形にした。

## 5. 残る科学的不足

| 方式 | A（N/RMSE） | C（価格） |
|---|---|---|
| Full PF（2nd/4th/m5_best） | 全H作用のmeanと、保存外Tのreference。PF公式・幅・各roundのRMSE誤差契約（現policyはPR用）。全roundの十分N | 同じ(公式, T, q)の決定論full-wrapper価格。保存価格は公式・T・q・FULL prefix・profileが一致する場合だけ。未取得の回路はdomainを確認したうえで取得 |
| Full-H RTE（明示layout） | 全H preparationでのfinite-K meanとnormalization。sampler lawの資格。全roundのdecoder/RMSE。十分N | 同じlayoutのcontrolled full-wrapper cohort価格。有限標本と全law平均の区別。保存のlight候補（n=9）はlayout・K・T・preparation hash・lawが一致する場合だけ |

公式・幅・M、Full-Hの実layout・K、比較menuは採択していない。標準4thが技術的に接続されたことは、正式なbaselineとしての採択を意味しない。

## 6. 技術テスト・保全・STOP

- **[TESTS](TESTS.txt)：96 PASS**。
  - D 16件は指示§7の13項目と、B照合の改ざん検出・欠測null・DF/Pauli・SOURCE存在の確認。
  - 主要guard（κ拒否、8次拒否、layoutのround範囲、B照合）は、無効化すると検出されることを確認した。
- 標準4thの係数は、package（pyscfを読み込む）をimportせず、pinned textだけで照合した。
- 保全：B/A/Cのbranch・controller、既存8次SOURCE・結果、SOURCE/input/raw/result/grant/ledger/marker/KEEP/STOPは変更していない。
- **STOP**：commit/pushとremote照合の後、**mandatory STOP**。
