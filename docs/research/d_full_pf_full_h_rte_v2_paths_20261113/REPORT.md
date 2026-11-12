# Full PF・DF Full-H RTEの物理評価経路とB V2接続（Codex D、2026-11-13）

> 後続（2026-11-13）：標準Yoshida4・κなし明示layout・physical対応表を[d_full_baseline_interface_update_20261113](../d_full_baseline_interface_update_20261113/REPORT.md)で更新した。Full-H RTEのκ必須という本文の記述は、そちらで置き換えられている。

既存のFull PF・DF Full-H RTEの物理SOURCEを調査した。そのうえで、B `PR_RPE_RMSE_V2`へ渡す物理request adapterを、必要最小限で追加した。

- adapterは、V2 configurationと、同じ物理候補を指すA向けrequest（N/RMSE）・C向けrequest（full-wrapper価格）を作る。
- **新14＋B回帰80＝94技術test PASS。**
- 新しい科学計算は0件：H/state load、signal/mean/N、normalization、sampling、build/compile、価格S/C、fit、G、κ/M/幅の探索。
- PF公式、Full-Hのκ・layout、比較menuは採択していない。

成果：
- [adapter](../../../src/trottertracks/full_baseline_v2_requests.py)／[test](../../../tests/check_full_baseline_v2_requests.py)／[出力CLI](../../../scripts/report_full_baseline_v2_requests.py)
- [SOURCE_CATALOG](SOURCE_CATALOG.json)（公式・SOURCE・保存Full-H preparationの転記）
- [REQUEST_EXAMPLE_ARTIFICIAL](REQUEST_EXAMPLE_ARTIFICIAL.json)（人工κ/policyのschema例。menuではない）
- [HANDOFF](HANDOFF.md)／[TESTS](TESTS.txt)

基点はB `af284d853890155bacd19e61850c196bd1d5c660`（V2 `6a9d7fd3`を含む）。新branch `codex/d-full-pf-full-h-rte-v2-paths-20261113`で作業し、B/A/Cのbranch・worktree・moduleは変更していない。既存Track Dとは別の作業である。

## 1. Full PF：既存SOURCEとV2接続

| 公式 | 係数SOURCE（binary64） | 負時間 | DF10/DF19の作用（signal） | controlled full wrapper（native） | B V2 generate |
|---|---|---|---|---|---|
| S2 `2nd` | `h46_m5_plan.weights` (1) | なし | `h46_long_time_signal.evolve_batch` | `h46_long_time_native.compose/cost_trial` | **対応** |
| 標準Yoshida4 `4th` | 同 (a, −u/(2−u), a)、u=2^{1/3} | 中央片が負 | 同 | 同 | **未対応**。B側に公式拡張が必要。m5_bestへaliasしない |
| `4th(m5_best)` | 同 TAIL5値＋中央1−2ΣTAIL、11 signed piece、`PF_M5_BEST_20D64C2_BINARY64_V1` | 中央片が負（−0.2771） | 同 | 同 | **対応** |
| 8次Morales/Yoshida | `product_formula._get_w_list`、`pf_decomposition.iter_pf_steps` | あり | h46 DF経路なし | なし | **未対応**。DF10/19のcontrolled RPE経路が存在しない |

既存の物理的な意味：
- 各signed pieceは、one-body＋全DF fragmentを保存snapshot順に前向き半step、逆順に後ろ向き半stepと進むS2 sweepである（`evolve_batch`。nativeでは`build_deterministic_native(formula='2nd')`をpieceごとに`T·w/q`で適用）。
- 負のpiece時間はそのまま負の時間発展になる。
- scalar（`ham.constant`）は全時間Tで一回だけ適用する。
- controlはsymmetric_directional。価格は測定付きfull Hadamard wrapperのcosine/sine、P=0。

**Full Hの意味**：
- `h46_long_time_signal`の`FULL`は、prefix=全rank（H4 10、H6 19）でone-body＋全fragmentを保持する。
- H4の保存B0/prefix7はtruncatedで、adapterは全rankと異なるprefixを拒否する。
- B V2は`deterministic_scope=FULL_H`／`TRUNCATED_H`、元Hと保持H、公式、qPF、実幅`T/q`をidentityに持つ。
- Full PFにはκ・K・RTE event mappingを付けない。付ければBが拒否する。

**既存実装だけでの接続範囲**：
- B V2のPF（2nd/m5_best）configurationは、既存generateで作れる。
- ただし、native生成のPF roundには物理候補IDとの対応がない（`physical_candidate`はA fixed-grid adapter経由でしか付かない）。
- そこで`full_pf_spec`／`requests`が、h46の物理候補（`method=FULL`、prefix=rank、T、q、`cid`）とV2 candidate IDを対応付ける。
- 保存の長時間grid（T∈{0.8,1.2,1.6,2.0}、`Q`、公式）に一致するroundは`SAVED_LONG_TIME_GRID_POINT_CHECK_CARRIER`と表示する。

## 2. DF Full-H RTE：既存SOURCEとV2接続

`9b83b96b:scripts/h46_full_h_pure_rte.py:prepare_full_h`の処理：
- B3 L_D=0のpreparationを起点にする。これはone-bodyを決定論blockに残すため、そのままではpure RTEではない。
- one-body固有基底のZ成分（`−ε_j/2 Z_j`）を、DF tailの末尾に順序を変えずに追加する。
- 決定論blockを空にする。
- one-bodyのscalar（Σε_j/2）を既存tailのidentityへ合算し、`ham.constant`と一緒に全T一回で適用する。

公開値（転記。再計算していない）：

| 系 | λ_R（全H） | one-body λ | one-body成分 | 全成分 | 決定論block | 抽出identity | constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| H4 | 21.0279441 | 7.3042455 | 8 | 368 | 0 | −2.6245791 | 2.2931012 |
| H6 | 41.2279152 | 13.1792764 | 12 | 1494 | 0 | −4.9286833 | 4.6038417 |

- **event grouping**：
  - 旧経路はS2 outer q stepごとにR個のevent（`allocation=[R]`、`ABS_TIME_PER_OUTER`）を置く。
  - 各eventは`τ=λ(T/q)/R`、paired finite Taylor（偶数K）。全event位置はs=q·R。
  - 公開候補はT=0.8のH4 q16/R16/K4（s256）と、H6 q32/R32/K6（s1024）。
  - T=2.0の後続研究はs=qRでmean/normalizationを共有するが、`priced_equivalence_claim=false`である。
- **V2への接続**：
  - B V2の`FULL_H_RTE`はevent_count中心のrequestで、groupingを持たない。
  - そこで`layout_mapper`が、round別の(q, R)とgroupingの意味を`physical_schedule_mapping.physical_layout`に記録する。記録内容はS2・allocation・τの定義・K・符号・scalar一回・compose SOURCE。
  - 同じevent数でもgroupingが違えば、V2 IDも物理ID（`FULL_H_`+cid）も別になる。
  - layoutのないroundは未解決（`UNRESOLVED_KAPPA_MAPPING`）のまま残す。
- **資格の検査**（`check_full_h_preparation`）：
  - preparationの由来がprepare_full_hであること（B3を拒否）。
  - one-bodyがRANDOMIZEDで成分数≥1、決定論blockが0であること。
  - 全fragmentを保存順でrandomizedしていること。
  - λにone-body l1が含まれ、tail-onlyのλではないこと。
  - normalization policy、scalarが一回適用でidentity成分の乱択が0であること。
  - 保存H/state/snapshotとV2 bindingが一致すること。

## 3. DFとPauliの違い

| 項目 | DF Full-H RTE | Pauli Full-H RTE |
|---|---|---|
| 表現 | one-body Z成分＋DF fragment成分（固有基底回転） | 非identityの全Pauli項、constantはglobal phase |
| SOURCE | `prepare_full_h`（9b83b96b）、`rte.py` | `pauli_signal_adapter`、`h4_pauli_pr_search.law_descriptor`（L_D=0）、`pauli_h2_h6_study.price_job` |
| law | ideal finite law。保存価格はPCG64 CDF samplerのcohort | 53-bit実効grid law（`C_EFFECTIVE_GRID_LAW_…`）。ideal lawと同一視しない |
| state・sector | DF保存state（H4 sector36） | Pauli側の保存raw state・sector |
| event回路 | basis変換＋対角Pauli積 | Pauli回転列 |

adapterはDFのみを接続する。Pauliは`pauli_reference()`でSOURCEを示すだけで、V2 requestは作らない。表現が異なるbindingを混ぜると、Bの`select`が共通H/stateの検査で拒否する。

## 4. 追加したもの・しなかったもの

追加したもの（`full_baseline_v2_requests.py`）：
- `catalog`：公式・SOURCE・V2状態。
- `full_pf_spec`／`full_h_rte_spec`：比較用の基底specからbinding・h・RMSE policy・profileを引き継ぎ、方式固有のsplit・lawを付ける。
- `layout_mapper`／`generate`：既存の`B.generate`を呼ぶだけ。
- `check_binding`／`check_full_h_preparation`／`saved_full_h_preparation`：公開値の転記。
- `requests`：A/C request。
- `pauli_reference`。

Bのcontroller・planner・provider、A/Cのmoduleは変更・複製していない。物理評価器、N/C評価器、optimizerは作っていない。

## 5. 範囲の区分

| 区分 | 内容 |
|---|---|
| 既存実装だけで接続できる | Full PFの2nd/m5_best V2 configuration、Full/Truncatedの区別、Full-H RTEの事前検査（B側）、Bのevidence plannerによる不足表示 |
| D adapterで接続した | 物理候補IDとV2 IDの対応、Full-H preparationの資格検査、明示groupingのmapping、A/C requestの同一候補参照、保存資産の照合ヒント |
| B側の技術拡張が必要 | ①標準`4th`をV2 generate/structureへ追加。8次は、旧DF/Pauliの高次展開・局所costはあるが、現DF10/19のcontrolled full-wrapper経路がない。②`FULL_H_RTE`はκが必須で、明示layoutだけでは生成できない（κ不要のlayout種別を許すか、κ定義を与える）。③native生成roundへの物理候補IDの添付（現在はD requestの外部対応表で保持） |
| 新しい科学計算が必要 | 全roundのmean・reference（保存T以外）、Full-Hのfinite mean・normalization、十分N・全体RMSE、full-wrapper価格（Full PFは決定論、Full-Hはcohort）、sampler lawの資格 |
| 科学判断・承認が必要 | Full PFの公式・幅・M、Full-Hのκ定義またはlayout規則・K・M、Full PF/Full-HのRMSE error contract（現policyはPR用。B.selectでは共通policyが必須）、価格policy、比較menu |

## 6. 次の限定科学検証（未承認の提案）

**Full PFを先に進めることを推奨する。**
- 決定論なので、normalization・law・cohortの問題がない。
- 公式（2nd/m5_best）と作用・価格経路が既存で揃っている。
- 保存の長時間signalが、T∈{0.8,1.2,1.6,2.0}の一部のroundで再利用候補になる。

Full PFの最小案：
1. ユーザーが公式（2nd/m5_best。標準4thにはB拡張が必要）と幅δ、Mを選ぶ。
2. Aが、保存signal carrierとの完全一致（公式・q・T・FULL prefix）を確認し、保存外のTについてだけ決定論のmeanとreferenceを有限個計算する。
3. Aが、Full PFのRMSE error contract（PF誤差＋統計誤差）で十分Nを評価する。
4. Cが、同じ(公式, q, T)で決定論のfull-wrapper価格を取得する。保存値の流用か、既存PF provider（H6 FULLのfixed δ等）の資格確認かを選ぶ。

Full-H RTEの最小案：
1. ユーザーがκ定義またはround別layout規則と、K・Mを固定する。
2. Aが、prepare_full_hのbinding（再確認済み）のもとで、finite meanとnormalization（λは全H値、tail-onlyではない）、sampler lawとの距離、十分Nを評価する。
3. Cが、同じlayoutのcontrolled full-wrapper cohort価格を取得する。保存のT=0.8の2候補（n=9）は、そのlayoutとlawが一致した場合のfinite-cohort参照に限る。

保存値の再利用と新規計算の区別、call上限、STOP条件は、A/Cが個別の実行計画で固定する。

## 7. 検証・保全・STOP

- **[TESTS](TESTS.txt)：94 PASS**。
  - 新14件は指示§8の13項目と、pinned SOURCEの存在確認。
  - 主要guard（one-body、全rank、layout必須、DF限定）は、無効化するとtestが失敗することを確認した。
  - Bの`check_pf_pr_rpe_pipeline`は、当初sparse checkout外のfixtureで失敗した。fixtureをsparse対象へ追加した後、無変更でPASSした。
- 保存値は公開JSONからの転記で、再解析・再計算はしていない。人工test PASSは科学的保証ではない。
- 保全：SOURCE、input、raw、result、grant、ledger、marker、KEEP、STOPは変更していない。共有環境も変更していない。
- **STOP**：commit/pushとremote照合の後、**mandatory STOP**。
