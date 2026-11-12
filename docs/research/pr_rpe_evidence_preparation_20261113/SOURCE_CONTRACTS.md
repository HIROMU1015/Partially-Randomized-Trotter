# Full PF・Full-H RTE：物理SOURCEからV2 requestへの対応

前回の[4方式SOURCE表](../pr_rpe_method_coverage_20261113/SOURCE_MAP.md)を再利用し、今回必要なformula・物理単位・bridgeの不足を追記する。SOURCEの静的調査であり、係数・norm・作用・価格を再評価していない。以下のB基点は`6a9d7fd36b687e58aecbed1278c6d7927d6827d6`。

## 強いFull PF baseline

| 利用可能なSOURCE | 何が既存か | V2での利用資格と不足 |
|---|---|---|
| `src/trotterlib/h46_m5_plan.py:weights/schedule` | S2、標準`4th` Yoshida4、`4th(m5_best)`のsigned composition。全Hはone-body＋全DF fragmentを保持 | V2 generate/structureはS2とm5_bestだけ対応。標準4thをm5へaliasしない。旧planのH4 rank12を現DF10へ移さない。元tol-only SOURCE/recipeと照合する |
| `scripts/h46_long_time_signal.py:candidates/context/evolve_batch` | H4 DF10/H6 DF19の保存snapshot、Full/rank全prefix、q/T/公式に依存する作用、PF/discard biasとfull-H reference。既存の長時間候補もここから辿れる | 実q/T・signed時間・数値state・fragment order・scalarを一致させる。保存作用の存在は新RMSE Nを意味しない。再解析はAの科学scope |
| `scripts/pf_common_run_native.py:cost_one`、`scripts/h46_long_time_native.py:compose/cost_trial`、`src/trottertracks/resource_applicability/ax2b_native_df_v5.py` | deterministic native evolution、symmetric-directional control、完成Hadamard cosine/sine/measure、P0、元Qiskit1.3/opt1/seed17 profileでの実compiled価格 | `df_deterministic_step_rz_cost`等のD-only/uncontrolled費用で置換しない。時間固定の旧cost_oneを任意Tでそのまま動かすこともしない。Cが新requestと元recipe/compiled evidenceを照合する |
| `src/trotterlib/product_formula.py:_get_w_list`、`pf_decomposition.py:iter_pf_steps` | `2nd`、`4th`、`4th(new_2/new_3)`、`8th(Morales)`、`8th(Yoshida)`等の既存係数・展開経路 | ラベルの存在だけで全H controlled-wrapper/RMSEへの接続資格はない。binary64係数、負時間、merged/unmerged順序、SOURCE・適用窓を区別。今回は新しい公式を実装・採択しない |
| `src/trotterlib/df_partial_randomized_pf.py:build_df_hd_trotter_blocks/df_deterministic_step_rz_cost`、`partial_randomized_pf.py:deterministic_step_cost`、`scripts/profile_h5_8th_cgs.py`等 | 旧DF/Pauliの高次PF展開、state-action・CGS/phase-fit・局所cost/profile経路 | full-wrapper平均Cや全RPE RMSE Nではない。古いDF rank・state・ground-cache/fit結果は現H4/H6へ無条件移送しない。新fit/再計算は今回0 |
| `scripts/pauli_h2_h6_study.py:initial/price_job`、`pauli_signal_adapter.py` | Pauli Fullは全非identity項＋constant、B0はprefix、q-step S2、central-control full wrapper。保存H4の専用B0経路も既存 | DFと同じ数値H/stateと仮定しない。表現ごとのSOURCE/sector/state/prior/decoder/price policyを先に固定する |

Full PFはκ/Kなし。formula・要求PF幅・実幅・q・元H・保持H・sourceをidentityへ結び付ける。H4 B0/p7はtruncatedのまま、元Hとの差を含む全体RMSE契約をAが評価する。全H Full PFとの比較を省略してPR優位を採択しない。どの公式と幅、truncatedのどの範囲を探索するかはユーザーの未確定事項。

保存成果への具体的な入口は、B基点の`artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/source_catalog.json`と`paper_manifest.json`、`track_a_h46_long_time/2026-10-25/signal_provenance.json`・探索/確認cost summary。これらは元candidate/cohort/科学SOURCE・保存carrierへ戻る索引として使い、既存順位やGを今回再解析しない。現13round Nの出所は`ac3e1ce87b5ed7d9114abc316545366dd097209e:docs/research/rpe_provider_integration_20261111/fixed_coverage/COVERAGE.json`に集約されている。H4の補正整数原本は`89c91c692e1d35de94d2e7afbe890aabe6d80b7e:docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json`。H4はB0、H6はFULLで、いずれも旧HP型N。新RMSEで再利用候補になるのは対応する保存mean/reference/SOURCEであり、旧Nそのものではない。private raw/archiveしかないcarrierは出典から回収し、未回収を理由に自動再計算しない。

## DF Full-H RTE：既存物理経路

固定実装：[`9b83b96b83f62142c9d87858dc24a94db7ac5d61:scripts/h46_full_h_pure_rte.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9b83b96b83f62142c9d87858dc24a94db7ac5d61/scripts/h46_full_h_pure_rte.py)。後続索引：[`7a8bc7f132e166390c943e6b32c92cc6abe1baf4:SOURCE_INDEX.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json)。providerはc31ef424175c02379eafebe8fc6c7548c1d9aabe、後続kernelは同索引記載の8ca723320。

| 物理要素 | 元SOURCEの意味 | V2 requestの必須対応 / 担当 |
|---|---|---|
| 元Hとstate | `context/restore`が保存H/state/snapshot/sector/referenceを使用 | binding/sector/decomposition/source/数値stateを一致。geometry/basisだけで同一視しない。A |
| 非scalar全H | `prepare_full_h`はDF tailに一体項固有basisのZ成分を追加し、deterministic blocksを空にする | one-bodyを含むpreparation/component列・basis registryとhash・順序を固定。prefix0だけでは不足。A |
| scalar | 一体項のidentity成分と元tail/constantの扱いを元SOURCEで保持し、元loopのtotal Tで一回適用 | scalar/preparation/phaseの意味を照合。scalarを二重加算したり全非scalar RTEと混同しない。A/C |
| λ・finite-K・law | λは一体項を含む全randomized成分の係数l1、paired finite Taylor、source-bound component/order law | B3のtail-only λを再利用不可。K、正負時間、ideal meanと実samplerのlaw/距離を分離。A（N/mean/normalization）、C（price law） |
| event単位 | 旧S2ではq groups×R_outerのs位置。Taylor orderは一位置内のproductを含み、primitive回数とは別 | 新BはqPF/formula/R_outerなしのevent_count request。旧q/R grouping・tail境界・scalar policyを失わずにsource-bound mappingへ記録。A/Cのbridge確認が必要 |
| normalization・N | `h46_long_time_signal.evolve_batch`がfinite distribution norm、log B、raw/corrected meanを評価。`qualify`は旧signal精度/shot policy | これをRMSE十分Nへ変換しない。新κ、finite-K/signed条件、state/prior/decoder/全round契約とNはA。Bはその証拠を受け入れる |
| full-wrapper C | `cost_job→h46_long_time_native.cost_trial/compose`が上記prepで制御付きfull-wrapperを一体compileし、XY RZ gate countを取得 | 旧価格はそのcandidate/law/compiler/cohortの有限標本。Cが新domain・期待値区分・不確かさを照合。全law平均資格は別証拠 |

**重要な非同値**：後続`scripts/h46_full_h_t20_opt.py:evaluate`には、同じs=qR/Kのmean/normalizationを共有する処理があるが、`priced_equivalence_claim=false`を明示している。BはqR一致だけで旧Cを新candidateへ移さない。新plannerは供給された`physical_schedule_mapping`全体を残す。グループ構造とsourceが異なればV2 IDも異なり、人工testで区別した。

`SOURCE_INDEX`には旧B3の依存ファイルも含まれるが、全H化は専用`prepare_full_h`による。**DF B3の一体項保持をFull-H RTEへrelabelしない。**

## Pauli Full-H RTE

B基点の`src/trotterlib/pauli_signal_adapter.py`、`h4_pauli_pr_search.py:law_descriptor/central_trajectory`、`h4_pauli_pr_mean.py`、`scripts/pauli_h2_h6_study.py:price_job`を再利用候補とする。固定event拡張結果`10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`、元科学SOURCE`e755ecf581600192fd2745d7066e410cebd933d3`は前回SOURCE表から辿れる。

L_D=0で全非identity Pauli項をfinite RTEへ入れ、constantは保持する。DFの一体項ブロックという構造とは異なり、元Pauli term/componentの全列とHamiltonian hash、saved raw stateを固定する。law descriptorにはcomponent/orderの実効53-bit grid、angles/scalar operands、B/normalization、target semanticsがあり、`C_EFFECTIVE_GRID_LAW_EXACT_RETAINED_PRODUCTS_NATIVE_STEP_SCALAR`とideal lawを無条件に同一視しない。価格はcentral-control full wrapperで一体compileした各軸RZ。既存Nはε_signalの旧policyで、全round energy RMSEではない。

## 今回Bが行わないbridge

物理producer、λ/normalization/mean/N、signed law、係数の新評価、元scalarの再構成、S再計数・compile・fitは実装・実行しない。A/Cがsource-boundのper-round物理requestと全体error/price契約を供給した後、Bの既存V2 metadata mapper/identityと供給record形式へ接続する。特に8次PFと旧grouped純RTEを「ラベル又は総event数が近い」だけで対応付けるadapterは追加しない。これらは新たな科学的判断が必要な境界である。
