# Codex A向け：Full PFの保存signal・referenceとN/RMSEに必要な追加情報

[REPORT](REPORT.md)／[COVERAGE.csv](COVERAGE.csv)（`signal_*`、`reference_*`、`independent_pair`、`numerical_endpoints`の列）。

## 1. 再利用できる情報（技術的な再利用候補。科学的な資格は未確定）

| 情報 | 所在 | 内容 |
|---|---|---|
| 保存Full-H signal（134件 EXACT） | `155529e2:…/track_a_h46_long_time/2026-10-25/signal_records.json.gz`（key=物理ID）、extensionは`extension_signal_records.json.gz` | raw、corrected、exact_tail、bias、log_B、allowance（最大中間norm・vector closure）。作成関数は`h46_long_time_signal.evolve_batch`（SOURCE `e417ffc9`、file hashは`signal_provenance.json`） |
| T=0.8の再利用signal（24件） | 同carrierの`origin`＝pf_common結果（`0b1886f5`／`90f1cd55`） | bindingは照合済み。長時間研究は代表2件で新旧adapterの一致（〜1e-14）を確認している（`new_adapter_compatibility.json`） |
| H6 fixed-deltaの再利用signal（2件、CONDITIONAL） | `signal_summary_v1.json`（B基点） | raw signalとpropagation evidenceがない（`saved_allowance_only`） |
| reference（136件） | signal recordの`signal.reference` | 元DF H（one-body＋全fragment＋constant）を保存sector stateで評価した値。T=0.8はreceipt付きの保存reference file。H6 T=1.6はfixed-delta。その他はcontext内のspectral full-H |
| 独立reference pair（T=2.0の38件） | private `{H4,H6}_context.json`の`reference_expm_multiply_check` | `expm_multiply`との差：H4 3.20e-15、H6 1.45e-15。他のTには独立pairがない |
| 数値gate | `signal_endpoints.json.gz`（候補ごとに5つのε） | 旧契約（経験的allowance、α_axis .025、u floor 1e-9）のELIGIBLE/INELIGIBLE。**RMSEのPASSとして使わない** |
| 物理候補 | COVERAGEの`physical_candidate` | `cid(system, FULL, formula, rank, T, q)`。Dの`bind()`の物理IDと同じなので、`for_packet()`で結合できる |
| state・target | `scientific_provenance.json`のbinding | 保存DF Hの同sector基底をtargetとする、従来の数値state条件付きの評価 |

## 2. 追加で必要な情報（新しい科学評価、または判断）

1. 保存外Tのmeanとreference：保存はT∈{0.8,1.2,1.6,2.0}だけ。RMSE scheduleのT=h·2^m（例：h=0.1）では、m3・m4以外の全roundが新規になる。
2. Full PF用の全RPE RMSE誤差契約：PF bias（元H targetに対する）、統計誤差、decoder/aliasの合成。現在のV2 policyはPR用。
3. 全roundの十分N（RMSE型）。保存endpointのshot数は旧型のN。
4. prior・decoder・alias/branchの検証（全round）。
5. 独立reference pairが必要なら、そのT（現在はT=2.0だけ）。保存値がないことを理由に自動計算はしない。

新しい計算を提案するときは、保存signalと完全に一致するセル（formula、T、q、FULL prefix、binding）を除外し、残りのセルと上限を明示すること。
