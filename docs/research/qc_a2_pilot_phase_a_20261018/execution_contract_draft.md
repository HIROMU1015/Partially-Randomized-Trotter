# A2固定DF pilot：未採択科学契約草案

**`DRAFT_NOT_ADOPTED_NOT_EXECUTION_PERMISSION`。** [machine草案](execution_contract_draft.json)、[SOURCE固定](source_freeze.json)、[入力identity](candidate_input_identity.json)、[条件proposal](candidate_conditions_proposal.json)、[引継ぎ](decision_handoff.md)。本文はmachine草案の意味を示す。採択済み契約Cやauthorizationではない。

## 固定する技術要素

SOURCE S `8d483f9bfa5983bb52fef55887e2f9f73a790511` と全31 file hash。入力候補は基点34d9a2f1…にあるfull DF NPZで、container・全member/array bytes/SHA256を固定済み。実際の係数値・モデルID・証拠値は未生成。

対象候補は `cI+Q+ΣλA²` の**保存binary64を正確係数とした固定DF**、固定N、保存mode順、全signed block、固定順序の正stage palindromic S2。既存K0のexact entrywise caps、scalar shiftとL/f/Λ、determinant U/W、raw K_SOS、centered capとgrouped構造のminによるCbar2を両routeへ共有する。capや対照を結果に応じて弱めない。

数学は固定db181958…のη0/η1/minとEW2、および同じPSD capsからβ=Λ/2を用いるcentered Cbar2対照。RFは `h>0,hΛ<1/4,η<1`、対照は `h>0,hβ<1` と既知center/unwrapを保つ。最大十分stepのenclosureを比較し、真のPF限界・QPE/native資源の勝敗を出さない。厳密ground支持は理論上の条件付き入力であり、trialはU専用である。

## 採択待ちのfield

| field | 今回の状態と必要条件 |
|---|---|
| input target／mapping／N | `UNSET`。H4名目条件・8 stored modes・全12 blocks・N=4はproposal。元分子との厳密対応は保証しない |
| occupied_modes | `UNSET`。proposalはstored indices 0,1,2,3のdeterminant。HF/groundと呼ばない |
| fragment_order | `UNSET`。proposalはone_body後に保存block0〜11。データを見て並べ替えない |
| comparison_policy.d | `UNSET`、exact Rat、d>0 Ha |
| comparison_policy.alpha_spec | `UNSET`、exact Rat、0<α<1 |
| sqrt_bits | `UNSET`、正のcanonical Nat。dyadic最小上側sqrt witness |
| root_width | `UNSET`、正のexact Rat、Ha^-1 |
| max_bisections | `UNSET`、正のcanonical Nat。未達はUNRESOLVED、結果に応じた自動増加なし |
| output_bytes／log_bytes | `UNSET`、正のcanonical Nat。record/log各file上限。集合サイズは別測定 |
| canonical_repository | この専用checkoutの絶対pathをproposalとして固定。同一契約の消費済みledgerを別checkoutで迂回しない |
| adoption | false。未解決fieldを埋めた別正式契約をユーザーが採択する |
| execution | 未承認。正式採択とは別のpinned authorizationが必要 |

model_idは `DERIVE_EXACT_FROM_PINNED_RAW_AND_MAPPING` policyとする草案。raw identityとmappingを正式採択すればexact targetを一意に指定できる。値のdecodeは承認済みPhase Bだけで行い、独立checkerがmodel_idを再構成・確認して後段の親とする。既知の外部model_idが別途採択された場合は、そのdigestと一致することも要求できる。今回このpolicyをCodexが正式採択したものではない。

## 呼出し・保存・STOP

1回のtop-level adapter、SPEC-01〜03 generator、独立checker、SPEC-04〜07 generator、独立checker。後段内のSPEC-01〜03 prerequisite再検査2回もcall inventoryへ明示する。K_SOS/Cbar2を含む各段の生成/検査は同じ固定inputについてだけ。合格したgenerator flagをcheckerの代用にしない。

認可/identity/UNSETの事前gate不合格ではinput IOせずclaimなし。gate通過後はinput IO前に1回のexclusive claimを消費し、input/finite/Hermiticity/certificate等が不合格でも消費を維持する。既存claim・中断claim・別run_idは拒否。自動retry、再探索、precision/trial/rank/分子/PF順の事後変更は禁止。正常・失敗・未解像の全てでmandatory STOP。

元NPZ、exact raw、全提案/検証、partial理由、contract/auth原bytes、stdout/stderr、親ID、SOURCE/入出力hash、呼出数、CPU/wall/RSS、Rat bit長、保存量を保全する。保存故障やSIGKILL等ではbest-effortの限界を示し、消費済みclaimを削除しない。

CPU/RAM/worker/wallに人工capを設定しない。空き資源と非干渉を優先し、重いexact algebraには同値なbatch/bit処理を使う。精度・独立checker・科学call範囲を高速化のために弱めない。既存の他研究の固定resource条件は変更しない。

費用は入力/変換、共通証拠、K_SOS生成/検査、Cbar2生成/検査、step、保存を区別する。stage CPU/wallと累積process RSS、記録したRat bit長/bytesを報告する。finalization IOの未測定範囲を明示する。量子query/T/CX/depth/state-prepの未知費を0や古典時間へ換算しない。

SPEC-08、QPE、native compile、state preparation、元分子保証、研究GO/NO-GO、A2正式採択は対象外。A2未採択、A1 CLOSED、K1-F未承認を維持する。**この草案の公開ではscience authorization/grant/markerを作らない。Phase A完了後mandatory STOP。**
