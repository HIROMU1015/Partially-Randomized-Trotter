# BF-A後review受領・BM-0設計packet

2026-10-05 JST。`BM0_DESIGN_PREPARED_REVIEW_REQUIRED`。
**GPT返却reviewを文書へ反映した。新science実行は0、BM-1は未認可、mandatory STOP。**
B-F現行仮説を限定negativeとして閉じる判断はGPT reviewに由来する。
B-Mは構成可能性を調べる候補で、new method・論文RQの最終採択ではない。

## 1. Reviewする順序

| 文書 | Review対象 |
|---|---|
| [返却計画書の原文snapshot](inputs/track_b_post_bfa_research_redesign_20261005.md) | GPTの方針・RQ・段階／§18のCodex作業範囲 |
| [B-F closure](bf_current_hypothesis_closure_20261005.md) | 限定negative、原INCONCLUSIVE／R0 BF-A、再実行なし |
| [native列・数学仕様](bm0_native_sequence_and_error_spec_v1.md) | m1/flat、作用順、internal/cut、finite mean、semantic obligations |
| [DF情報・claim比較](bm0_df_information_and_prior_art_v1.md) | 一次本文との重複、具体的評価案、同情報baseline、MISSING量 |
| [小型pilot提案](bm1_small_model_pilot_proposal_v1.md) | 4 control families、有限domain、cost/metric案と未固定事項 |
| [design status manifest](../../../artifacts/track_b_bm0_design/2026-10-05/design_status_v1.json) | 文書identity・入力hash・停止状態。科学結果台帳ではない |

## 2. 入力provenanceと作業場所

branch `track-b-bm0-design-20261005`、独立worktree
`/home/abe/Project/prt-worktrees/track-b-bm0-design-20261005`。
baseは公開handoff `0da4d18acf3f5d32d1bc32c9661b667885bcf5f2`。
Aのworktree、rootの別branch、既存uncommitted変更を編集・stageしない。

選択して保存した未commit入力は次の**2件だけ**。raw bytesを変更せずcopyした。

| 入力 | 元の場所／identity | 公開snapshot |
|---|---|---|
| 詳細計画書 | root `track_b_post_bfa_research_redesign_20261005.md`、branch `all-r-coherent-opt2-reoptimization`、読取時HEAD `e098c54c78f589055082f9cfc2b13de50c90ca94`、untracked、38,970 bytes、SHA256 `8bd9a740892a5d2640c7b94765c5605026f8670b92ee8972fa5375ba0b25ef53` | `inputs/track_b_post_bfa_research_redesign_20261005.md` |
| 利用者の貼付review | `/home/abe/.codex/attachments/c7628a04-7398-4ace-95ae-f9e62e28a0ae/貼り付けたテキスト.txt`、29,623 bytes、SHA256 `d45cd7900f7f324869262b74a0be7d5c9785b5328e66b2827447107e12c376d5` | [raw review](inputs/post_bfa_gpt_review_20261005.txt) |

root HEADは作業場所のidentityであり、未commit本文を含むcommitとは表示しない。
raw reviewの`chatgpt-content-reference`／`sandbox:`は元UIの参照記録で、GitHubから解決可能な根拠ではない。
本文の一次出典は本packetのclaim比較表を使う。過去handoffの入力snapshotは書き換えない。

## 3. 変更しない科学identity・A参照

| 対象 | 固定commit／入口 |
|---|---|
| B原science source | `e59344a564e70d64dc3ea39d640581c72676df31` |
| B元authorization-only child | `cc971e4a2bff9b0c5708003fde7b9519eed27241` |
| B原中断result | `09c9e89555032213b52a6a60b38f55563d07a34d` |
| B R0 result | [6d2645a result validation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d2645a09440f50e5b869ef42a1b73a1b625a1af/docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md) |
| Aのreview指定参照 | [4c23453 claim/evidence map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c23453c541700c6a41ba71fc5ec9323b53858d6/docs/research/track_a_post_pm2_claim_evidence_map.md) |
| 過去STOP・構造結果の固定入口 | [0da4d18 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0da4d18acf3f5d32d1bc32c9661b667885bcf5f2/docs/tracks/algorithm_codesign/research_redesign_handoff_20261005.md) |

Aの指定commitを固定参照する。このB checkoutに残るAのM2 snapshotを並行Aの最新正本としない。
H4 1.00 Åはdevelopment、1.30 Åは既開封。Bの新held-out／independent validationとしない。
既存source、authorization、markers、artifact、fingerprint、status、pathは変更しない。

## 4. 技術上の確認と未確定の差

- tail-centered native列の内部leading項は`K_A/(4m^2)`。説明用full-interval式の`K_A/m^2`と区別した。
- nested m1は一般にflat S2と異なる。両方のnative/fusion後countを比較する仕様にした。
- finite mean、normalization、control relative phase、sector義務を分離した。BF用testをBM検証済みとしない。
- coalescing・三層hybrid・一般BCH・DFでの誤差評価は既知。ordered-word DF案も独立methodとはまだ認定できない。
- general compact BCHに同じDF backendを許す対照を入れた。共通化cacheだけなら差は工学／適用に限定され得る。
- 4種類のmodel recipeは提案。行列を生成・採点しておらず、有限時間の改善／悪化は未観測。

BM-0の具体化は「このnative列を何の情報で選び、何を対照にするか」まで。
独立した技術差がないままB-M new-method claimへ進むことは提案しない。
新定理の完成を小型pilotの唯一の条件とすることも提案しない。

## 5. GPTへ戻す判断

1. このB-M構成候補と小型pilotに情報価値があるか。同情報compact法との一致ならapplicationへ限定するか。
2. BM-1をleading heuristic＋oracle評価に限定するか、finite-T certificateを要求するか。
3. model recipe、時間／精度、固定tail、人工cost proxyの採否。比較範囲とmetricのclaim限界。
4. selector・判定threshold・numerical guard・資源上限をどこまで結果前固定するか。
5. near-integrable／SPRINT列の対応をこのpilotで求めるか、比較外としてBM-1のclaimを限定するか。

これらは研究の必要性・範囲の判断でありGPT側。通過した場合に限り、Codexで新source・限定semantic tests・
preregistration／manifest／authorizationを準備する。返却reviewやこのpacketだけでBM-1を実行しない。

## 6. 今回の操作範囲

textの読取、関連一次本文の閲覧、文書編集、明示した2入力のraw copy、文書provenance監査だけ。
science run／Hamiltonian・state・target生成／NPZ resolve-stat-hash-load／signal／trajectory／
circuit-build-compile／GPU query-use／test suiteは全て0。
保存済みscience resultの再集計・再採点・再分類も0。global validation manifestと既存resultは変更しない。
BM専用design manifestは文書inventoryであり、新しい科学evidenceではない。
AGENTS.mdの継続的な公開指示に沿い、この独立feature branchの必要資料だけをoriginへcommit・pushする。
公開後の完全SHAと固定packet URLを利用者へ報告し、mandatory STOPする。
