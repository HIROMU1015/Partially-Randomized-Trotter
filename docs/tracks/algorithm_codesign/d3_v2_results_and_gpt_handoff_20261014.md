# D3 v2：一回実行結果・保存監査・GPT引き継ぎ

2026-10-14 JST。最終分類は **D3_GUARANTEE_PRICE_MAP_COMPLETE_AWAITING_GPT_REVIEW**。
36新規合成、48行、四ケースの資料を保存した。外側exit0、retry0、mandatory STOP。
重要な科学的解釈・研究継続・algorithm採択は行っていない。

## 1. 凍結source・実行承認・入力

Repository：`HIROMU1015/Partially-Randomized-Trotter`。
実行branch：`track-b-d3-v2-one-shot-execution-20261014`。

- science source S2：`dceccaa774229527a46a1386131583a4f618f8ad`。
- authorization-only直接子A2／実行開始HEAD：`daa2569208a945f8e41b3c4d0de715c1e11b104b`。
- [承認receipt](d3_v2_execution_authorization_receipt.md)：利用者の「次の作業に進んで大丈夫です」を原文で記録し、直前の準備完了報告が示したD3 v2一回実行へ限定。
- [固定contract v2](../../../artifacts/track_b_d3_source/2026-10-14/v2/execution_contract_v2.json)、SHA256 `d017559c4dddee510be2caa40097592e93d0ae40303572508f8f6889842313cf`。
- [authorization](../../../artifacts/track_b_d3_source/2026-10-14/v2/authorization.json)、SHA256 `b15c1e36f9e712998bf2eb211f5f92d45a4cbf2fb7cc8608f4120404a7b0ae59`。
- [one-shot marker](../../../artifacts/track_b_d3_result/2026-10-14/v2/one_shot_consumed.json)、SHA256 `e55db65c7bae156503e09ff61309f5fc1fcaed160451782281c7ca1b5b3e2271`。

公開確認HEAD `560205d4d017b7f1503903841156bd7bf9dcde6c` はA2のparentとして使用していない。
A2の変更はauthorization JSONとreceiptだけ。結果公開commitはA2の後の証拠commitであり実行HEADとは区別する。
固定sourceと型adapterの検証範囲は[準備報告](d3_v2_precision_repair_and_source_preparation_20261014.md)を参照。
旧v1のtechnical inconclusive、旧承認、消費済みmarker、結果・STOPは変更していない。

対象は既知developmentの明示**3-system-qubit synthetic involution**：
`p=(1/5,3/10,1/2)`、`tau=5/7`、`sigma=+1`、初期状態`000`。
`Q0=Z0`、`V1=R_XX01(pi/4)`、`V2=R_XX12(pi/4)R_ZZ01(pi/4)`、`Qi=Vi†ZiVi`、`R=sum p_i Qi`。
同じfinite `P5(-i tau R/r)^r`、`r=1,2`、ordinary、partial_return_tail、closed_P3_tail、full_return、closed_P5_full、matched_CTS。
分子geometry、軌道basis、DF rank、deterministic split L_Dはこのexplicit domainには適用しない。
独立held-out validation、全体compilerによる新回路評価、分子PR総資源評価ではない。

## 2. 結果前固定された四ケース

| ケース | Taylor step保証 | 合成価格 |
| --- | --- | --- |
| G0_P0 | `3 |x|^6/720` | 旧D1 strict1e-6の保存価格 |
| G1_P0 | Hermitian contractionの`|x|^6/720` | 同じ旧価格 |
| G0_P1 | 旧保証 | 全36 keysを一回ずつfresh strict1e-7合成した価格 |
| G1_P1 | 改善保証 | 同じ新価格 |

**全ケースのnative誤差会計は旧1e-6に固定**した。新1e-7の精度をshot forecast削減へcreditしていない。
同じguaranteeのP0/P1でbudget/Nが一致することを保存値監査で確認した。
旧law、dyadic weight、production moment/range、保存native IR、D2のliteral境界相殺額を再利用。
219,303 pairs、provider/signal matrix、回路生成/compileは再実行していない。

axis accuracyは1/200、alpha_axisは49/24000、resource failureは1/12000/row。
四つの12-row ledgerはそれぞれfamilywise0.05の**代替resource forecast**である。
96 axes全体の一つの同時confidence0.05 campaignや量子shot実行とはしない。
state-uniform保証は採択済みHermitian contractionの前提と誤差会計に基づく。

## 3. 保存済み比較の直接表示

以下はpreregistered comparison JSONの`h0_ranking`の先頭をそのまま表示したもの。
`h=0`は準備・読出し費用を0と置くモデル上の断面であり、実装でその費用が0と測定されたという意味ではない。
新しい最適化、追加条件、性能GO thresholdは導入していない。

| ケース | 全六方式×r1,2のT forecast最小（h=0） | primary partial/P5×r1,2のT forecast最小（h=0） |
| --- | --- | --- |
| G0_P0 | r2 closed_P5_full | r2 closed_P5_full |
| G1_P0 | r1 closed_P5_full | r1 closed_P5_full |
| G0_P1 | r2 closed_P3_tail | r1 closed_P5_full |
| G1_P1 | r1 closed_P3_tail | r1 closed_P5_full |

これらは今回の固定provider・合成器・seed・nativeモデルの保存forecastである。
P5の一般的優位、P3の新規algorithmとしての採択、最良precision/rの普遍性は主張しない。
保証と価格が方式/r選択へ与える研究上の意味はGPTの科学レビューに返す。

全48行のexact native T/CX/1Q切片、prep/readout slope、N、bias、remaining、moment/range、acceptance、workspaceは
[exact rational CSV](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/resource_rows_exact_v2.csv)から取得できる。
小さい[表示用48行CSV](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/resource_rows_display_v2.csv)は10桁の丸め表示で、順位・認証の正本はexact CSV/JSONである。
全hの最小領域、同r対照、primaryの等しいr自由度、quantum Paretoは元comparison JSONに保持した。
[比較の直接projection](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/saved_comparison_projection_v2.json)は同じ保存fieldを索引化したもの。
未知classical online費用は0とせず、`NOT_ACQUIRED`を保持する。
Tはnative1Qに含まれるためT+1Qを合成scalarへ重複加算しない。
formalな独立h-domainとphysical条件`h_1Q >= h_T`を区別する。

## 4. 一次データの入口と取得性

- [resource map](../../../artifacts/track_b_d3_result/2026-10-14/v2/resource_map.json)：48行の全数値・予算・来歴。
- [G0_P0](../../../artifacts/track_b_d3_result/2026-10-14/v2/comparisons/G0_P0.json)、[G1_P0](../../../artifacts/track_b_d3_result/2026-10-14/v2/comparisons/G1_P0.json)、[G0_P1](../../../artifacts/track_b_d3_result/2026-10-14/v2/comparisons/G0_P1.json)、[G1_P1](../../../artifacts/track_b_d3_result/2026-10-14/v2/comparisons/G1_P1.json)。
- [factorial effects](../../../artifacts/track_b_d3_result/2026-10-14/v2/factorial_effects.json)：保証、価格、interactionのexact切片・slope差。
- [36新sequence cache](../../../artifacts/track_b_d3_result/2026-10-14/v2/synthesis_cache_new.json)：各keyのsequence、SHA、T/Tdagger/1Q/W count、strict error upper。
- [inventory](../../../artifacts/track_b_d3_result/2026-10-14/v2/frozen_inventory.json)、[全key process receipts](../../../artifacts/track_b_d3_result/2026-10-14/v2/synthesis_process_receipts.json)。`jobs/key_000`〜`key_035`にjob・started・token・result・receipt・raw logsを保持。
- [writer evidence manifest](../../../artifacts/track_b_d3_result/2026-10-14/v2/evidence_manifest.json)：original JSONとlossless chunksのbytes/SHAを結ぶ。

resource mapは**531,617 bytes**、SHA256
`a4bc655cdeda9a5a438f3034a70dce0d0b7bf647029c115b4d3696292f60a43a`。
GitHub connectorのsize制限に触れる場合は、順序どおり以下を結合する：

1. [part0000](../../../artifacts/track_b_d3_result/2026-10-14/v2/resource_map.json.part0000.txt)：524,288 bytes。
2. [part0001](../../../artifacts/track_b_d3_result/2026-10-14/v2/resource_map.json.part0001.txt)：7,329 bytes。

両partはUTF-8で読み取れ、結合後に元bytes・SHA256と完全一致することを確認した。
各rowは`rows/<ledger>/row_00.json`〜`row_11.json`から個別に取得できる。
原JSON・原chunks・source・markerを再生成・書換えしていない。

## 5. 外側processと資源使用

[最終terminal](../../../artifacts/track_b_d3_result/2026-10-14/v2/terminal.json)、
[inner terminal](../../../artifacts/track_b_d3_result/2026-10-14/v2/inner_terminal.json)、
[外側receipt](../../../artifacts/track_b_d3_result/2026-10-14/v2/outer_process_receipt.json)、
[STOP](../../../artifacts/track_b_d3_result/2026-10-14/v2/STOP.json)を原記録のまま保存。
内外分類は一致、outer exit0・signalなし・resource failureなし。
新jobs36、予約token36、completed sequence36、個別process receipt36が揃う。
token自体は依然として合成完了claimではなく、complete resultとexitを別に照合した。

| 記録 | 値 |
| --- | --- |
| outer wall | 215.15832338575274秒 |
| outer/childrenを含むCPU receipt | 257.727476667秒 |
| wait4 child peak RSS | 165,056 KiB、161.1875 MiB |
| observed aggregate peak RSS | 381,415,424 bytes、363.74609375 MiB |
| observed single-process peak RSS | 160.3125 MiB |
| 36 key workersのCPU合計 | 17.061447秒 |
| 最大per-key CPU | 0.490907秒 |
| 監査開始前raw出力 | 323ファイル、2,774,028 bytes |

worker CPUにはruntime検証・保存・guardも含む。純backend時間やonline sampler費用と扱わない。
固定two workers/thread1、CPU1800秒/20秒key、RSS512 MiB/process・1536 MiB aggregate、output16 MiB等の上限内で終了した。
wall capsはnullのまま。独立したspeedup benchmarkはない。
actual provider/signal matrix0、circuit build/compile0、quantum shots0、sampling/trajectory/LP/DF/分子/GPU0。
今回許可されたstrict2×2 synthesis guardsは36回であり、全matrix処理0とは呼ばない。

## 6. 保存監査と保護

[saved-only audit](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/saved_output_audit_v2.json)：
**2579 checks PASS、FAIL0**。
[stdlib監査script](../../../scripts/tracks/algorithm_codesign/audit_d3_v2_saved_outputs.py)は保存bytes・有理数scalarだけを読む。
backend、matrix、science module、pair kernel、budget optimizerを呼ばず、再ランキング・再最適化をしていない。

確認した範囲：source current472 hashes、protected437 files、historical source3 blobs、raw323 files、
writerの58 original entriesとchunks、S2→A2直接子、全job/sequence/cache/process receiptの一致、
strict saved bound、旧G0/P0 anchor、48行のsaved scalar積、同GでP0/P1 budget一致、factorial差、
外側/内側statusとhash links、caps、retry0、STOP。
[raw snapshot](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/raw_output_snapshot_v2.json)、
[prelaunch metadata](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/prelaunch_metadata_verification_v2.json)、
[post-STOP runtime/親worktree保護](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/post_stop_runtime_and_worktree_protection_v2.json)を保存した。

runtime executable/package treeはprelaunchから不変。
旧v1 execution worktreeとv2準備worktreeはclean、親50 dirty files・index・HEAD・statusは不変。
旧D1/D2/source/contract/authorization/markers/STOP、v1失敗記録を保持した。
この監査はlocalの保存値整合確認であり、独立provider・compiler・CIによる外部再現ではない。

## 7. GPT側へ戻す科学的論点と停止

今回の固定された四ケースで、保証変更と価格変更を分離した結果が取得できた。
GPTのレビューでは、preregistered primary pairと全六方式を区別し、次を判断する材料が揃っている：

- 誤差保証によるr選択の変化と、合成価格による表現選択の変化をどこまで説明できるか。
- 全六方式とprimary partial/P5の差、固定r対照、未知prep/readout・classical costを含む成立条件。
- 今回の一provider、一precision追加、一seed、加法native/literal境界モデルの一般化限界と論文claim。
- 追加検証・独立実装・研究継続の必要性と範囲。

Codexは科学的採否、general superiority、新algorithm採択、次のprecision/seed/r/provider探索を判断していない。
GPTの重要な科学レビューは、利用者の開始承認を受けて実施する。
資料取得のための公開・remote byte確認を終えた後、**mandatory STOP。追加合成、再会計、compile、sampling、分子実行は行わない。**
