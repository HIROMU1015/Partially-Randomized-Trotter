# R1：source review PASS受領・別authorization準備

2026-10-06 JST。利用者から返却されたGPT reviewの
**`PASS_FOR_SEPARATE_R1_AUTHORIZATION`**を記録する。
固定sourceの修正要求はない。別の明示one-shot実行指示は未受領のため、
science executionは未認可のまま保持する。R1 run、登録合成・資源評価は0、markerは未作成。

## 固定identityと作業分離

- source S：`d43d64a821a0249a0dfab12a2472bd3a72fdee74`。
- [固定source review依頼](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d43d64a821a0249a0dfab12a2472bd3a72fdee74/docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)。
- [固定結果前契約v2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d43d64a821a0249a0dfab12a2472bd3a72fdee74/docs/tracks/algorithm_codesign/rte_reallocation_r1_preregistration_v2.md)。
- [固定native semantics](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d43d64a821a0249a0dfab12a2472bd3a72fdee74/docs/tracks/algorithm_codesign/rte_reallocation_r1_native_semantics_v1.md)。
- contract SHA256：`e8a4201203dc8d2fbb35c7f3025079d7f0f1dce270be55afd1b17f2505717025`。
- [source manifest](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json) SHA256：`797b1fb0e31ae1c3b264303899e7f3a6a1e21613e3b76cc1717f4b40cc5783a9`。
- [authorization準備JSON](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/authorization.json)。

独立branchは`track-b-r1-authorization-preparation-20261006`、worktreeは
`/home/abe/Project/prt-worktrees/track-b-r1-authorization-preparation-20261006`。
Sから必要な文書・準備artifactだけをsparse checkoutした。
変更はcontractが許すauthorization JSONと本receiptの二pathだけ。
この直接子Pはsource review・準備の記録であり、実行用authorization Aではない。
root/Track A worktree、共通API、source/contract/tool、既存証拠とSTOPを保持する。
不変sourceの索引・source manifestにあるpending記述は当時の準備履歴として残し、
今回のreview通過は本receiptとJSONで確認する。

## 受領reviewと効力

今回の利用者返答から、判定とsource修正不要の文をそのまま記録する。

> **`PASS_FOR_SEPARATE_R1_AUTHORIZATION`**

> つまり、**研究方針をここで練り直す必要はなく、固定source `d43d64a...` のまま別authorization-only childを作り、R1 one-shotへ進む価値があります。** sourceの修正を要求するblocking issueは、確認範囲ではありません。

利用者は、Sの直接子authorization-only A、別の明示one-shot指示、R1一回、
mandatory STOPの順序を指定した。この返答を実行指示へ読み替えない。
JSONは`SOURCE_REVIEW_PASSED_AWAITING_EXPLICIT_EXECUTION_INSTRUCTION`、
`source_commit=S`、`science_execution_authorized=false`、
`explicit_execution_instruction=null`とする。runs=1/retries=0は将来の上限であり、実行済み件数ではない。
[固定launch gate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d43d64a821a0249a0dfab12a2472bd3a72fdee74/src/trottertracks/algorithm_codesign/rte_reallocation/launch.py)は
`APPROVED_FOR_ONE_R1_RUN`、science=true、明示指示、cleanなSの直接子を要求するため、
この準備状態では実行できない。runner/launch/marker処理は呼び出していない。

JSONの既存`review_instruction_snapshot_sha256`はSに収録した**以前の準備指示**のidentityであり、
今回のsource review本文のhashではない。今回のreviewは本節の引用と以下の受領要旨で記録する。

## 受領したscope・解釈

| 項目 | 今回のreviewで受領した限定 |
|---|---|
| 目的 | normalization改善とodd/native controlled実装の追加費用の交換を判別するmechanism test |
| baseline | ordinary / 同I0 PTSC-K0 / A。collected CTSはPauli controlだけのI1対照 |
| distinct-basis toy | 実際にはPauli展開可能。Pauli reductionを使わない登録native実装の比較に限定し、実問題のI0-only access優位やDF規模のclassical acquisition優位は主張しない |
| primary | resource vectorとtrade-off。B²改善だけでGOにせず、全座標のstrict dominanceも要求しない |
| secondary | 共通confidence taskによるG_T/G_CX/G_1Q。scalar winnerや研究GOへ自動変換しない |
| semantics | odd complementのextra Q・符号反転・相対phase、controlled ancilla位相、distinct-basis順序を保持 |
| numerical | up_to_phase=falseのstrict operator guard、coherent measurement biasへ2Σαδを戻す既存契約を受領 |
| tests | off-domain x=1/3で確認した既存27 focused local tests。登録x=1/8,1/4を開封したscience evidenceでもimmutable CIでもない |

## 固定契約・資源上限

| 項目 | 変更しない値 |
|---|---|
| target | P₃(-iσx(3Q₀/4+Q₁/4))、m=3 / K=2、x={1/8,1/4}、σ=±1 |
| domain | Pauli commuting / noncommuting controls、distinct-basis controlled primary。ordinary native rowsはdiagnostic |
| precision | native operator ε={10⁻³,10⁻⁴,10⁻⁶}、100 dps、strict phase-preserving synthesis |
| inventory | 42 static angles / 126 synthesis keys / 264 resource rows / 132 controlled tasks / 72 comparison groups |
| accuracy/confidence | ε_complex=1/100、ε_axis=1/200、264 axes、α_axis=1/5280、familywise α=1/20 |
| metric | shared exact native cancellation後のadditive synthesized-primitive Clifford+T counts。whole-circuit最適化済みcostやDF改善とは呼ばない |
| total caps | wall1200 s / CPU900 s / RSS512 MiB / virtual address1536 MiB / output16 MiB / one process |
| per-key caps | wall30 s / CPU20 s / sequence20000 characters |
| run policy | 一回、retry0、全outcome mandatory STOP。shot capは各axis10⁹ |

新しいtarget/precision/grid、IS/PAI最適化、分子NPZ、DF Hamiltonian、trajectory、GPU、
旧pilot再開、追加scienceは行わない。tool identityとsourceの既存hashを保持する。

## 今回の読み取り専用照合

- contract/source manifestを固定identityと照合した。
- SのGit blobからcritical 26 pathと既存証拠66 pathのSHA256を照合し、すべて一致した。
  対象はsource・文書・JSON・既存log/markerだけであり、分子NPZ操作は0。
- 既存27 focused tests PASSと静的key inventory PASSの保存記録を確認した。testsの追加実行は0。
- R1 result directoryはSのtreeにも本worktreeにも存在せず、新marker作成・登録合成・資源評価は0。

この照合はprovenance確認であり、登録scienceの新しい数値結果ではない。

## 明示one-shot指示を受けた後だけ行う手順

1. 同じSから別の実行用branch/worktreeを作る。Pから追加commitを積んで実行しない。
2. Pの二pathだけを参照元commitとidentity付きで引き継ぎ、実際に受領した実行指示を正確に保存する。
3. JSONを`APPROVED_FOR_ONE_R1_RUN`、science=true、source_commit=S、runs=1/retries=0/mandatory_STOP=trueとし、Sの直接子authorization-only Aへcommitする。
4. HEAD=A、parents=[S]、clean worktree、二path差分、source/contract/tool identityとfresh markerを照合して固定runnerを一回だけ実行する。
5. 完了・partial failure・timeout・memory cap・numeric/guard failureのすべてでretryせずmandatory STOPする。結果・監査・GPT向け必要資料だけをcommit/pushし、研究判断をGPTへ戻す。

Pは準備記録として保持し、amend/reset/force-pushしない。
次に必要な明示指示の例を示す。これは未受領の指示例であり、authorizationではない。

```text
source d43d64a821a0249a0dfab12a2472bd3a72fdee74 の固定契約でR1を一回だけ実行し、終了後はmandatory STOPしてください。
```

## R1後のGPT判断へ戻す事項

| 保存結果のpattern | 利用者reviewが示したGPT側の判断候補 |
|---|---|
| B²だけ改善、native/shot resourceは明確に悪化 | 主algorithm route縮小、restricted theorem / technical resultとしての価値 |
| ordinaryには勝つがPTSC-K0と同等 | 実用method deltaは弱く、理論差中心へ |
| distinct-basis controlledで有用なPareto点 | Track B主候補への昇格を検討 |
| PauliでCTSが強く、distinct-basisではAに意味 | algebraic accessとrepresentation選択の関係を検討 |
| 全contextで実装costに負ける | A実用route STOP／方針再設計 |
| technical/guard failure | retryなしで原因監査 |

これらはrunner分類・contract閾値の変更ではなく、結果後にGPTが判断する候補である。
positiveでもmethod採択、世界初の新規性、DF改善、別geometry・pilot・next stageは自動認可しない。
現在は本準備記録を公開して停止し、別の明示実行指示を待つ。
