# SP-1：最終source review PASS受領・別authorization準備

2026-10-06 JST。利用者が返却したGPT reviewを受領し、
**PASS_FOR_SEPARATE_SP1_AUTHORIZATION**を記録する。
source reviewは通過したが、今回の返答は実行authorizationそのものではない。
登録SP-1 resource/signal mapは未実行、science markerは未作成。

## 固定identityと作業分離

- science source S：`0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`。
- 最終review対象のreview-only commit R：`d0502f22039dc0b5de227ce4b7b5a229c55f1e74`。
- [固定Sの結果前契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)。
- [Rの最終review依頼](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0502f22039dc0b5de227ce4b7b5a229c55f1e74/docs/tracks/algorithm_codesign/sp1_final_source_review_request_0d01ed9.md)。
- contract SHA256：`f95931dbde2cfbba23c7585e851a26e3f3217d5820343f88b155b9a3a45093d1`。
- [source manifest](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)：
  SHA256 `2c691732e2b0d802bb1b10137a0c6dca5a09da8ede97234d964e4123f7ffdd3f`。
- [別authorization準備JSON](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/authorization.json)。

独立branch/worktreeは `track-b-sp1-authorization-preparation-20261006`、
`/home/abe/Project/prt-worktrees/track-b-sp1-authorization-preparation-20261006`。
RではなくSからsparse worktreeを作り、Sの直接子Pとして承認待ちの準備記録を固定する。
Pの変更はauthorization JSONと本receiptの二pathだけ。Pは実行用承認commitではない。
A/root worktree、共有API、source/contract/tool/test、SP-0.5 result/marker、過去STOPを変更しない。
Sの研究概要・source manifestのpending記述は結果前履歴として保持し、今回のPASSは本receipt/JSONを読む。
二path制限を守るため、不変sourceの索引・manifestを更新しない。

## 返却reviewの効力

利用者の返答から、次の文をそのまま記録する。

> 判定は
> **`PASS_FOR_SEPARATE_SP1_AUTHORIZATION`**
> でよいと考えます。

> これはSP-1の実行authorizationそのものではなく、**source reviewを通過させ、別authorization-only childを作って一回実行してよい段階**という意味です。

reviewはmask非依存fusion、登録16 pathのfusion全0、additive synthesized-primitive metric、
共通moment/bias/range/Bernstein/shot-cap/NONE/5%会計、59 focused testsを確認した。
研究方針の全面再設計をせず、SP-1 one-shot後にGPTが再評価する順序を採用する。

今回のJSONは `SOURCE_REVIEW_PASSED_AWAITING_EXPLICIT_EXECUTION_INSTRUCTION`、
source_commit=S、science_execution_authorized=false、explicit_execution_instruction=null。
正式な実行指示を捏造せず、APPROVED_FOR_ONE_SP1_RUNにも変更しない。
launch guardはこの準備を拒否する。run mode・科学評価・marker作成は呼ばない。

## 固定scopeを維持する

| 項目 | 固定値／境界 |
|---|---|
| domain | 2-qubit synthetic A/B/C、n={8,16,32,64}、NONE/D/R/DR、12 wrappers/48 rows/96 axes |
| fusion | logical→joint lowering→role/mask非依存exact fusion→placement→PAI、登録16 pathはcandidate/actual/cross-role全0 |
| catalogue/input | exact kπ/4一つ、SP-0.5固定保存列・guard・係数区間。新synthesis0、native error10^-6 |
| primary | fusion-normalized additive synthesized-primitive T cost。cross-primitive reducer/compiled wrapper claimなし |
| confidence | complex ε=.05、familywise α=.05、α_a=1/1920、common Bernstein sufficient count |
| materiality | 同wrapper NONE比5%。A/B duplicateは独立positiveへ数えない。Cがplacement主要証拠 |
| limits | wall1200 s / CPU900 s / RSS1GiB / output8MiB / one process / shot cap1e9/axis / retry0 |
| 解釈 | Cはtoy outer coinで、finite RTEではない。D/R populationのconfoundingを保持 |

追加target/catalogue/precision/grid、分子NPZ、Hamiltonian/DF、trajectory、GPU、
旧16-cell/BF/BM再開、次stage自動実行へ進まない。
source-bound静的planは13保存keysの参照を列挙しただけで、science scoreや新targetではない。
不変sourceの59 testsは既存local passとして保持し、full suiteやscience testsを追加実行しない。

## 明示実行指示を受けた後だけ行う手順

1. 同じSから**別の実行用branch/worktree**を作る。P/Rを親にしたgrandchildから実行しない。
2. 本準備のJSON/receipt二pathだけを参照元commitとSHAを記録して引き継ぎ、
   実際に受領した明示指示を正確に保存する。
3. JSONをAPPROVED_FOR_ONE_SP1_RUN、science_execution_authorized=true、source_commit=S、
   runs1/retries0/mandatory_STOP=trueとしてSの直接子authorization-only Aへcommitする。
4. clean HEAD=A、parents=[S]、二pathだけの差分、source/contract/tool identity、
   fresh markerとcapsを照合し、固定runnerを一回だけ実行する。
5. 完了・partial failure・timeout・memory/output cap・numeric failureのいずれでもretry0でSTOP。
   48 rows/96 axes、全path/sequence/guard/moment/bias/shots/G/ratio/分類とprovenance・資源を保存する。
   必要最小限の結果・照合・GPT review資料をcommit/pushし、研究判断をGPTへ戻す。

Pは準備の記録として保存し、amend/reset/force-pushしない。
現在まだ受領していない実行指示の例：

```text
source 0d01ed9a332ebc5b66ed08acf56214a9b9c0236d の固定契約でSP-1を一回だけ実行し、終了後はmandatory STOPしてください。
```

## SP-1後にGPTへ返す事項

利用者reviewの分岐は研究判断の候補として記録し、runnerの自動GOへ変換しない。

| 保存結果のpattern | GPTが再評価する事項 |
|---|---|
| A/Bの短いnでも利益消失し、CのD/R/DRも不利 | synthesis-placement主線closure／研究B再設計 |
| Cのselective DまたはRがgain、DRが不利 | selective placementの情報価値 |
| CのD/R/DRが広く有利 | cheap probabilistic synthesisの支配／既知法との強い比較 |
| nでgain/loss反転 | accumulation boundaryをactual DF/RTEへ接続する必要性 |
| 多数のbias/shot-cap不適格 | 適用域制約と方針縮小 |
| technical INCONCLUSIVE | gridを増やさず原因review |

accumulation/crossover自体はPAI/Sparse PS/TE-PAI等で既知。
positiveでも新規性成立、actual compiled cost優位、DF-native improvement、一般D/R lawは主張しない。
次stageの必要性・範囲、RQ・新規性・論文着地点はGPT側へ戻す。
現在はこの準備記録をGitHubへ公開してSTOPし、別の明示実行指示を待つ。
