# SP-1：別one-shot実行authorization receipt

2026-10-06 JST。最終source review PASSと利用者の別明示実行指示を受領し、
固定SのSP-1を一回だけ実行する。全結果でmandatory STOP、retry0、次stage未認可。

## 固定source・参照元

- source S：`0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`。
- review R：`d0502f22039dc0b5de227ce4b7b5a229c55f1e74`、判定PASS_FOR_SEPARATE_SP1_AUTHORIZATION。
- 準備P：`c2a49f72eaffba4514743689eb8ccee02ca060a8`。P/Rを実行HEADに使用しない。
- contract SHA256：`f95931dbde2cfbba23c7585e851a26e3f3217d5820343f88b155b9a3a45093d1`。
- [固定Sの契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)。
- [Pのreview受領記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2a49f72eaffba4514743689eb8ccee02ca060a8/docs/tracks/algorithm_codesign/sp1_execution_authorization_receipt.md)。

独立branch/worktreeは`track-b-sp1-one-shot-execution-20261006`、
`/home/abe/Project/prt-worktrees/track-b-sp1-one-shot-execution-20261006`。
Sから作成し、変更はauthorization JSONと本receiptの二pathだけ。
準備Pの必要二pathだけを固定git blobで参照し、identityを以下へ記録する。
旧uncommitted資料の一括copyをしない。source/contract/tool/test/threshold/capsを変更しない。

## 実際に受領した明示実行指示

利用者の今回の指示。JSONのexplicit_execution_instructionを原文UTF-8照合対象とし、
末尾の二space・改行も保持する。表示用JSON文字列は次の通り。

```json
"source `0d01ed9a332ebc5b66ed08acf56214a9b9c0236d` の固定契約でSP-1を一回だけ実行し、終了後はmandatory STOPしてください。  \n"
```

原文UTF-8 SHA256：`446f9ca1ef802d9a7a36847e813602f9a18054e4ab843b954950b8c8dbddd33b`。
APPROVED_FOR_ONE_SP1_RUN、science_execution_authorized=true、source_commit=S、
runs=1、retries=0、mandatory_STOP=trueへ固定する。
A自身のSHAは自己参照で埋めず、実行時HEADからresult/markerへ記録する。

## 参照blob identity

- P authorization JSON SHA256：`999159503f070014684b87a123b21dd045d5bb57a925ec62bef3f21775dce75b`。
- P receipt SHA256：`8354374fd05e3abb919b12776d8a5542305cedcbcf82c5052c34b81793ae13a2`。
- 元SP-0.5 result SHA256：`d92c3a3d618219f3f46c3cc8143a49b085772e3d1943036511bf39c792dfb759`。
- tool identity SHA256：`803d902a18d925a9565749fbc64422dca4979bfa6c0bd8b438b28de18d3aa984`。
- 固定source manifest SHA256：`2c691732e2b0d802bb1b10137a0c6dca5a09da8ede97234d964e4123f7ffdd3f`。

## 実行scopeとSTOP

2-qubit synthetic A/B/C、n={8,16,32,64}、NONE/D/R/DR、
12 wrappers/48 mask rows/96 axes/16 outer paths。一つのexact π/4 catalogue。
保存primitive列を使用し、新synthesis0。Cはtoy coinでありfinite RTEではない。
common pre-placement fusionとadditive synthesized-primitive T costを保持する。
complex ε=.05、familywise α=.05、α_a=1/1920、5% NONE materiality、
Bernstein sufficient shots、1e9 shots/axisを結果前固定のまま使う。
wall1200 s/CPU900 s/RSS1GiB/output8MiB/1 process/retry0。

clean HEAD=A、親[S]、二pathだけの差分、source/contract/runtime identity、
fresh output/markerを確認して、固定runner runを一回だけ呼ぶ。
exclusive markerは測定前に消費し、partial/numeric/cap/runtime failureでも削除・retryしない。
登録条件の拡張、旧16-cell/BF/BM再開、分子NPZ/DF/Hamiltonian、trajectory、GPU、
whole-wrapper compilation、次stage自動実行へ進まない。

実行終了後は保存値のidentity/inventory/classification predicatesとprovenanceだけを監査する。
resource/coefficients/信号を再評価しない。必要最小限の結果・監査・GPT資料をcommit/pushする。
全outcomeでmandatory STOP。研究Bの方針、RQ、新規性、論文着地点、追加検証の必要性・範囲はGPTへ戻す。
