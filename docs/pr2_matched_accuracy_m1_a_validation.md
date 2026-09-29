# PR-2 matched-accuracy M1-A validation

実行日：2026-09-30
status：`SELECTION_LIMITED`
result fingerprint：`422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e`

## 結論

H4 linear 1.00 Å、STO-3G、DF rank 12、sector 8 qubitの保存済みdevelopment snapshotだけで、
compile-free M1-Aを完了した。208 base候補に結果前規則で2件のr64 boundary候補を加えた210候補を
評価し、random候補194件は全件accuracy適格だった。

16-cell selectorは64件のproxy frontierから16件を選んだが、52件の未選択proxy非支配候補が残った。
固定hard barrierは`unselected_proxy_nondominated_candidates`を理由に`SELECTION_LIMITED`を返した。
したがってwinner、held-out候補、M1-B direct compileを確定せず、compile job 0で停止する。

## 入力と認可

- series：`pr2-rebaseline-de7a5492-v1`
- development：H4 linear 1.00 Å、STO-3G、DF rank 12、sector 8 qubit
- development raw SHA-256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`
- v1.1 authorization SHA-256：`846231a09f5b8e69aa55b78e7045eaae7a562198d01f2b229585f0b0336a9c41`
- result file SHA-256：`1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086`
- `T=0.8`、`q={1,2,4,8}`、`delta=T/q`、random `r={1,2,4,8,16,32}`、`K={2,4}`
- boundary規則による追加：B3 rank 0 q8 r64 K2、B2 rank 3 q1 r64 K2

v1初回実行は固定KとRTEConfig toleranceの実装不整合でresult作成前に停止した。v1.1はcandidate、K、
signal polynomial、normalization、accuracy threshold、selectorを変えず、固定Kの一step residualを受理する
self-consistent toleranceだけを修正してから実行した。

## signal・accuracy集計

| method | 評価 | accuracy適格 | 最小analytic shot | 最大complex bias | 最大normalization |
|---|---:|---:|---:|---:|---:|
| B0 discard | 12 | 8 | 14,073 | 0.1234869502 | 1 |
| B1 deterministic rank 12 | 4 | 4 | 14,073 | 0.007267057821 | 1 |
| B2 partial | 145 | 145 | 14,073 | 0.007280308945 | 1.215666659 |
| B3 random-dominant | 49 | 49 | 22,520 | 0.007470319990 | 13,227.75851 |

全210候補のうち206件がaccuracy適格で、非適格4件はB0の`nonpositive_axis_allowance`である。この段階の
`W_action`と`W_tail`はcompile前action proxyであり、compiled costまたは最終total costではない。

## selectorとhard barrier

- accuracy適格random候補：194
- proxy frontier：64
- compile cap：16
- 選択：16
- 未選択proxy frontier：52
- 未選択boundary：0
- tier内訳：split anchor 4、q anchor 4、boundary 2、tail challenger 4、frontier round-robin 2
- barrier：`SELECTION_LIMITED`
- next action：`STOP_AND_REVIEW_COMPILE_BUDGET_OR_TECHNICAL_NOTE_SCOPE`

この結果からpartial、deterministic、random-dominantのwinnerは主張できない。compile capを増やすか、
technical noteへ縮小するかは別の研究方針判断であり、このauthorizationでは追加計算しない。

## correctness・resource audit

- Rayleigh residual：`1.2680e-15`
- dense block reconstruction relative error：`1.6167e-15`
- 最大tail reconstruction relative error：`4.6115e-13`
- 最大direct/log normalization差：`1.8190e-12`
- 最大raw/corrected再構成誤差：`1.0167e-14`
- wall time：`62.2415 s`
- peak RSS：`1,289,084 KiB`
- development hash check/load：1/1
- signal evaluation：210
- molecular calculation、held-out path/stat/hash/load/signal/cost/ranking：0
- circuit build、compile、trajectory sample、full wrapper、quantum shot：0
- GPU query/allocation/kernel：0/0/0
- `S3_authorized=false`、automatic next stageなし

実行前後のfocused suiteは各26 passed。専用result validatorと凍結JSON schemaの両方を通過した。これは
local evidenceであり、immutable CIまたは独立外部再現ではない。
