# B/C HANDOFF：H4 K0 PASS・K1四値

[REPORT](REPORT.md)／[K0](K0.json)／[K1](K1.json)／[CHECKS](CHECKS.json)／[STOP](STOP.json)。実行SOURCEは`19b95d8b3c88063d2dbae4e5bb47455b52b00903`。C既存branch `codex/r-variable-compiled-price-design-20261111`を継続し、Bのmodule/worktreeを変更していない。

## Bへ渡す対象と資格

```text
quantity = FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT
expectation = IDEAL_FINITE_K_SOURCE_LAW
K0 = PASS
mean_compiled_C = null
qualified_for_compiled_price = false
full_rpe_resources = null
```

**FULL_WRAPPER_MEAN_COMPILED_PRICEとして供給しない。** BのC providerへ値ありrecordを自動登録していない。必要なのは今回の構造状態・平均Sの診断metadataへの固定参照であり、新しい共通API/価格schemaは追加していない。

| m / axis | physical ID | E[S] |
|---|---|---:|
| 0 / cosine(X) | `53e00d8b07d1547d19a7` | 1743.6682557187337 |
| 0 / sine(Y) | 同上 | 1743.6682557187337 |
| 6 / cosine(X) | `2a8c6718220eec59d419` | 1764659.275564426 |
| 6 / sine(Y) | 同上 | 1764659.275564426 |

各値の数値区間・全identity/law/sourceはK1.jsonをそのまま参照する。X/Yの同値は今回SOURCEの結果だけであり、compiled Cの軸一致を意味しない。

## 固定identity / SOURCE

対象は同H4 DF10/S2/L_D1/δ.1/K2/h.1/M12/P0、元symmetric-directional control/測定付きfull Hadamard wrapper、元H/state/snapshot・fragment/basis/qubit順・scalar一回。exactκは前回固定PRICE_REQUESTSの文字列を維持し、近似から生成しない。

- preparation hash `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、324非identity/9basis。
- m0 T=.1/q1/R2、m6 T=6.4/q64/R128。2lawだけ。
- configuration `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。
- B正本は`030f3f548a35d98a88e62e92a35f132e0243a293:docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json`、SHA256 `22a1bd1af8704006568c545eb8bcbce5bd476202ad59aaeaf79f50850815f943`。
- B `f0c4bb55`受入索引と、指定最新`3216d27a69ac8532d2a22e78f5f94491d202d238`の同索引はbyte一致。前回のC重点requestと正式B identity/lawを完全照合した。旧ID付替えなし。

## kernelの再利用と今回の証拠

kernel配列はKERNELS.npz、SHA256 `85784d00c31c851a22b5ab6d4e51ef84b43b4ab7424fc338cb6402e7e101c07d`。各wire/RのP,g、方向丸め区間Plo/Phi/glo/ghi、close/reverse rewardのterminal配列を保持する。状態順・entry indices・first/steady prefix・axis suffixはKERNEL_STATES.json。126配列・18状態辞書で、行列や状態を別順へ並べ替えない。

同じstateはactive basis/phase flag/正確なpending-Z hexを保全する。system wire0〜6は56state、wire7は48、ancillaは22（両law）。wire間独立性を仮定せず、reward和を期待値の線形性で集約する。outerの固定出口を元Accountingで確認したうえで合成し、counterの途中reset/finishはしない。

元native branch3,978件、event metadata3,240件、phase/control1,155件と有限局所reward/出口/terminal完全一致、全18件のlongdouble確率検査、最大mass defect約3.02e−14（固定tolerance1e−10）をK0.jsonへ保存した。K0 PASSとkernel保存後だけK1を実行した。K0.jsonのK1_started=falseは開始前snapshot、完了はK1.json/STOP.jsonを正本とする。

数値区間は固定NUMERICAL_CONTRACTのIEEE/BLAS前提での構造・確率丸めenclosure。実sampler/ideal law距離、compiler C−S、rare寄与/全law価格精度、共有covariance、total CI・全RPE保証は未資格。既存18技術testの記録と今回metadata1再確認を再利用し、実parityはK0として別に記録した。

## 保全と次の判断

run01のUNKNOWN/STOP/log、元C1/C2/connected-cluster/S0a/source/input/raw/cohort/grant/ledger/marker/KEEP/STOPは不変。今回run02のSOURCE改変・scientific retryなし。sampling/full-path S/build/compile/transpile/compiled C/fit/N/RMSE/G、他round/H6/8次/直接8wrapper pilotはいずれも0。

不足するのは同条件のE[C−S]とその不確かさ・採用精度。C1固定R係数の移植や、今回Sのcompiled価格への昇格はしない。新しい直接価格を取る場合は保存C/S一致を先に確認し、sampling/compile・cohort/seed・目的・標本数・校正/holdout・STOPを有限scopeとして具体化する。旧n2/最大8wrapper案はdevelopment候補で、全law精度の保証ではない。

次の担当：**ユーザー**が残差/compiled価格検証の有限scope・必要精度を判断、**Codex C**が指定scopeの保存再利用/実行計画を準備し明示許可内だけ実施、**Codex B**はS診断metadataだけを参照しcompiled C/Gは欠測に保つ。公開・remote SHA/tree/主要成果物照合後mandatory STOP。
