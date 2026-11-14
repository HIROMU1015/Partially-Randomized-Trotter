# C向けHANDOFF：PR-DF LD1 m3の8行S/C

m3 (T.8/q8/R16、physical `045843b2429bece6f29b`) の元normal sampler fresh4 pathを取得した。**予定8／再利用0／新規compile8／有効8／保留・欠測0、科学retry0**。Cによる受入・残差集計・fitは未実施。今回の終了点はこの入力の公開と引渡しである。[REPORT](REPORT.md)に条件・gate・実測資源・未実施項目を示す。

- 作業branch：`codex/d2-pr-df-ld1-m3-acquisition-20261115`（元P2から独立worktree）。Pauli holdout/STOPとDのm9環境は変更していない。
- 実行SOURCE：**`2f423befbba5075bad4ede3057f89edd788d2661`**、元P1/P2実装・結果：`39ddd59edf54a3a901e5302207b2765ddbd03fbb`。pipeline本体はbyte一致、m3専用callerと段階観測だけを追加した。
- 元結果公開commit：**`4031dbab5871d8b51513dae5e97b7a4e45704071`**。`run01/RESULT.json`、`p0..p3.json`、`receipts/p0..p3/`の元result/STRUCTURAL/cosine/sine等。元実行SOURCEをこの公開SHAへ付け替えない。
- 取得前固定：実行SOURCEに含まれる[INPUTS](INPUTS.json)、[ACQUISITION_PLAN](ACQUISITION_PLAN.md)。C計画8eef・C索引c99のfixed locatorを保持する。

## 8行と読取入口

cohortは `D_PR_DF_LD1_M3_CAL_NORMAL_20261115_V1`。seed表selectorは実行commit `2f423befbba5075bad4ede3057f89edd788d2661` の `INPUTS.json`、全axis receiptのcommitは `4031dbab5871d8b51513dae5e97b7a4e45704071`・selector=[]。

| path | axis | S (構造RZ) | C (compiled RZ) | seed表selector | 元axis receipt |
|---|---|---|---|---|---|
| 0 | X | 36370 | 37973 | `cells[0].seeds[0]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p0/cosine.json` |
| 0 | Y | 36370 | 37973 | `cells[0].seeds[0]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p0/sine.json` |
| 1 | X | 33795 | 35455 | `cells[0].seeds[1]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p1/cosine.json` |
| 1 | Y | 33795 | 35455 | `cells[0].seeds[1]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p1/sine.json` |
| 2 | X | 35333 | 37267 | `cells[0].seeds[2]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p2/cosine.json` |
| 2 | Y | 35333 | 37267 | `cells[0].seeds[2]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p2/sine.json` |
| 3 | X | 34116 | 35834 | `cells[0].seeds[3]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p3/cosine.json` |
| 3 | Y | 34116 | 35834 | `cells[0].seeds[3]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p3/sine.json` |

[機械可読C_HANDOFF](C_HANDOFF.json)の`rows`8項目は、physical/元canonical ID、cohort/index/seeds/axis、整数S/C、event digest、wrapper fingerprint、元request/law、原path/axis/structural receipt、gateの固定locatorを持つ。元native receipt bytesのhashとprivateの所在も保持した。[PATHS.csv](PATHS.csv)にも8行を収録している。X/Yは別receiptだが同一pathの別軸で、独立8標本としない。

```python
# 保存済みJSONを読むだけ。sampler/compiler/fitは呼ばない。
import json
from pathlib import Path
handoff = json.loads(Path(
    'docs/research/d_pr_df_ld1_m3_results_20261115/C_HANDOFF.json').read_text())
for row in handoff['rows']:
    # row['S'], row['C'] は同じpath/axis/event digestの一組。
    # row['axis_receipt'] 等のcommit/path/selectorで元JSONへたどれる。
    assert row['m'] == 3 and row['physical_id'] == '045843b2429bece6f29b'
```

## Cに引き継ぐ条件

H4 linear1Å/STO-3G/tol-only DF10、LD1/S2/Δ.1/K2、元exactκ/H/state/snapshot/preparation・fragment/basis順、新ZZ、元normal sampler Q、scalar c+s_R総T一回、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、元control/P0/測定付きfull Hadamard wrapperを維持した。正規化Bを価格へ掛けていない。

D2では入力/preparation/scalar/fragment、seed、signed allocation/time、同event digest、軸/整数S/C/compiler/fingerprint、wrapper測定とmutationを確認した。**full-wrapper statevector・新しい数値作用gateは元P2 freshと同じ未実施(false)**。元P1人工parity/unitaryとP2 replayは元SOURCEの証拠であり、今回それらを再実行していない。今回8行についてCの科学的受入やモデル資格を先取りしない。

`saved_P2_fresh_input_references`はm0/m6各4 pathの元C索引・原receipt・元SOURCE/cohort/seedを示す。P1/replayは参照のままでpoolしない。今回保存値の再compile・新平均・残差集計は0。C fitは別指示の範囲で、式は軸別 `saved E_I[S] + θ0 + θq q + θL qR`。IとQの差、小標本のばらつき、未観測rareは未評価。4 pathから全law E[C]/CIを認定しない。

後続m4 q16/R32/T1.6、physical `3df0cfd3356c358aeda8` のfresh4 path/cohort/index/seedは `future_holdout` にm3取得前固定済み。今回m4取得/開封0。別指示でC fit・m4予測凍結後、さらに別指示でm4を取得し、軸別4 path平均Oと凍結PのP−O/絶対/相対差をexact報告する。O=0の相対はnull、欠測時にn4変更なし、モデル数値許容差なし・PASS/FAILなし。規則本文はACQUISITION_PLANを参照する。

## 終了点とロードマップ

m3入力待ちは解消した。次は別指示のC残差集計・fit・係数/m4予測凍結であり、今回そこへ進まない。Dのm4はさらに後続の別指示。ロードマップ本体は統合担当へ委ねる。全law E[C]・CI・N/G/winnerは未認定。旧raw/receipt/STOP、Pauli STOPとm9環境を保全し、commit/push・remote照合後に**mandatory STOP**。
