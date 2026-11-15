# C・統合担当向け：PR-DF LD9 m6直接価格比較

**m6の指定有限cohortでは、X/YともLD9平均492,573 RZ/shot、LD7保存平均393,837.25 RZ/shot。差LD9−LD7=+98,735.75、比LD9/LD7=1.25070190795817307783862。** LD9のnormal4 path/8 compileは全有効。LD7は別cohortの保存4 path/8軸receiptを再利用し、新しいLD7 compileは0。

## 固定機械可読入口

**commit：`d85150cd1b51f87da4e767b32cf52f532765d358`**

- `docs/research/d_pr_df_ld9_m6_price_20261116/C_HANDOFF.json`：LD9 configuration/physical ID・request・cohort/index/seed・実行SOURCE・軸別S/C・fingerprint・compiled receiptに接続。両分割の8行ずつと比較2行を含む。
- 同directory `COMPARISON.json`：exact平均/差/比、有限cohort資格、件数、LD9全round価格/総費用null。
- `PATH_PRICES.csv`：LD7/LD9各4 path×X/Yの計16行。Cは直接compiled RZ、SとD=C−Sは診断用。
- `SOURCE_LOCATORS.json`：全40桁commit/path。

| 軸 | LD7平均 | LD9平均 | 差LD9−LD7 | 比LD9/LD7 |
|---|---:|---:|---:|---:|
| X | 1575349/4 | 492573 | 394943/4 | 1970292/1575349 |
| Y | 1575349/4 | 492573 | 394943/4 | 1970292/1575349 |

LD7 cohort=`D_PR_DF_LD7_PRICE_CAL_20261116_V1`、LD9 cohort=`D_PR_DF_LD9_M6_DIRECT_PRICE_20261116_V1`。異なるcohortで、indexが同じでも分割間のpaired解析ではない。分割内のX/Yは同pathを共有する。平均は各軸4 path、pool・欠測時のpath削減・補完は0。

| 固定物 | commit | repo-relative path |
|---|---|---|
| 使用LD9 configuration/scalar | 460d77d8f56ef0bde41dbddaa27852dd8b05513e | docs/research/d_pr_df_ld9_v0_results_20261116/generation/CONFIGURATION.json / SCALAR.json |
| 実行/比較SOURCE | 8f0a82842481270e0581658b5294ecd59e79104f | scripts/run_pr_df_ld9_m6_price.py / scripts/compare_pr_df_ld9_m6_price.py |
| 取得前SOURCE/cohort/seed/request | e733f6beaa6471c63062ef2379f3f56cb8f2499a | docs/research/d_pr_df_ld9_m6_price_20261116/INPUTS.json |
| LD9原raw | 6392aefdad0438e619cba0c0e9e55d170eb99a49 | docs/research/d_pr_df_ld9_m6_price_20261116/acquisition/RESULT.json、m6/p*/PATH_RECEIPT.json・cosine.json・sine.json |
| LD7保存直接receipt | 5c5d87ab32d03bb107a9a6bfed3cba253aef2aa0 | docs/research/d_pr_df_ld7_price_results_20261116/calibration/acquisition/m6/p*/PATH_RECEIPT.json・cosine.json・sine.json |

LD9 configuration=`ea087a7e3b311be7efae27558347e9eb471f616b8b55f59cbef8e506dda7a833`、physical ID=`09fa97187d6eee510d0c`。LD7 physical=`e7194af048772d7160c6`。保存H4 linear1Å/STO-3G/tol-only DF10、同H/state/snapshot、元順、one-body別head、S2/Δ=.1/K2/共通κ、T6.4/q64/R1。分割固有のhead/tail/λ_R/s_R/c+s_Rは元configurationから照合した。LD9実prepはhead0..8/tail9、総scalar hex0x1.258457e75abe9p+1を確認。元scalar一回規則を維持。

価格quantityはP0の測定付きfull controlled Hadamard wrapperのcompiled RZ/shot。control=symmetric-directional、new-ZZ CONTROLLED_ZZ_CX_CRZ_CX_V1、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17をLD7から変更せず、元normal samplerの固定64 outer seed/pathを使用。seed差替え・強制event・retry0。SOURCE付き16軸receiptのmetrics/fingerprintと同一profile、4 pathの完全性を照合した。

元P2/LD7と同じ取得・実装gateの範囲で有効とした。full-wrapper statevector/numerical action gateは未実施、旧保存H4 P0数値再検査は0。C自身の独立受入・law期待値認定は未実施。原SOURCEと公開commitは別で、Cは直接有限平均を厳密E[C]として扱わない。

## ロードマップ差分とSTOP

LD9価格証拠を「未取得」から「m6のnormal4 path・8軸receipt取得、LD7との同T/q/Rの直接有限平均比較」へ進められる。このm6ではLD9単価が高かった一方、13 round全体の価格・配分・総費用順位は判断できない。今回の一点からalpha/beta等をfitせず、LD7係数も移送していない。`LD9_total_model_cost=null`を維持。

§10.5はf89/460dの候補・旧価格調査を再利用し、以後の同LD9候補/cohort・receipt/model/job・出力競合の差分を確認した。同等の現行LD9 m6結果は確認範囲で未同定、未解決0。PR-Pauli LD5や決定論Yoshida4は別方式として除外・非介入。後続実行前は今回公開以後の差分を照合する。

新規4 path/8 compile、再利用LD7 4 path/8 receipt、有効16軸、科学FAIL/UNKNOWN/欠測/保留0。新規signal/Q3/fit/別価格点/総費用/CIは0。資源はREPORTと原stage/process記録を参照。ロードマップ本体は統合担当が更新する。旧raw・SOURCE・STOP、LD7/LD9の信号・Q3・既存価格成果を保全。commit/push・remote一致確認後mandatory STOPし、追加取得へは進まない。
