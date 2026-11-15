# 統合担当・次担当向け：PR-DF LD7 共通V0完了

**LD7は共通V0で適合。次の判断は「LD7固有価格の較正と候補別配分の比較」。** configuration1件、信号13 round、Q3一回を完了し、sampling/回路build/compile/fit/価格予測/独自配分は0件。公開・remote一致確認後mandatory STOPとする。

| 分割 | 共通V0のMSE lower..upper (Ha²) | 判定 |
|---|---|---|
| LD1（保存再利用） | 4.72462588934653801951643E-7 .. 4.72462588934653960729534E-7 | 適合 |
| LD2（保存再利用） | 4.72409731948051054908399E-7 .. 4.72409731948051194818804E-7 | 適合 |
| LD7（新規1 call） | 4.28414160318294721301353E-7 .. 4.28414160318294721365383E-7 | 適合 |

exact bounds・元receipt・同一shotsの照合は[SUMMARY.json](SUMMARY.json)、[REUSED_V0.json](REUSED_V0.json)、[CHECKS.json](CHECKS.json)。X/Y各 `[64,60,56,52,47,43,38,34,30,25,21,17,12]`、元exact閾値τ `15136964828156312001/5960464477539062500000000 Ha²`。ideal finite-RTE raw IID model MSEの条件付き結果であり、価格順位の判定ではない。

## 次担当が再利用する入口

全40桁locatorは[SOURCE_LOCATORS.json](SOURCE_LOCATORS.json)。Git参照は `{commit, path, selector}` とし、公開commitと実行codeのcommitを混同しない。

| 内容 | 固定公開 | path（このdirectoryから） |
|---|---|---|
| 採択元計画 | f89fdc9d4ee1f82dbaf1748608b1461f456d5057 | `../d_pr_df_next_split_choice_20261116/NEXT_FINITE_PLAN.json` |
| LD7 configuration・mapper gate・scalar | 12efc9dca | `generation/CONFIGURATION.json`, `SCHEDULE_GATE.json`, `SCALAR.json` |
| 13信号のraw | 3ce3e5dde | `signals/m0.json`〜`m12.json`, `signals/RESULT.json` |
| 全13信号のB bundle | 1cbc24cfc | `B_MEANS_LD7.json` |
| Q3観測前固定入力 | dcc9c93a1 | `Q3_INPUTS.json` |
| Q3結果・API・閉包・DP・資源 | 8ebf3c982 | `q3/LD7_V0/RESULT.json`ほか |
| LD1/LD2保存V0 | 8ec7c882806809460d38a638a891bafcf18fafd1 | `../d_pr_df_ld_split_q3_resume_20261116/SUMMARY.json` menu[0]/[3] |

configuration IDは `ce9c7f08234089f0b50fd786c20403fe02c5de9073e89d6d35a12ee9d966c5e1`。保存H4 linear1Å/STO-3G/tol-only DF10、元H/state/snapshot/fragment順、one-body別head、DF0..6 head / DF7..9 tail、S2・Δ=.1・K2・共通κ。全13 R=1、単一pieceを元mapperで確認済み。tail s_Rとcの別SOURCE、total scalar一回適用の証拠はSCALARと各rawのscalar_onceを参照。

生成/信号caller実行SOURCEは `04e37b3f7`、信号数値SOURCEは変更なしの `fefd548ecd56f60d72031b3d385f6f5e6ee95d06`。Q3 caller実行SOURCEは `1cbc24cfc`。Q3は既存3.11.0rc1環境・元worker・元kernel/accelerator AST guardをそのまま使った。期待hashの差替えは0。正常終了、科学retry0。資源は[REPORT](REPORT.md)と元RESOURCE_FINALを参照。

## 後続の有限判断（今回は実行していない）

f89計画の残りを進めるなら、LD7自身の測定付きfull controlled Hadamard wrapper、CONTROLLED_ZZ_CX_CRZ_CX_V1、P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17で価格を取得する。LD1/LD2係数を移さない。

| 既存計画上の役割 | m・q/R・physical ID | path / X,Y compile上限 |
|---|---|---|
| 較正 | m0・1/1・`2b4a68084b0c88723f93` | normal fresh4 / 8 |
| 較正 | m6・64/1・`e7194af048772d7160c6` | normal fresh4 / 8 |
| 価格の独立記述的holdout | m3・8/1・`a6a83aa244eddd8b0956` | normal fresh2 / 4 |

これは後続価格の座標を既存計画へ結んだ参照であり、seed生成・path取得・compileの開始ではない。cohort/seed・実prefix7取得入口・receipt公開経路は後続実行前に固定し、計画所定の人工確認を行う。全R1ゆえqR=qであり、計画の価格式は軸別 `alpha + beta*q`（offset0、full-wrapper価格）。3特徴のqとqRを別々にfitしない。m7..12はq64外挿、m3は補間域の確認にとどまる。

依存順序は**較正receipt公開 → LD7軸別価格モデル・26参照値固定 → 独立m3価格holdout → LD7自身のA_SHAPE/B_PRICE配分と最大2 Q3 → 保存LD1/LD2との有限モデル費用比較**。今回のLD7 V0は再実行しない。費用の意味と数値許容差未採択を引き継ぎ、再fit・追加path・欠測補完を自動で行わない。

§10.5では今回、計画公開後のremote成果・関連job・出力競合、元SOURCE変更、生成した13物理IDと旧archive、LD7/V0の同一候補・同一配分を照合した。同等結果未同定、曖昧な対象なし。PR-Pauli LD3、決定論p7の並行jobは対象外・非介入。後続開始時には今回公開後の価格receipt/model/Q3/jobの差分だけを再確認し、受入可能な保存物を優先する。

ロードマップ差分：LD7を「descriptor-ready・未評価」から「現行configuration/全13信号・共通V0 Q3適合」へ進められる。次の不足はLD7自身の価格と候補別配分の達成モデル費用。LD3/LD5/LD9は未評価のまま。LD2長round価格の既知の制約は解消していない。ロードマップ本体は統合担当が更新する。

厳密E[C]・CI・実PCG64への移送・正式G/winner・L_D最適性/単調性は未認定。旧SOURCE/raw/STOP、LD1凍結価格とm4独立holdoutを保全した。SOURCE view選択の科学call前停止も[technical_preflight](technical_preflight/NOTE.md)へ残した。今回の適合から、未実行の価格段階の成功を推定しない。
