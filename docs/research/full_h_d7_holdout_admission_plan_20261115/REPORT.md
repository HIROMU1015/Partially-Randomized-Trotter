# Full-H κ2 D7 holdout受入と固定shot期待費用の次段階

Cの推奨は、**26軸の母平均を個別に精密化する前に、保存凍結モデルを使う0-compileの固定shot総額診断F1を行うこと**である。これは次の有限科学実行案であり、今回実行していない。母lawの期待総額やwinnerを確定するものではない。D1/Bの追加作業待ちはない。

今回、D1の**6 receipt／6、4記述的比較／4を受入、保留0**。Bの固定1.5倍配分を**26軸／26へ接続**した。`DESCRIPTIVE_COMPLETE`、`NOT_ASSESSED_NO_ADOPTED_TOLERANCE`、`tolerance=null`を維持する。全law期待compiled価格の資格は全26軸で未取得。

## 固定SOURCEと受入経路

| 項目 | 固定参照 |
|---|---|
| D1最終公開 | [`06956c64380107392f1b24fe54b9273f0f5ae1c0`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/06956c64380107392f1b24fe54b9273f0f5ae1c0/docs/research/d_full_h_d7_holdout_20261115/HANDOFF_C.md) |
| D1実行 | `9cbabc617e95a81810924413c1b2b06ced329514:scripts/run_full_h_d7_holdout.py` |
| raw・軸別receipt | `b5fac0ee4eac83f027f883e5ea5630623a98bf29:docs/research/d_full_h_d7_holdout_20261115/run01/` |
| 元凍結2モデル・4予測 | `b5165f11bb63f9eb2a6bfe385ba75d8003b7e724:docs/research/full_h_d7_model_freeze_20261115/run01/FROZEN_MODELS.json` |
| 元C→D1引渡し | `81596ce7c561d7b75890cfe2758a0d4a12fb6b47:docs/research/full_h_d7_model_freeze_20261115/HANDOFF_D1.md` |
| 記述的規則 | `f0765067a5a8b39888e36f3a8b80f1bdbda69223:docs/research/full_h_d7_holdout_rule_proposal_20261115/PROPOSAL.md` |
| B Q3・1.5倍配分 | `6c5569c3479013b080b2ba53842585f1d670d3c9:docs/research/full_h_kappa2_scale3o2_q3_results_20261115/` |
| PF比較値の正本 | `1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/HANDOFF.md` |
| ロードマップ参照 | `ad6ab45bbcc7139bf499760d28efad794e4eae9b:docs/research/RESEARCH_ROADMAP.md` |

[ADMISSION.json](ADMISSION.json) の `rows` が26軸の入口。各行の `holdout_admissions` → `comparisons` → `frozen_prediction` と `comparison`、`receipt_admissions` → `original_record.locators` から元のX/Y価格、implementation evidence、compile record、raw manifest・decoded path selectorへ戻れる。`sources`はすべて完全SHA付き。D1公開後処理SOURCE `03cc993eeff827e87fc120a1aedb14a34fefcdf2` と元compile実行SOURCEは別である。

受入は既存Git reader `price_implementation_guard.evidence`、既存Cのmetadata比較helper、D1の公開軸別索引・lossless raw検証を再利用した。compile receiptの公開bytes/hash、価格・fingerprint・compiler、保存seed、preparation hash、scalar、new ZZ、測定軸・P0、元gate scopeを照合。巨大rawの全eventを再展開せず、D1の `actual_path_and_compile_evidence_checked=true` と同じraw hashを持つmanifestを参照した。D1の数値gateを再実行したという主張ではない。

## 保存holdout：転記のみ

| round | 軸 | fresh path数 | 保存観測平均RZ | 保存符号付き相対誤差（%） | C受入 |
|---|---|---:|---:|---:|---|
| m5 | X | 2 | 1,392,948 | −0.445257 | 2 receipt＋1比較 |
| m5 | Y | 2 | 1,392,948 | −0.445257 | 2 receipt＋1比較 |
| m6 | X | 1 | 5,551,296 | ＋0.134608 | 1 receipt＋1比較 |
| m6 | Y | 1 | 5,551,296 | ＋0.134608 | 1 receipt＋1比較 |

平均・予測・誤差は再計算せず、D1 JSONの分子・分母をそのまま保存した。X/Yは同じpathを共有する別軸であり、独立な6 trajectoryではなく3 fresh path。各cell index0の元7数値gate、全pathのparity/scalar/event/XY gateという実施範囲を保持した。旧 `scalar_once_legacy_diagnostic=false` は既知の不適切な旧指標であり、採用済みscalar/event phase gateのPASSと置換・混同しない。

小さな記述的誤差は、この2物理点・当該cohortについての観察である。2%、95%、全roundの許容差、CI、モデルのPASSを新設しない。較正5 cellへのrefitやholdoutの較正への移動も行わない。

## 全13 roundとBの固定配分

対象はLinear H4 1.00 Å、STO-3G、保存tol-only DF10、同じH/state/snapshot、finite K2、one-bodyを含む全非scalar H、決定論headなし、κ=2。`q`はFull-H groupingでありPF回数ではない。原carrierの `method=PR,prefix=0` に加え、`family=FULL_H_PURE_RTE,L_D=0,NO_PF_FULL_H_RTE` の意味を保持する。Full-H R448やPR-DFと統合しない。

| m | q grouping | R | event位置数（保存metadata） | N_X=N_Y | この物理点の保存証拠 |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 9 | 9 | 96 | D7直接価格なし |
| 1 | 2 | 18 | 36 | 90 | 同上 |
| 2 | 4 | 36 | 144 | 84 | 同上 |
| 3 | 8 | 71 | 568 | 78 | P1 single-path wiring X/Y。D7較正ではない |
| 4 | 16 | 142 | 2,272 | 71 | D7直接価格なし |
| 5 | 32 | 283 | 9,056 | 65 | D7独立holdout、2 path/軸 |
| 6 | 64 | 566 | 36,224 | 57 | D7独立holdout、1 path/軸 |
| 7 | 128 | 1,132 | 144,896 | 51 | D7直接価格なし |
| 8 | 256 | 2,264 | 579,584 | 45 | 同上 |
| 9 | 512 | 4,528 | 2,318,336 | 38 | G4528とはRだけ一致 |
| 10 | 1,024 | 9,056 | 9,273,344 | 32 | D7直接価格なし |
| 11 | 2,048 | 18,112 | 37,093,376 | 26 | 同上 |
| 12 | 4,096 | 36,223 | 148,369,408 | 18 | G36223とはRだけ一致 |

Tは保存binary64 h=0.1のschedule。各行にBのexact T、physical ID、original/total canonical ID、signal原結果、law selectorを保存した。original configuration `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`、total decoder view `265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612`、allocation `FULL_H_KAPPA2_TOTAL_Q3_USER_SCALE_3O2_01`。total viewへのbridgeは精度側の対応であり、receiptの生成IDを書き換えない。

Bの状態は、この固定配分について条件付きideal-model Q3閾値適合。十分Nの新認定ではない。今回Nの再評価・積算は0件。

## 較正と適用domain

各軸のモデルは `a + b·(qR) + c·(q−1)`、5 cellを各1行とする等重みexact最小二乗、rank3、`CALIBRATED_UNVALIDATED/qualified_for_candidate=false` のまま。

| 較正cell | T | q | R | 親path数・役割 | 実roundとの関係 |
|---|---:|---:|---:|---|---|
| G566 | 0.1 | 1 | 566 | 2 normal | m6のRのみ一致、q/Tが違う |
| G4528 | 0.1 | 1 | 4,528 | 2 normal | m9のRのみ一致 |
| G36223 | 0.1 | 1 | 36,223 | 2 normal、完成結果 | m12のRのみ一致 |
| Q8R566 | 0.8 | 8 | 566 | 2 normal | m3のqのみ一致、Rが違う |
| P2R566 | 0.2 | 2 | 566 | 保存G566親2本の順序付き連結1件 | m1のqのみ一致。独立第3親標本ではない |

これらは非configuration較正cellで、架空のround IDを付けない。元10較正record/18軸の受入、G36223 completion・producer/validator SOURCE、paired依存を再利用する。m5/m6で初めて実scheduleのq/R同時条件を独立に観測した。m0〜m4は低R・角度方向、m7〜m12は大q・同時増大の移送が未資格。特にq=1の大R較正をq=4096のfull-wrapper価格証拠へ昇格しない。

## 求める期待値と未解決事項

比較したい量は、Bと同じ理想finite-K2 law `I_m` における、各roundの全group・order/component選択を含むfull controlled measured wrapperの `E_I[C_m,a]`。元scalar −0.33147781341681615（Hamiltonian定数＋Full-H identity抽出）をtotal Tで一回適用し、new `CONTROLLED_ZZ_CX_CRZ_CX_V1`、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional、P0を固定する。

D7の応答はPCG64/float64 CDF samplerの保存有限cohort平均である。固定seed列自体には母期待値を定義しない。seed ensemble／独立乱数近似 `Q_m` と理想 `I_m` の対応、浮動小数点CDF・乱数格子、joint law差は未資格。PR-DFの324 componentに対するF0の上界をFull-Hへ移送しない。

`E_I[C]−モデル = (E_Q[C]−モデル) + (E_I[C]−E_Q[C])` を分ける。前項には有限cohort・モデル形・移送・共有係数誤差、後項にはlaw差と価格の全support変動幅が必要。order2未観測でも寄与0とはしない。m6の1 pathでは分散が得られず、m5の2 pathから全round CIは作れない。paired較正、X/Y共有path、全round共有係数の共分散も未取得である。

今回新しい平均・予測・fit・価格補正・N/Gは0件。保存E[S]をFull-H E[C]として利用せず、決定論PFのQPY閉包や2%規則も転用しない。

## 次の有限作業と技術確認

[NEXT_SCIENCE_PLAN.md](NEXT_SCIENCE_PLAN.md) にF1を具体化した。最大新規path/compileは**0/0**、fit/Q3/samplingも0。既存4予測を再利用し、未予測22軸と固定shot総額を次の明示承認後に診断する。未知のモデル誤差を0とせず、正式E[C]は引き続きnull。価格の非負性を満たさない場合にclamp・refitせず、モデル全額を未評価として保存する。

実装は薄い保存consumer `scripts/admit_full_h_d7_holdout.py` のみ。新6＋既存10の人工/回帰テスト16件PASS（[TESTS.txt](TESTS.txt)）。これは作者自身の技術確認である。実行経路の保存metadata受入も上記6/4/26件で完了。科学runner・価格provider・会計器の変更は0。

§10.5では指定固定SOURCE、関連Git履歴・公開branchを確認し、CのD7受入＋1.5倍26軸接続済みの同等成果は見つからなかった。既存fit・D1 raw/比較・B Q3は再利用した。他担当のworktree/private raw/processは変更・探索していない。全サーバーの非公開job不在を証明する主張はしない。今後の科学実行直前に同等算出の公開/進行確認を再度行う。

ロードマップ差分：D1 holdout受入・B 1.5倍固定配分への接続が完了。次は0-compileの固定shotモデル診断の承認判断。全26軸の正式E[C]/CI/Gは未確定。ロードマップ本体は更新しない。公開照合後mandatory STOP。
