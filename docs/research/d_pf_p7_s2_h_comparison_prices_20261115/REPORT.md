# B0I p7/S2・幅hの比較用26軸価格と固定配分モデル費用

保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、B0I p7/S2、要求幅h=.1、P0除外。元H/state/snapshot、一体項＋fragment0..6保存順、S2符号付き構成、scalar c_H＋tail identityのtotal T一回適用、symmetric_directional control、測定付きfull controlled Hadamard X/Y wrapperを対象とする。Qiskit1.3.0 / basis rz,sx,x,cx / opt1 / seed17。

**採用式 `C_ref(q)=5513q+307 RZ` で比較用価格を26/26軸へ接続した。直接価格は10軸のまま、残る16軸は外挿。固定Q3適合配分のモデル総費用は993,110,014 RZで、同配分・幅16h/7の保存達成費用435,481,090 RZとの差は＋557,628,924 RZ（＋128.048941%）となる。** 新compile・QPY監査・fit・Q3・配分生成は0。

## 結果と証拠の区別

| 部分 | 軸数 | 費用 RZ | モデル総費用に占める割合 |
|---|---:|---:|---:|
| m0〜m4の受入済み直接価格 | 10 | 16,000,932 | 1.6111943062% |
| m5〜m12の採用式による外挿 | 16 | 977,109,082 | 98.3888056938% |
| 合計 | 26 | **993,110,014** | 100% |

表の全額は**採用価格モデルに条件付けた比較値**で、全26wrapperを直接compileした達成費用ではない。長qの誤差上界・価格CIは取得していない。未知誤差を0とせず `price_model_uncertainty_RZ=null`、`certified_total=false`、`G_qualified=null` とした。

価格は [run01/COMPARISON_PRICES.json](run01/COMPARISON_PRICES.json)／[CSV](run01/COMPARISON_PRICES.csv)、固定配分の26項とexact整数総和は [run01/FIXED_ALLOCATION_COST.json](run01/FIXED_ALLOCATION_COST.json)。表は元configuration `e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc` と各roundのphysical ID・axisで接続する。元のdirect表を変更せず、モデル値を `compiled_RZ` やdirect receiptへ書き戻していない。

## 規則との一致と適用根拠

| m | q | 直接X RZ | 直接Y RZ | 採用式 RZ | 各軸の差 |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 5,820 | 5,820 | 5,820 | 0 |
| 1 | 2 | 11,333 | 11,333 | 11,333 | 0 |
| 2 | 4 | 22,359 | 22,359 | 22,359 | 0 |
| 3 | 8 | 44,411 | 44,411 | 44,411 | 0 |
| 4 | 16 | 88,515 | 88,515 | 88,515 | 0 |

X/Y各5行を別々に比較し、10/10が整数完全一致。これらは**規則の選択に使用した5物理点・10軸の観測**で、独立holdoutではない。今回係数のfit・推定・再選定はしていない。m0〜m4の単価は保存直接価格そのものを使い、式を適用して新しい単価を与えたのはm5〜m12だけである。

固定scheduleは q=2^m、T=.1×q、actual幅binary64 `0x1.999999999999ap-4` が全roundで同じ。固定fragment順と同じS2ステップの反復であり、反復数に比例する本体価格と有限の端効果を持つ規則を採る構造上の根拠になる。ただし5513を解析的に導いたゲート数、307を厳密に数えた端価格と認定したものではない。

full wrapper全体の最適化は、step間の相殺・回転角の簡約・数値的な枝分岐・終端runに依存し得る。特にtotal Tに一回のcontrolled scalar phaseはqとともに変わり、X/Yの終端操作も別である。5点での一致だけではq=32〜4096で同じ価格則が厳密に続く証明にはならない。今回の採択は、この候補の費用比較用モデルの利用を認めるもので、別幅のQPY閉包採択・m6/m9の2%条件・他公式や候補の価格資格を移送しない。長qは独立検証0軸、誤差上界UNKNOWNのまま残す。

## 同じshot配分での精度と費用

保存配分は X/Yとも `[59,55,51,47,43,39,35,31,27,23,19,13,4]`、全892 shot。整数計算は `Σ_(m,a) N_(m,a) C_(m,a)`。直接部分と外挿部分を分け、別形 `5513×Σ(Nq)+307×ΣN`（ΣNq=180,090）でも993,110,014を照合した。これは同じ条件付きモデルに対する算術検算で、追加科学観測ではない。

以下のQ3は `bec032fc26093177f529754060df06d5375944b4:docs/research/p7_s2_d01_same_shots_q3_results_20261115/SUMMARY.json` の保存値を引用した。

| 要求幅 | 保存Q3モデルMSE区間 Ha² | 保存判定 | この配分の費用 RZ |
|---|---|---|---:|
| h=.1 | [1.113188114132502061033934e−6, 1.113188114132502061033945e−6] | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | **993,110,014（直接＋外挿モデル）** |
| 16h/7 | [1.506660616251764083585358e−6, 1.506660616251764083585362e−6] | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | **435,481,090（保存達成費用）** |

保存MSE差（h−16h/7）は `[−3.934725021192620225514282e−7, −3.934725021192620225514126e−7] Ha²`。この配分を保って細幅にすると、保存Q3条件付きモデル上の精度は改善し、採用価格規則上の費用は増える。費用差は正確に＋557,628,924 RZ。hでshotを減らした場合の有利不利・新しいQ3適合配分・最適性は未評価。

保存Q3はZ1＋T1 total decoder、保存solver proxy、全事象平方loss、IID二項モデル、exact τの条件付き評価。元資料の数値的・物理的資格限界を引き継ぎ、十分N・厳密ground-energy保証・global RMSE保証へ昇格しない。粗幅の435,481,090 RZは粗幅にだけ保持し、hの直接達成額として使わない。

## SOURCE・検査・資源・保全

- 直接価格入力：`e04a14797a56a586dcb72065a55f4d97bfa1479b:docs/research/d_pf_p7_s2_h_direct_prices_20261115/run01/AXIS_PRICES.json`。
- 精度・配分入力：`bec032fc26093177f529754060df06d5375944b4:docs/research/p7_s2_d01_same_shots_q3_results_20261115/C_RESULT_JOIN.json`。SUMMARYも同commit。3入力のSHA-256とlocatorは各出力 `provenance.inputs` に保存。
- 今回の実行SOURCE：**`33d75cfbf0e891a53184e2973286f6cb9d6ef82b`**、[build_comparison.py](build_comparison.py)。元native・compiler・価格受入・Q3・C/B branchは変更なし。
- §10.5の照合： [SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。同候補・同規則・同配分の後続表／総費用は関連D/Bの確認範囲で未確認。保存入力は全部再利用、新観測は0。
- 人工fixture3 test PASS、軸/ID/q/幅/配分不一致・誤った直接昇格を拒否。 [TESTS](TESTS.txt)。公開表は26キー一意、CSV一致、10直接receipt保全、16外挿のdirect receipt null、整数加重・部分額・比較差・入力SHAを照合。 [CHECKS](CHECKS.txt)。primary agent自身による検査で、独立レビューではない。
- 計算はstdlib・固定JSON3件・26整数項のみ、wall約0.0417 s。worker起動・重い科学APIの再実行が不要な規模で、システム設定・他者processは変更していない。

旧direct 10/26、元SOURCE/input/raw/cache/receipt/sidecar、凍結モデルと各STOPは保全。比較表が26/26になっても、直接価格coverageの増分は**0軸**。公開commitを指すGit履歴と実行SOURCEを区別する。

## Bへの接続・ロードマップ差分・STOP

[HANDOFF_B](HANDOFF_B.md) にID・単価・originと使い方をまとめた。この候補の別shot配分は同じ26軸表を再利用できるが、精度評価・配分選択はBの担当で今回未実行。統合担当へは「hの比較用価格26軸が完成、この保存Q3配分のモデル費用を取得、直接証拠は10軸のまま、外挿費用比98.389%」を渡す。ロードマップ本体は変更しない。

commit/push・remote SHA/tree/主要成果物の一致確認後 **mandatory STOP**。追加compile・QPY監査・fit・Q3・別配分生成・他候補比較・winnerへ自動進行しない。
