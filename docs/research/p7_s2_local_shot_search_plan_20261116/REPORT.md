# p7/S2・16h/7：保存適合点からの有限shot近傍探索案

2026-11-16 JST。**保存適合点から1〜2roundだけを変更し、新規Q3は最大8件、435,481,090 RZを厳密に下回る適合点が1件見つかれば終了する案を推奨する。** 最初に1shot減と2shot減を別々に調べ、未達なら少量のshot振替を試す。各候補は全13roundを結合してQ3判定する。

今回は計画のみ。新配分の生成・費用算術・MSE再解析・Q3/DP/二項確率・価格取得は実行していない。以下の探索規則・最大8件・結果依存の次候補選択は**提案**であり、科学実行には別途採択・明示承認が必要である。

## 1. 既存適合点と固定入力

H4 linear / 1 Å / STO-3G / 保存tol-only1e−8 DF10、sector(2,2)、B0I p7/S2・要求幅16h/7。元H/state/snapshot、候補固有の13round schedule、scalar、raw、normalizationは不変。

| 保存配分 | N_X=N_Y、m0〜m12 | 保存費用 RZ | 保存Q3上界 Ha²（表示） | 使い方 |
|---|---|---:|---:|---|
| **B_PRICE** | **[59,55,51,47,43,39,35,31,27,23,19,13,4]** | **435,481,090** | **1.506660616251764083585362E−6** | 主出発点S。原API・資源まで保存済み |
| A_SHAPE | [46,43,40,37,33,30,27,24,21,18,14,11,8] | 440,329,944 | 9.797745654510526762188602E−7 | 別の保存適合点。履歴照合に利用し、再評価しない |
| N0 | [59,55,51,47,43,39,35,31,27,23,19,15,11] | 593,555,352 | 8.356033913794202152063318E−7 | 過去の達成点と探索上限U。原API等の部分保存状態を維持 |

すべて保存値の引用で、新しい合計・比率は算出していない。正本は **b4296b110b129d226dbef2743c14f725a27c3bf6** の [PF6 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4296b110b129d226dbef2743c14f725a27c3bf6/docs/research/pf6_cost_budget_q3_results_20261115/REPORT.md)、同ディレクトリのINPUTS/SUMMARY、`run01/B0I_P7_S2_16h_over_7__B_PRICE/API_RETURN.json`。N0の元componentは8dba096dc…で、原API欠測を今回補わない。

固定するidentityと入力：

- 元configuration：`a6e557770a2e4d96585ad5ca5903c4626ae47dda9775e9a9babc756149c38d84`。
- total view：`e496358b012be53878305a83a3bfed42ae4d444d3c0b6fe6dee061d2f44a7abc`。
- binding：`d54a41b4520e36c02b306ad35715f4a731fef0fd17754a4d5c828badb756faad`。
- signal/PLAN：`06a15ee45beb177c018f6260ea4a5ac8cf66cfe6:docs/research/pf_shortlist_mean_completion_20261115/run01/B0I_P7_S2_16h_over_7/` の `B_SIGNAL_INPUT.json` と `PLAN.json`（configuration selector）。表示幅からqを生成し直さない。
- 26軸価格：`1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/AXIS_TERMS.json` の当該configuration行。p7/S2分は**16直接軸＋10条件付き閉包軸**であり、B2全体の96＋60という内訳と区別する。元receipt/生成ID・compiler/lowering・測定付きfull wrapper・P=0・CI/価格誤差上界nullを保持する。

比較するQ3契約は、Z1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象loss=(Ehat−Eproxy)²、保存deterministic PF raw・shots/X/Y/round間ideal IID二項、EXACT_AFFINE_OF_BINARY64_RAW。閾値は十進εE=0.00159360010199040 Haのexact二乗τ。元Minkowskiは別参照、AδEは再加算しない。

## 2. 推奨する最初の有限探索

### 探索範囲と変更単位

`1 <= N[m,a] <= U[m,a]`、Uは上表N0。今回は両軸を同量変更するround単位の操作を使う。`e_m`はround mのX/Yをそれぞれ1増やす操作であり、他roundは不変。X/Yを独立最適化した結果ではなく、初回の探索族を小さくする選択である。

1候補の変更は出発点Sから**最大2round、各round各軸の変更量は1又は2**。公式・幅・physical candidateは変えない。この上限は科学探索領域の提案で、CPU/RAM制限ではない。

### 第1段階：1roundずつ減らす、最大4新規call

| 優先順 | 候補の作り方 | 確認したいこと |
|---:|---|---|
| 1 | S−e_12 | 最小の変更で高単価roundを削り、適合余裕を費用削減に使えるか |
| 2 | S−2e_12 | 1shot減と異なる整数・偶奇を直接評価する |
| 3 | S−e_11 | m12を保持したまま、次の高単価roundを削れるか |
| 4 | S−2e_11 | m11の2shot減を独立に確かめる |

順序は高単価の末尾roundを小さく変えるという実験設計である。保存単価はm12各軸9,879,603 RZ、m11各軸4,939,955 RZ。**単価をMSE寄与と同一視していない。** 各案の全整数列・実費用・差額は、承認後に元26軸から生成してQ3前に固定する。

どの候補もSから作り、直前の不適合点を次の減算の出発点にしない。1shot減が不適合でも2shot減を棄却せず、逆も推論しない。全体のMSEがNに対して単調、凸、1/Nに比例すると仮定せず、単調探索・二分探索はしない。

### 第2段階：価格と既評価Q3を使う2round振替、最大4新規call

第1段階で目標点が見つからなかった場合だけ、次を候補とする。

```text
F = S − d e_r                         r ∈ {12,11}, d ∈ {1,2}
V = F + s e_k                         k ≠ r, s ∈ {1,2}

採用条件：
  Fは第1段階で有効Q3結果を取得済み
  1 <= V <= U
  T(V) = Σ[m,a] V[m,a] C[m,a] < 435,481,090 RZ
```

すなわち、高いroundを減らした費用の一部を、別roundへ1〜2shot戻す。受取roundは元のcapに空きがあるものだけとし、価格差で純削減になる案だけを残す。追加shotの効果も単調とはしない。単純な減算で失敗しても、2roundを結合した案のQ3は未評価のままである。

次候補の選択は以下の固定規則を提案する。

1. 各Fとそこから既評価した振替を一つのfamilyとし、そのfamilyで得た**有効MSE上界の最小値**を優先度とする。未知案のMSE予測値や、十分性の判定には使わない。
2. 未評価Vが残るfamilyをその優先度順、同値なら候補の実費用Tが小さい順、さらに変更量合計・r/d/k/sの辞書順で選ぶ。同一family内も費用・変更量・辞書順を使う。
3. 新Q3が返ったら履歴を追加し、次の優先度と未評価リストを更新する。失敗した案を他案の不適合証明にせず、1shot戻しの不適合から2shot戻しを棄却しない。
4. 振替の新規callは合計4件まで。候補が尽きれば終了し、roundや変更量を拡張しない。

この優先度は近い適合点を探すためのheuristicで、risk上界を補間・外挿するモデルではない。first-unsafeやzeroの頻度だけから「このroundへ戻せば直る」と断定しない。

## 3. 保存再利用・判定・停止

### 各callの前

- configuration/13roundのphysical IDs・signal SOURCE・law/normalization/scalar、全26個のN、decoder、targetの値/意味/SOURCE、loss、probability rule・数値allowanceを§10.5で照合する。保存APIを受け入れられる場合は再利用し、Q3件数を増やさない。同じNでも別幅・別方式のQ3は代替ではない。
- 候補固有の元価格とexact整数/有理数会計でT、基準点との差、直接/閉包等の内訳を求める。欠測を0や新モデルで補完しない。Tが同額なら目標未達であり、strict improvementへ丸めない。
- 実行SOURCE、具体allocation ID・X/Y列、変更元、価格参照、T、親Q3結果と選択理由、未使用call数を固定する。適応的に生成した入力を、元seedの固定INPUTSに含まれていたと表示しない。
- `run_pf6_cost_budget_q3_fixed.count_bounds`で各具体列を確認。ρ=1、outcomes4,225、verification55,043,173、非終端frontier8,192、transition141,559,036という既存scopeを提案する。cap以下でも候補固有のbinary64閉包PASSは推論せず、各新call内で検査する。

### Q3判定と目標

| Q3の有効区間とgate | 扱い |
|---|---|
| gate成立、MSE上界≤τ | モデル適合。T<435,481,090なら目標達成して**直ちに終了** |
| gate成立、MSE下界>τ | 当該列のモデル不適合。履歴に保持し次候補へ |
| gate成立、下界≤τ<上界 | INCONCLUSIVE。適合点には使わず、別候補を規則内で検討可 |
| SOURCE/閉包/質量/残余loss/上下界等のgate不成立 | UNKNOWN。部分証拠を保存して探索停止。無断retry・scope拡張なし |

新規Q3上限は**4＋4＝8件**。保存再利用・重複・整数範囲外・価格条件不成立による省略は新規callに数えない。全候補を一度ずつ訪問し、再利用だけの無限loopを避ける。目標達成、8件消費、有限候補枯渇、重大異常のいずれかでmandatory STOP。

初回は「1件でも厳密に安い適合点を見つける」が終了目標であり、発見後に残枠を使って最小化を続けない。不発でも今回の近傍で未発見と報告し、幅全体の不可能性・局所最適性・最小Nを主張しない。

## 4. 他の幅・公式・PR候補にも使うための入力

新controllerは作らず、候補固有の次の入力を既存APIへ渡す薄い手順とする。

| 入力 | 必要内容 |
|---|---|
| candidate context | 元/total configuration・bridge、全round physical IDs、H/state/snapshot、schedule、law、normalization/scalar、signal packetとSOURCE |
| price table/policy | 同候補・round・axisの測定付きfull wrapper/P0単価、元receipt、compiler/lowering、直接/条件付き閉包/モデル/有限cohort/全lawの区分と資格 |
| seed/history | 同一契約で有効Q3適合のwhole-vector、元結果SOURCE、既評価配分のAPI又は元componentと保存状態 |
| search policy | cap U、基準費用、donor2round、減算幅{1,2}、振替加算幅{1,2}、8call/早期終了規則 |
| Q3 contract/scope | decoder・target/loss/τ・測定モデル・数値allowance・finite scopeを明示固定 |

別候補では、既存26軸表から高いround単価を持つdonor2roundを選ぶ手順を同じにする。X/Yが非対称なseedなら各軸の元値を保持して同量を加減し、同一Nへ強制しない。paired-axis変更は初回の探索族であり、軸別最適性を主張しない。

PRのfinite-K raw、法則、normalization、辞書、fixed-grid等は候補固有bridgeを使う。PFのdeterministic/B=1や粗幅の価格を流用しない。価格policyがモデル・有限cohort比較を採択している場合も、その条件付きTと全law期待価格を区別し、未資格値を正式Gへ昇格させない。signalと価格だけで足りず、有効なseedと同一risk契約がなければ受入準備に留める。

再利用先は `rpe_shot_allocation.prepare`、`rpe_total_binding`、既存Q3・高速DP、PF6 callerのcount-only/worker保存/supervisor、既存exact会計。将来必要な追加実装は**履歴から上記有限候補を選んで既存1call workerを呼ぶ順序**だけで、decoder・risk・価格モデルを作り直さない。科学実行前に人工fixtureでdedup、非単調な合否、INCONCLUSIVE/UNKNOWN、早期終了、call上限を確認する案とする。今回は実装・人工テストも追加していない。

候補選択は結果に依存するため逐次とし、投機的に後続Q3を並列投入しない。各Q3内部は既存高速DPとprocess並列・段階log・worker自身の原API/資源保存を再利用する。8件は科学callの上限であり、CPU/RAM/workerの人工制限ではない。

## 5. 成果の位置付けと今回のSTOP

新成果は有限探索の設計。関連する現行caller・PF6履歴・近傍に関する資料/Git履歴を限定確認した範囲では、この適応的8call手順の同等実装/実行結果は未同定。従来のbreakpoint生成は再利用可能な費用会計・履歴であり、今回の局所編集・結果依存の順序とは異なる。幅hでの2列不適合を粗幅の新候補判定へ転用しない。

将来得られるのは保存raw/IID/solver proxyと価格policyに条件付く、指定近傍で確認した費用改善点である。十分N・物理的ground RMSE・最適N・方式winnerは未認定。元Minkowski、AδEの非二重計上、数値certificateなし、元SOURCE/raw/結果/旧STOPを保全する。計画公開後mandatory STOP。科学実行や費用算術を自動開始しない。
