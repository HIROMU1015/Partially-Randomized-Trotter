# Cへ：PR-DF LD1 全13 round E[C] 評価計画

**推奨する次の1作業は、既存F0を保存H4確率へ適用する有限law診断。新path0・compile0。今回は計画公開のみで、F0も未実行・未採択。** CのFull-Hモデル凍結やC2比較規則とは独立に進められる。

[REPORT](REPORT.md)／[26軸差分索引](AXIS_EVIDENCE.json)／[CSV](AXIS_EVIDENCE.csv)／[次の有限scope](NEXT_F0_PLAN.json)／[metadata検査](CHECKS.json)。既存C26表の複製・価格集計をせず、各行を元SOURCEへ結んだ。

## 保存証拠の受入境界

- Bの固定Q3 `a82b0f0e…` は `ideal_finite_RTE_law`、sampler移送資格false、十分N/Gはnull。現在の条件付きG計画はIを維持する。
- C26 `c99d511e…` / HANDOFF `8eef0ad7…` のE_I[S]は26軸接続済み。Sの数値enclosureは実Qのlaw差やMC/total CIではない。
- m0/m6はP2 fresh各4 pathを較正、元role DEVELOPMENTを保持。P1各2・replay各1は参照のまま。
- D2 m3 `2592d442…` のfresh4 pathは較正。C `3cac53da…` の軸別凍結モデルはm0/m3/m6の3点・3係数、lack_of_fit_DOF=0。
- D2 m4 `063f2b3b…`、C最終受入 `9edfa5d3…` は独立holdout4 path×X/Y・8 receipt/2比較。約−0.544502%はその有限平均に対する保存済み差であり、再計算していない。m4を較正へ戻さず、モデルPASS/FAILなしを保持する。
- 全26軸の正式E[C]・CI・長round移送は未資格。I/Q差、有限path/rare、共有係数の不確かさをm4の1点だけで分離・認定しない。

I-targetにはE_I(C−S)の根拠が必要で、Q残差平均だけでは不足。Q-targetには保存E_I[S]を代用する補正が必要。Q上で直接Cを評価すれば構造law置換は不要だが、有限標本資格は別に要る。既知のE_Q[S]を制御変量へ使う利点はあり得るが、同じ標本の平均S＋平均(C−S)に書き換えるだけでは平均Cと同じである。

## 次の有限案（別の実行指示用）

| 項目 | 範囲 |
|---|---|
| helper/計画の再利用元 | `d72dc4e6228bb380f4d1d000e5e030794fce7164:src/trottertracks/finite_k_sampler_law.py` と同commit `docs/research/finite_k_sampler_law_alignment_20261113/SCIENCE_PLAN.json` |
| 保存入力 | a637ed1b… closureの324 componentとm0/m6のlaw、9d95d3ea… LAWSの残11round。完全commit/path/selectorはNEXT_F0_PLANに固定 |
| 数量 | 1 component表＋13 order表＝最大14 categorical audit、13 K2 event TV＋13 path TV、X/Y26軸へ参照 |
| 新path / build / compile | **0 / 0 / 0**。E[S]/E[C]/fit/予測/CI/Q3/N/G評価も0 |
| 比較するlaw | SOURCE raw weight比・choice入力・丸めCDF・一様53-bit格子。実PCG64 joint-lawとは区別 |
| 得るもの | exact TVの分解、support零化、order2 mass、仮定を明記した全round条件付きpath bound |
| 得られないもの | actual PRNGのIID証明、解析実数入力誤差、価格全support range、E[C]/CI/rare精度・長roundモデル資格 |
| 出力案 | `docs/research/d_pr_df_ld1_f0_law_diagnostic_20261115/run01`（現在未使用） |
| 実行担当案 | D2が既存branchで最小readerを準備し、SOURCE/入力を実行前commit固定。Cの別worktree変更は不要 |

旧F0のSOURCE調査と人工12＋toy5件は再利用する。数値適用は未実施なので、それだけが新しい差分。§10.5の既存照合は本計画で再利用し、実行直前は新しい同等結果/job/出力先/SOURCEの差分だけを見る。受入可能な既存診断があれば新規14件から減らし、出所不明は保留する。

STOP：保存入力/identity/順序/SOURCE不一致、CDF mass/非有限等の異常、同等job/receiptの不明点で該当分を止める。補完sampling、確率の再生成/修正、guard緩和、無断retryはしない。大きい又は有用性不明の上界も保全し、追加価格取得やgrid期待構造の計算へ自動拡張しない。F0の公開後もSTOP。

## C/Bへの接続と未採択事項

F0は価格provider recordではなくlaw診断である。既存C guard/consumerのUNQUALIFIEDを自動解除しない。price shiftには対象observable（IならC−S、QならS）のfull-support oscillation等が別に必要。構造の数値区間・m4差・観測min/maxでは代用しない。

将来の正式E[C]/Gには、実PCG64を使う確率モデル/保証水準、価格精度・同時信頼性が未採択。PF2%、他方式の許容差、旧n4や95%を流用しない。Gへの価格誤差は固定Nで重み付けして設計し、共有係数/X-Yの相関を無視したtotal CIを作らない。今回N×C積和は0。

ロードマップ差分：E[S]/有限価格/凍結/holdout/正式E[C]の26軸対応を整理。m4 C受入待ちは解消済みで、主要不足はlaw接続と母平均/モデル資格。次は既存F0を別指示で有限実行し、その結果と精度契約から追加n/roundを決める。長roundの便宜的少数compileを先に置かない。ロードマップ本体は統合担当が更新する。

固定SOURCE/raw・凍結モデル・m4比較・旧STOPを保持。本計画のcommit/push・remote/成果物一致確認で **mandatory STOP**。後続の数値解析/取得を今回実行したものと扱わない。
