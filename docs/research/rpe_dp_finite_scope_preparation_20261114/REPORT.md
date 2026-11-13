# 改善RPE MSE上界：実H4の部分追跡範囲の具体案

2026-11-14 JST。基準・数値kernelは`06db9aefec13bb5c5be6d7c0f81f29b3a10b929c`。既存branch `codex/rpe-decoder-statistics-20261113`を再利用した。今回は**有限範囲の具体化と人工benchmarkだけ**で、実H4の新しい2評価・旧risk結果の再解析・新shot探索・MC・価格/Gは実行していない。src、既存test、科学caller、旧input/raw/結果は変更していない。

## 推奨する有限2評価

| 項目 | 提案 |
|---|---|
| 対象 | 保存H4 PR-LD1/LD2各1件、計2 D.evaluate |
| 科学条件 | 前回43898ada1:INPUTSのraw/target/loss/IID/元configuration/全13round X/Y vector/ρ1を保持 |
| maximum_frontier_states | **8,192／round** |
| maximum_transition_pairs | **141,559,036／構成**、計2件で283,118,072組まで |
| 元count列挙・good検査 | 4,225 outcomes/round、55,043,173 good-support pairs/構成を保持 |
| 出力 | 新しい出力先。旧/DP/min上界、η、既知未解決u、残余r、zero/tie/未解決の確率又は範囲、保持状態数・組数・時間/RSS・UNKNOWN |
| 対象外 | 配分追加、N資格・精度達成判定、価格/G、追加再試行・自動追跡拡大 |

[FINITE_SCOPE.json](FINITE_SCOPE.json)はこの未実行案の機械可読値、[HANDOFF](HANDOFF.md)は承認後の処理・分解・STOP条件である。既に採択されたtarget等の再判断は不要。**今回の利用者指示どおり、この具体的な追跡範囲を提示した後、2件の科学実行を別途明示許可された場合だけ進める。**

## 範囲をこう選ぶ理由

既存DPの保持上限Kはround末に適用される。組数上限が小さいと、早いroundで予算を使い切り、以後の状態が丸ごと残余へ移る。これを避け、選んだKで全13roundを伝播できる組数を割り当てる。

C_m=(N_mX+1)(N_mY+1)、q_m=2^mとすると、前roundの保持状態は最大min(K,q_(m−1) C_(m−1))個、現在のcount群はC_m以下である。よって、

`P_C(K)=C_0+Σ_(m=1..12) min(K,q_(m−1) C_(m−1)) C_m`

を組数上限とすれば、**保持状態を組数不足で落とさず全roundを処理できる**。m0は仮想初期状態との組を含む。この定義はraw確率やtargetを使わない、count直方体の上限である。

| 保持数K | 保守的なP_C(K)／構成 | 固定count位相格子でのさらに鋭い組数上限P_G(K) |
|---:|---:|---:|
| 4,096 | 79,380,609 | 49,938,888 |
| **8,192** | **141,559,036** | **86,171,598** |
| 16,384 | 233,696,672 | 139,288,086 |

P_Gは同じcount格子を**人工raw=(0,0)、target=0**で全列挙し、zero flagとbinary64 principal位相の完全一致群G_mを数えた技術的な幾何量。G_mは`2593,2247,1951,1707,1883,1612,964,777,584,550,383,261,97`。[COUNT_GEOMETRY](technical01/COUNT_GEOMETRY.json)。保存signalの確率・MSEから推定した値ではない。実行上限にはこのfloat格子の細かさへ依存しない、より保守的なP_Cを採用する。

K8192は固定格子のm0..2の全候補状態（上限2593/4494/7804）を包含する。m3以降は必要に応じて最大質量順の部分追跡となる。小さい保持数で早期からtailを落とす案を避け、全roundの伝播組数も確保した。K16384は追加の候補だが、今回の中心案へ自動併合しない。

## 人工benchmarkと負荷

元DPを変更せず、[BENCHMARK_SYNTHETIC.py](BENCHMARK_SYNTHETIC.py)で測定した。全モデルは分子/H4を持たず、h10、target−.07、raw_m=.85 exp(i .7 2^m)、人工loss=[0,20]。保存signal/solver/科学結果は一切読み込まず、SavedArtifacts.readとMC/物理providerをmockで禁止した。

| 人工入力 | K | 実遷移組 | wall / CPU秒 | peak worker RSS (KiB) | 追跡外質量r（人工のみ） |
|---|---:|---:|---:|---:|---:|
| 2round、N_X=N_Y=64,60 | 8192 | 5,826,817 | 202.651 / 202.640 | 57,072 | 0 |
| 13round、各軸N=8 | 4096 | 1,302,273 | 17.839 / 17.836 | 114,684 | 約1.65625e−6 |
| 13round、各軸N=8 | 8192 | 2,205,441 | 31.122 / 31.119 | 222,832 | 約2.19176e−7 |

人工13roundではK8192で小さい残余まで保持でき、K4096より細分化できる例を確認した。これはH4で同じ残余になる証拠でも、N/budgetに対する一般の単調性の証拠でもない。H4で厳密な改善幅を保証するには今回保留した実2評価が必要である。

[整数演算probe](technical01/ARITHMETIC_PROBES.json)では、16,384bit×8,192bitの積＋和が約25.76μs、65,536bit×2,048bitは約135.31μs、65,536bit×8,192bitは約180.33μs／組（CPU、固定人工operand）。実H4の整数bit長は測っていない。NumPy遷移、Python loop、dict/sort、最終Fraction lossや保存も別に必要になる。**1件が数十分〜数時間になる可能性を見込むが、完了時間の上下界や保証ではない。** 数秒で終わる旧評価と同等とは扱わない。wall cutoffを新設せず、科学範囲は保持数・組数で定義する。

Kは一時メモリ上限ではない。round内には枝分かれ後の状態を作ってからpruneするため、count直方体では最大q_m C_m=692,224状態、固定位相格子では最大534,528状態まで増え得る。整数のbit長・dict/sortの一時領域も必要で、人工N8のRSSをH4へ比例外挿しない。各worker数GiB以上になる可能性を含め、実行時のRSSを記録する。観測時は64 CPUと約476,342,644 KiBのavailable memoryが見えていたが、他担当を凍結せず、将来の利用可能量の保証にしない。

独立2構成を自然な2workerで評価する案。CPU/RAM/workerの恒常capや共有設定変更は導入しない。他担当・監査のworktree/processへ介入せず、自分のworkerだけを観測する。

## 改善可能性と限界

旧最大lossを残余に使うため、保持数だけ大きくてもrが大きければ改善を得られない。評価後はη、u L_U、r L_badを分離し、どれが支配するか判断する。`r L_bad < U_old`は改善に必要な条件の一つだが、ηと未解決寄与もあるため十分条件ではない。今回は旧H4数値を代入した新比率・感度計算をしていない。

出力のmin上界は旧上界より悪化しないが、改善を保証するものではない。known zero確率と実全体tie範囲も保存し、部分追跡でunknownな履歴を0にしない。原UNQUALIFIEDは既知の条件付き診断scopeとして保持し、科学的energy資格/十分N/global bound/Gはnull又はUNKNOWNのまま。AδEの加算・Minkowski変更はしない。

技術確認は新しい人工3モデル、count幾何、整数3probeと保存artifactの整合確認。[CHECKS](CHECKS.txt)。数値kernel不変のため旧57件は再実行せず、既存PASSを参照した。新しいoptimizer/caller/科学kernelは不要だった。

**次：ユーザーが本範囲でLD1/LD2各1件・計2件を明示許可するか判断し、Codex Bは許可後だけ実施する。今回はcommit/push・remote照合後mandatory STOP。**
