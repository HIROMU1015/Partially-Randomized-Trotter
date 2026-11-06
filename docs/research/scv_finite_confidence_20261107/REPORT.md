# O2-SCV：有限confidence・総費用への接続調査

2026-11-07 JST。**静的調査・設計資料。新科学結果・新統計定理・科学的GO/NO-GOの正式採択なし。**

SCVの層内分散削減を有限confidenceの十分shot数へ接続する理論・実装経路は存在する。
ただし、独立calibrationでactual frozen残差の分散上界を取得し、range・全失敗予算・biasを含む整数Nを
比較する必要がある。224-shot sample varianceだけではこの条件を満たさない。
train/cal込み総RZ改善およびPR＋RPE全体の優位性は未検証である。

## 成果物

| 文書 | 内容 |
|---|---|
| [SCV_FINITE_CONFIDENCE_MAP.md](SCV_FINITE_CONFIDENCE_MAP.md) | 不偏性、oracle/有限train/Born分解、range、cal上界、Bernstein/Hoeffding、失敗・bias・返却 |
| [SCV_TOTAL_COST_COMPARISON.md](SCV_TOTAL_COST_COMPARISON.md) | 同quotaと最適化B1-OL、連続oracleと整数N、全stage RZ、期待費用と実現費用 |
| [SCV_RPE_CONNECTION.md](SCV_RPE_CONNECTION.md) | normalization、一般半径、訂正後pi/3、energy、round別入力・全学習費、現APIの制約 |
| [SCV_REUSE_AND_VALIDATION_MAP.md](SCV_REUSE_AND_VALIDATION_MAP.md) | 実在APIの入出力・条件・固定版・テスト・最小接続、224-shot後の受領項目と判断材料 |
| 本REPORT | 要点、取得資料、未実施事項、保全と公開境界 |

## 1. 確認した正式指示・固定資料

最初に元workspaceのPROJECT_MAP、適用AGENTS、正式共通指示2026-10-25を取得した。
共通指示§8.1/8.2/8.4/10/10.3を読み、公開保存版とのbytes一致を確認した
（SHA256 e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720）。
独立worktreeの固定版AGENTS/PROJECT_MAP、研究概要全文、研究目的・方法、
VALIDATION_STATUSとmanifestの該当statusを確認した。
公開リンク・全API対応は[再利用表](SCV_REUSE_AND_VALIDATION_MAP.md)に集約した。

| 固定commit | 読み取った範囲 |
|---|---|
| [560c6092bfb5e71af8eedc4a4be5416922764f98][prep] | SCV準備REPORT、SCV_IMPLEMENTATION_REPORT、SCV_224_SHOT_EXECUTION_DRAFT、scv_scalar/scv224_pilotと再利用SOURCE |
| [84e021b6a8decde7fa3ed3fa44d7148cc5428061][design] | 層内control候補、SCV理論設計、次科学草案 |
| [8d8a290fa4d4cf24f968aea27953ad5422e418b3][theory] | A1対B1-OL理論 |
| [744e1a09661ebba6841fe5faf456f88c41c47780][n1] | N1 finite law報告 |
| [5e9bd08068738cdf1070bdc43bee8194b113ae33][n2] | N2 state認証報告。総engine保証はN2単独で閉じない |
| [fdf00cee3c87d804845497afc89ac6a71f3b3f59][engine] | 後続のengine条件付き認証報告 |
| [3c8e55a2728f86a1eda17a2e0105050a637e7c86][pilot] | 752-shot attempt02の既存完了報告。raw再解析なし |
| [a53c1d6eedea0d85fb484425db9012aa89738129][expected] | 既存期待RZ保証計画。設計案であり、実装済み保証とは扱わない |

一次文献は[Glynn–Szechtman §2](https://web.stanford.edu/~glynn/papers/2002/GSzechtman02.pdf)、
[Maurer–Pontil v1 Theorem 10](https://arxiv.org/pdf/0907.3740v1)、
[Vershynin First Edition Theorem 2.8.4](https://webapps.math.uci.edu/~rvershyn/papers/HDP-book/HDP-1.pdf)の該当箇所を確認した。
既知式の適用整理に限定し、詳細GPT/Claude科学レビューは開始していない。

固定対象は保存linear H4 / 1.00 Å / STO-3G / legacy DF rank12 / 8 system qubits / 全4電子sector /
$L_D=3$ / $m=0,q=1,\delta_{\rm time}=.02,r=1,K=2$ / coefficient_atol=0 /
N1 actual finite-grid Q・selected full / N2保証state・認証済みexact engine / X,Y。
価格scopeは状態準備だけを除くfull interrogation、
Qiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed17。理想gate・保存DFモデル限定。
本調査はこの条件の新しい平均・分散・価格・biasを算出していない。

## 2. 分散削減からshot保証へ

trainから独立な評価、actual条件付き平均$\nu_h$、評価前の係数・quota凍結、全正層保持により
SCVは同じfinite signalを不偏推定できる。
削減可能なのはpath依存平均のうちcontrolで説明できる成分で、Bornノイズは残る。
finite train係数の損失をoracle改善と区別する。

既存calibrateは固定件数の独立calからMaurer–Pontil型の残差分散上界を作り、
sufficient_shotsは二側Bernstein/Hoeffdingの十分数を外向きceilし、小さい方を返す。
sample variance=0を母分散0にせず、h1の幅$D_1=2+|\lambda_1|\leq4$、
inactive4層の幅2と、cal/prod失敗確率、bias headroomを使う必要がある。
分散が下がっても、range増大、cal上界の保守性、ceil/floorにより十分Nが減らない場合がある。
これは最小必要shot数を求める保証ではない。

## 3. 総RZ改善の条件と不足

連続oracleのvariance目標$v$では
$C_a(v)=F_a+K_a/v$、
$K_{\rm SCV}=(\sum_hp_h\tau_h\sqrt{c_h})^2$。
改善条件は$K_{\rm OL}-K_{\rm SCV}>v(F_{\rm SCV}-F_{\rm OL})$だが、
有限confidenceの実十分Nではない。

実装で比較すべき条件は、**同じ精度・成功/返却条件で減る評価RZが、追加train/probe/cal RZを上回ること**。
不足するのはactual係数/range、独立cal上界、両方式の整数quota、全stage価格、返却率とSTOPの扱い。
期待費用の保証にはさらに価格上界・期待単価区間・履歴変動とideal bit lawへの移送根拠が要る。
sample variance×実現RZは探索的point scoreであり、その代用ではない。

## 4. 公平なB1-OL比較

224-shot案と対応する同quota paired比較は、層内追加効果の診断である。
将来の保証付き比較ではB1-OLにも独立cal、同じ有限誤差配分候補、quota最適化・価格情報を認め、
同じ$\epsilon/\alpha$、full価格scope、floor/cap、返却条件を使う。
共有評価は実支出で一回だけ課金し、baseline評価参照費とSCV全費用を別々に表示する。
有利な一軸・完走者だけを選んで全complex signalの優位とはしない。

## 5. 再利用できる実装と最小差分

ConditionalSampler、ConditionalBank、FrozenCV、fit_cells、calibrate、sufficient_shots、
choose_allocation、production_mean、SavedPrices、全stage会計・journal/STOP原則は再利用できる。
旧module7件のblobは84e021b6と560c6092で一致した。

ただし、CellModel.applyはprod限定、plan_productionは旧A1/global bank限定、
confidence_cellsは旧4方式family、scv224.Driverはtrain/prod固定・cal拒否である。
将来必要なのは、**SCV全層のcal dispatch、明示family、層別boundsを持つplan wrapperと全費用保存**。
統計core・回帰法・管理システムの重複開発は不要。
これらの薄い接続は今回実装しておらず、実H4で保証付き完走できることを確認した意味でもない。

## 6. RPE全体へ接続する未検証事項

既存RPE v3はunit radius・有効partial-S2固有状態を仮定する。
元Hのexact groundから自動的にその条件は導けない。
訂正後の認証条件はstrict $\beta_{\rm RPE}<\pi/3$で、数値入力域$\leq\pi/2$とは別。
同じenergyへの全round位相関係、正半径、PF/finite-law/selected bias、
normalization、alias-free interval、全roundのtrain/cal/production費と返却を確認する必要がある。
現RPE APIはSCV校正N・層別価格・pre費・cal alpha・一般半径を受け取らない。
固定m0でSCVが有利でも全PR＋RPE優位は自動ではない。

## 7. 224-shot完了後の最小判断

受領項目は、exact $\nu_1$と実現出現count、X/Yのactual係数・rank・range、
raw/residual sample variance、paired covariance、all-equal/低count、
実現RZとtrain/eval内訳、classical setup/engine/compile/wall/RSS、STOP・未返却・未価格tail。
これらは[再利用表 §6–7](SCV_REUSE_AND_VALIDATION_MAP.md)で既存serializerへ対応付けた。

identity/guard/全tailが整い、fresh h1残差に追加低下の兆候があり、
rangeとpre費を回収する余地があれば、独立cal＋fresh評価の限定scopeを検討する意味がある。
相関の兆候が乏しい、低countで未判定、range/cal/floorまたは学習費が支配する場合は、
追加検証を保留・縮小・見送る判断材料になる。
少数pilotから無相関を証明せず、新しい有望性閾値・GO/NO-GOは採択しない。
224-shotの再実行、結果の自動取得・解析、次科学runは行わない。

## 8. 今回実施していない作業

実H4の新計算、保存値の数値再解析、新しいmean/variance/RZ算出、Born評価、sampling/RNG/replay、
回路build/compile、state solve/engine再認証、新control・科学用実装、cal/production、
新統計定理の非自明な証明、詳細GPT/Claudeレビュー、C1凍結解除、科学的正式採択は全て未実施。
既存人工テストはSOURCEを読んだだけで、今回は実行していない。

## 9. 保全・公開・mandatory STOP

- 独立worktree：/tmp/o2-scv-finite-confidence-20261107。
- 専用branch：codex/o2-scv-finite-confidence-20261107。
- base：560c6092bfb5e71af8eedc4a4be5416922764f98。
- 元workspace main HEAD：0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf。既存dirty/untrackedを変更する操作なし。
- 変更・公開対象：本directoryの5 Markdownのみ。旧main、科学SOURCE、raw、結果、grant/ledger/marker/STOP、
  manifest、研究概要・研究ノート・既存索引は変更しない。
- 224-shot側のSOURCE/設定/入力/stream/cache/output/journal/manager、job/process/worker/lock/markerに
  変更・操作なし。進行状況の観測・待機、途中結果の取得・再解析もなし。
- 科学runの既存one-shot/retry制限・mandatory STOP・C1凍結を維持する。
  新しい承認証明書・grant・共有資源制限・管理基盤は導入しない。

本調査の検査は文書のリンク・固定SOURCE所在・数式転記・変更範囲・公開bytesの技術照合に限る。
本directory以外のtree差分がないこと、公開branchのSHAと5本文/blob/treeの一致を確認して停止する。
最終公開commitのSHAと照合結果は最終chat報告に記す。
5成果物の完成・公開・取得照合を終えた時点で**mandatory STOP**。
後続224-shot結果が公開されても自動的に取得・解析して次段階へ進まない。

[prep]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/560c6092bfb5e71af8eedc4a4be5416922764f98
[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/84e021b6a8decde7fa3ed3fa44d7148cc5428061
[theory]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8d8a290fa4d4cf24f968aea27953ad5422e418b3
[n1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/744e1a09661ebba6841fe5faf456f88c41c47780
[n2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5e9bd08068738cdf1070bdc43bee8194b113ae33
[engine]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/fdf00cee3c87d804845497afc89ac6a71f3b3f59
[pilot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3c8e55a2728f86a1eda17a2e0105050a637e7c86
[expected]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a53c1d6eedea0d85fb484425db9012aa89738129
