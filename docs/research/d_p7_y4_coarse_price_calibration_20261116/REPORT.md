# p7／標準Yoshida4・32h/5：幅固有較正、q80一致、A_SHAPE条件付き費用

**軸別に新しく較正した比較用価格は X/Yとも `C_a(q)=307+16539q RZ`。m9/q80の独立取得は各1,323,427 RZで凍結予測と完全一致した。保存Q3適合A_SHAPEの条件付き総費用は372,035,410 RZ。指定3基準を条件付き値では下回るが、総額の75.400727%はq>80への未検証外挿で、価格誤差上界・CI・正式G/winnerはUNKNOWN/null。** 新規6 compile完了、信号/Q3/他配分の追加は0、公開後mandatory STOP。

保存linear H4/1.00 Å/STO-3G/tol-only DF10、一体項＋fragment0..6保存順のB0I p7、標準Yoshida4、PF要求幅32h/5、h_RPE=.1、P0除外。元configurationは `dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`。scalar c_H+tail identityをtotal Tへ一回、symmetric_directional制御、測定付きfull controlled Hadamard X/Y、元native/loweringとQiskit1.3.0/rz,sx,x,cx/opt1/seed17を保持した。

## 較正と事前凍結したm9予測

| role | m | T | q | actual幅 | X compiled RZ | Y compiled RZ |
|---|---:|---:|---:|---:|---:|---:|
| CALIBRATION | 5 | 3.2 | 5 | .64 | 83,002 | 83,002 |
| CALIBRATION | 6 | 6.4 | 10 | .64 | 165,697 | 165,697 |
| 独立HOLDOUT | 9 | 51.2 | 80 | .64 | 1,323,427 | 1,323,427 |

両軸を個別compile・guardした。価格が同じことを理由に片軸を省略していない。CALIBRATION m5/m6の4軸だけをCの既存 `price_model_validation.record/fit/predict` へ渡し、軸別exact二係数を解いた。alpha_X=alpha_Y=307、beta_X=beta_Y=16539。早期m0..m4はfitへ渡さず、m9もfitへ戻していない。

**係数とm9予測は `d4ae4c408fb2218c041e2a24aa04c1c2d42df629:FROZEN_MODEL.json`（本ディレクトリ）へ保存・commitしてからm9を開始した。** 凍結時点でholdout出力namespaceは存在せず、HOLDOUT JOB_METADATAも同commitを指定する。CALIBRATIONのC受入公開は0e28adbd9、HOLDOUTの科学実行SOURCEはd4ae4c408。

| 軸 | 凍結予測RZ | 有効な観測RZ | 観測−予測 | 符号付き相対差（観測分母） | 結果 |
|---|---:|---:|---:|---:|---|
| X | 1,323,427 | 1,323,427 | 0 | 0/1 | EXACT_MATCH_AT_Q80 |
| Y | 1,323,427 | 1,323,427 | 0 | 0/1 | EXACT_MATCH_AT_Q80 |

新しい許容差を採択しておらず、旧m6/m9の2%規則を今回へ移していない。結果はこのq80点での一致であり、連続するq範囲やq640での誤差保証ではない。frozen modelの `qualified_for_candidate=false` と長q誤差UNKNOWNは保持する。

## 保存10軸の再利用と価格の適用範囲

plan d01c780d0の元receipt・trial・既存C受入へ、各m/axisを個別に対応付けた。新候補とのphysical context、T/q/actual幅、signed structure、scalar、measurement、full wrapper/P0、compiler、native/lowering SOURCEを照合した。

| m | q | actual幅 | X/Y各軸の保存RZ | 用途 |
|---:|---:|---:|---:|---|
| 0 | 1 | .1 | 16,846 | early直接価格のみ |
| 1 | 1 | .2 | 16,846 | early直接価格のみ |
| 2 | 1 | .4 | 16,846 | early直接価格のみ |
| 3 | 2 | .4 | 33,385 | early直接価格のみ |
| 4 | 3 | 16h/3 | 49,924 | early直接価格のみ |

m3の2軸はe2179f68の旧native producerで、既存C受入3b1301738/1002f1dfeの同等性証拠を再利用した。他8軸のaf284 SOURCEと同じSOURCEへ書き換えていない。元receipt cohortと、元trialにcohortがない場合のnullも区別して保存した。compiler hashやRZの一致だけでSOURCE同一性を認定していない。

26軸は直接16軸（保存10＋新規6）とモデル10軸。モデルは同actual幅.64のm7/q20・m8/q40・m10/q160・m11/q320・m12/q640だけに適用した。m7/m8も未観測のモデル参照であり、直接証拠ではない。m10..m12はq80を越える未検証外挿。現幅の閉包証明・細幅S2規則・Sの上下界を流用していない。新QPY監査も行っていない。

## 適合済みA_SHAPEの条件付き費用

Q3は `8855f420a9fe2577efc77f099c5a39ab0cb9b7e6:docs/research/d_p7_y4_coarse_a_shape_q3_20261116/C_RESULT_JOIN.json` を引用した。新Q3は0。allocation ID `B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01`、X/Y各軸 `[37,34,32,29,27,24,22,19,17,14,11,9,6]` を変更していない。

| 部分 | 軸数 | N×単価の和 RZ | 総額割合 |
|---|---:|---:|---:|
| 有効な直接価格 | 16 | 56,433,222 | 15.168777% |
| 凍結モデル参照 | 10 | 315,602,188 | 84.831223% |
| **合計** | **26** | **372,035,410** | **100%** |

モデル部分のうちq>80の6軸は280,517,404 RZ、総額の75.400727%。各軸のN・q/T/actual幅・単価・N×単価・physical/canonical ID・直接/モデルの区分・元receipt/元trial/frozen modelへのlocatorを [PRICE_AND_COST.json](PRICE_AND_COST.json) に保存した。Fractionによるexact集計で、表示百分率を使って費用を丸めていない。

| 今回固定した比較先 | 比較先RZ | 粗幅A_SHAPE総額−比較先RZ |
|---|---:|---:|
| 同じA_SHAPE・現幅Yoshida4 | 444,807,010 | **−72,771,600** |
| 現幅p7/Yoshida4の最安適合点 | 435,641,568 | **−63,606,158** |
| 指定されたp7全体の最安適合点 | 395,962,678 | **−23,927,268** |

比較値3件は実行前PLANに固定したユーザー指定値で、他担当の後続成果に合わせて途中変更していない。第1行は同じ整数配分での幅変更、第2/3行は別配分/公式を含む固定基準への比較であり、差全体を幅だけの効果へ帰属しない。今回の値はモデルに条件付けた費用で、保証された達成費用や最適性・正式winnerではない。価格誤差上界/CI、十分N/global RMSE/G_qualifiedはnull。

## 技術確認・初期化復旧・資源

既存m0 native child/worker、fast context/compose部品再利用、canonical fingerprint/cache、元compiler、元7作用gateとC guardを再利用した。cal2 cellは2 workerで並列、m9は係数凍結後の1 cellでX/Yを個別取得。CPU/RAMの恒常的上限や共有設定変更なし。元作用gateの通常1e−9、compiled-vectorの1e−7を緩めていない。

技術検査は最終11件PASS（metadata/人工9＋実worker経路の人工2）。元S2 callerとm0/m1/m2の既定scopeを保ち、新phaseのscope callback・stage別evidence pathを検査した。分子入力を使う新しい技術compileはなく、人工wrapper価格はmock、実H4の科学compileは6件だけ。primary agent自身の検査で、独立レビューとは呼ばない。

初回2a8f861ecのCALIBRATION run01は、sparse checkoutで固定reference JSONが未展開のためmodule import時に停止した。H/state復元・prepare・回路build/compileは0だった。元Git blob59f21d1aを同pathへ復元し、旧run01/UNKNOWN/STOPを保持して、同じ最大6 compile許可内の未取得分を新SOURCE7f7622a94・run02で継続した。新しい数値条件や既存値のretryはない。実際のnative依存importと原blob完全一致の事前検査を追加した。[INITIALIZATION_RECOVERY](INITIALIZATION_RECOVERY.md)参照。

CALIBRATIONのC bindingでは共通helperのevidence pathが旧namespaceを指す不具合を検出した。原unbound record/価格/回路はそのまま保持し、bound sidecarの4参照だけを実際の公開proofへ結合し、無変更のC guardを通した。新callerはstage別pathを明示し、HOLDOUT側のpath補正は0。数値gateの緩和・再compile・自動refitはしていない。

| phase | 科学SOURCE | worker数 | phase wall | 最大worker peak RSS |
|---|---|---:|---:|---:|
| 較正m5/m6 | 7f7622a944a6c6b446b959cec54f0cb48ecc9f08 | 2 | 172.276 s | 704,888,832 bytes |
| m9 HOLDOUT | d4ae4c408fb2218c041e2a24aa04c1c2d42df629 | 1 | 1,222.373 s | 3,419,529,216 bytes |

両phaseのchild/worker終了code0、signalなし。scalar/preparation/fragment順のcontext gateと全7作用gateが成立し、C guardは4＋2軸を受入れた。元のstage logと終了code/signalを保存し、periodic faulthandlerやprivileged debuggerは使用しない。m9の主な待ち時間はcompile/archiveと複数のstatevector照合。別幅・別qのwall比を高速化倍率とは呼ばない。

§10.5は [PREFLIGHT](PREFLIGHT.json)。既存coverage、旧ID carrier/catalog、全late16軸の同物理点、元trial/receipt、関連D/B/A後続metadata・同ユーザーjobを差分照合した。確認範囲で重複取得は0。未参照host/archive全体の不存在は主張せず、他担当のprocess/worktreeに干渉していない。

## 公開SOURCEとSTOP

- 計画/初期SOURCE：2a8f861ecbf89fec75f762a0cb27389c3317183d。
- 較正科学SOURCE：7f7622a944a6c6b446b959cec54f0cb48ecc9f08。raw：e59329ddb49f3a8a886601ca6e38eb4275b780d6。C受入：0e28adbd9580d37ae3f7a7063c8857477ccd7933。
- 係数/予測凍結およびm9科学SOURCE：d4ae4c408fb2218c041e2a24aa04c1c2d42df629。
- m9 raw：0fb16570382cf36e0e1472c93e7806c378df8ea1。C受入/費用projection code：51c94af414e3171b5ee37cf6a8891afc858d67b3。
- **C向け機械可読結果：`c7c8c1edd0996ad9ef97629961f67ca508a182f0:docs/research/d_p7_y4_coarse_price_calibration_20261116/PRICE_AND_COST.json`**。

ロードマップ差分は「粗幅Y4/A_SHAPEに幅固有較正4軸・q80直接2軸・条件付き費用1点を追加。q640価格誤差はUNKNOWN、正式G/winnerは未認定」。本体は編集せずHANDOFFで統合担当へ渡す。旧SOURCE/input/raw/凍結モデル/STOP/KEEP/grant/ledger/markerを保全し、commit/push・remote SHA/tree/成果物照合後 **mandatory STOP**。追加信号/Q3/compile/fit/他配分/N最適化へ進まない。
