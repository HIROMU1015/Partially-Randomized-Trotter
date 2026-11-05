# 方法差・強いbaseline・既存重複

2026-11-06 JST。文献ID・直接確認範囲は [SOURCES.md](SOURCES.md)、概説は [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md)。本書は限定された方法対応の整理であり、新規性の正式判定・研究採択・新しい同値性証明を行わない。

## 1. 同じtaskを比較する条件

全方式が同じ元HのE₀を目指し、同じ厳密基底状態アクセスを持ち、状態準備費を除く。候補だけに真のE₀、gap、exact bias、最適なpriceを与えない。表現を変える場合の状態・sector・既知scalar/phaseの対応を明示する。近似Hamiltonianのground stateを与えた保存実験は、この条件を満たす結果として自動採用しない。

PF次数、split、r/K、β/α配分、compiler最適化は、適用可能な範囲で対照側にも再最適化を認める。同じL_Dの値でも、Pauli項数とDF fragment数、generation順とweight順は別である。classical preprocessing、normalizer・sampling lawの取得費、cold/hot cache条件を双方でそろえる。

成功条件はRMSE、固定時間複素信号誤差、エネルギー推定の失敗確率を区別する。RZ/CX/T/depthの比較はそれぞれ独立に行う。本調査では新しい数値条件も実験も設定しない。

## 2. 近接方式との対応

| 改善方向 | 最も近い既知方式・同じ部分 | 同一と断定できない部分／残る問い | 強いbaselineへ与える機会 | 既存台帳との対応 |
|---|---|---|---|---|
| 表現・係数分布・splitを総費用で選ぶ | PR L01、UWC L03。norm単独でなくdeterministic費とrandom burdenを交換 | native whole-wrapper、finite-RTE、元H状態、誤差情報取得費まで含む有効条件。UWC全frameworkとの完全同値性は未判定 | UWC適合変換、collected Pauli/DF、同精度の強いPF・discard+scalar対照 | RQ-01/02、IDX-REP-08/09、SUP-C-* |
| mixed S/P辞書、反可換pair | L01 §VII、cost-aware L07、unitary partitioning L17 | production router/selectorの実用性とN資格。pairのoperator生成中核は旧限定照合で既知対応済み | 同じS/P/pair pool、同じcollect・既知selector・同じprice情報 | MIX-*、I-METHOD。追加pair探索を新候補として再提案しない |
| basis共有・Givens・diagonal-only control | 既存builder、Gaussian/Pauli compiler、L07/L19 | 同じphase-preserving共有後に残る独立構造。単なるW†W相殺は共通 | actual adjoint、global phase保持、support/full共有、D境界fusionを全armへ | RQ-03、IDX-CIR-01/02、MD-CONTROL。P-Aのinterval差なしを保持 |
| 高次PF・processing・near-integrability | SPRINT L02、PR L01、既存高次PF/B-M/BCH | 原H stateに対する位相/半径、長round、追加gateと誤差情報費の成立条件 | 同じ次数family、既知processed/compact対照、同情報の再最適化 | ADD-PF、BF/BM、IDX-OPT-04/05。既知係数の再命名を除く |
| resource-aware IS | L04のp/√c、L05のh/C、全path IS/Parent配分 | 目的が異なるlawを比較するための全trial finite-confidence・生成アクセス・非加法価格 | 同じcost/moment情報、whole-path law、cache、既知IS/KKT機会 | RQ-04/05、ALLOCATION、SAMPLING-FREEDOM。新lawの式だけを方法差にしない |
| 相関slice・adaptive sampling | PRHS L22、qSHIFT L23、既存相関audit | controlled first moment・符号weight・finite bit・古典L^r work・全roundの適合 | 同じslice/history情報、符号/normalization/取得費を含む既知対照 | IDX-RB-18、I-TRACKB。旧表示留保とSTOPを保持 |
| 確率的角度合成・biasをshotへ交換 | PR L01のrounding-to-residual、TE-PAI L08、TETRIS L10、SPRINT L02 | real finite RTEとnative合成を接続した保証付き全wrapper、長roundでの負担 | 同じ角度辞書、precision、actual adjoint、既知PAI/RUS等の適合処理 | RQ-05、SP/R1、D1–D3、Parent/S4。旧toyだけで一般化しない |
| round別表現/split | UWC L03はround依存表現を明示。repositoryはr/K/β/α配分済み | 複数の近似signalが同じenergy/branchを推定する条件と、同情報の固定構成対照後の利益 | 最適化した固定表現＋既存round別r/K/配分。候補と同じ誤差情報 | RQ-06、SUP-RPE-*、M1-S、研究C。着想だけでは方法差なし |
| control overheadをreferenceへ移す | SE-QPE L11、既存diagonal-only control。L20はlocal-model条件の周辺対照 | pathごとの共通固有基底・真空reference・補正位相とPR平均signalの整合 | 既にcontrolを軽量化したDF/Pauli回路。SE側にもCSWAP・register・routingを計上 | RQ-03/06/07。指定調査集合ではSEのPR接続証拠未確認 |
| signalを感度最適化する | QSP-PE L12、既知adaptive PE | ideal unitary/query保証からfinite-RTE/compiled価格への接続。RPEと異なる推定器 | 同じoracle誤差・入力state・失敗確率と、最適RPE配分 | RQ-06に隣接。採用には推定器familyの選択が必要 |
| connectivityによる方式選択 | L19のswap network、既存routing/compiler、L11のregister間干渉 | 同graph/layout/探索費後に残るPR固有効果 | 全方式に同じmapping・routing budget・compiler機会 | RQ-07。証拠未確認は世界的未探索の意味でない |

## 3. 同一視できない保証・会計

| 文献・保存資料の量 | 本研究で追加して確認すべき接続 | 誤った読み替え |
|---|---|---|
| channelのdiamond距離 | 制御ancillaを含む実装とcoherent first moment | system channelが近いのでHadamard位相も近い |
| 平均振幅の正scalar補正 | raw振幅、corrected estimatorのrange/moment、必要N | 位相不変なのでnormalization費用もゼロ |
| oracle query complexity | controlled evolution単価、合成、wrapper、random lawを含む実回路 | query数削減率がRZ/T総費用削減率 |
| 固定Tのsimulation/信号精度 | 全時間範囲、共通energy target、alias/branch、全round N | 短roundの低Gで全RPEも低G |
| energy RMSE資源model | 指定失敗確率の保証とdecoder条件 | RMSEをそのまま成功確率とする |
| DMRG/CCSD(T)/保存exact stateからの経験的bias情報 | 実際に取得可能な情報とその費用、必要ならrigorous allowance | 真のE₀/誤差情報の無料oracle |
| native primitiveのoperator誤差 | ordered path/全step、control、weight、全lawでの蓄積 | primitive資格からfull-wrapper/energy資格を推定 |

## 4. 保存結果が課す反証・境界

以下は [固定統合台帳](../pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv) の引用であり、今回の新評価ではない。性能の改善率・費用は再計算していない。

| 元記録・保存条件 | 維持する到達点 | 再提案から除くもの／未解決点 |
|---|---|---|
| IDX-CIR-02/P-A：元forced-support order-2、固定fragment分離の機構確認 | 一区間対照と同じplan/compiled指標、interval固有claim停止 | intervalを増やすだけの再開。既知basis共有そのものの価値まで否定しない |
| IDX-OPT-03/04/05：P-C/P-D/R3の各元domain | 広域予測破綻、強対照との選択一致、具体的方法差不足を保持 | 同じscore/gridを広げるだけの新研究。全非prefix法の不可能性とはしない |
| IDX-PR-02：固定2qubit toy、tail levels 4/8/16、complex RMSE条件 | tail外挿のbias低減よりvariance/backbone費が支配しSTOP | bias低減だけの外挿候補。全外挿一般の否定にはしない |
| MIX-F2：4mode/2粒子/occupied(0,1)、2残差fragment、L_D=1、T=1/5、q=1、δ=1/5、r=2、h=1/10、K=0、topology-free Qiskit 1.3/opt1/seed17 | 修正版registered operator/signal・単回資源成功、N/weighted costはnull | mixedは常に勝つ、同精度全費用が評価済みというclaim。分子geometry/AO basis/分子DF rankは非該当 |
| MIX-CLOSE/PS48：同合成fixture・固定recipe subset | singleton局所改善/悪化、一般selector未確立、正式一時終了 | 別名称のselectorや新namespaceでの自動再開 |
| C1-PAUSE：linear H4 1Å/STO-3G、legacy DF12、generation-prefix L_D=3、T=4/5、q=1、r=4、m=5、h=1/5、tail-only+scalar | logical path完了とnative/共通N/full-wrapper不足、一時凍結 | G6数学構成やAPI correctnessから分子総資源優位を導くこと |
| SAMPLING-FREEDOM：固定静的surveyと人工test | NO_DISTINCT_FREEDOMは限定探索の結論 | IS式/自由度を変数追加して新法にすること。新分子samplingの証拠ではない |
| SUP-C-064/WORD：H4 1Å/STO-3G、number-CDF8、L_D=2、δ=h=.1、r=32、K=0、m=0..12、own-DF ground、Qiskit1.3/opt1/seed17 | nominal価格、固定wordの数値correctness。全law/actual CAは未資格 | 元物理Hの厳密基底状態条件を満たす証拠への無条件移送 |
| SUP-RPE-V3R1/ENERGY：H4 1Å/STO-3G、legacy DF12、L_D=3対12、δ=.02、K=2、round別r、18round/36軸、直接compile q≤32と長q外挿を分離 | 条件付き成功・energy接続を保持し、正式status UNDETERMINED | 未着手扱いと最終優越扱いの両方を避ける。実装全体/true molecular energy/最終費用は別 |

## 5. 研究機構と通常実装の区別

mixed router、native wrapper、既知IS、SE gadget、既知QSPを追加するだけなら、まず既知法の接続・実装課題である。未実装という事実だけでは研究上の新規性にならない。

一方、同情報・同target・強対照で、例えば生成アクセスの複雑さ、非加法なwhole-path価格を利用できる条件、長roundでも保つsignal保証、構成変更間のenergy接続という**追加の性質**が成立すれば、既知要素の組合せから研究上の知見を得られる余地がある。その性質が成立するかを本調査で判定していない。問いと必要な証拠は [RESEARCH_OPPORTUNITIES.md](RESEARCH_OPPORTUNITIES.md) に渡す。
