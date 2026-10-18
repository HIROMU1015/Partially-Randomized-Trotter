# Track A V1：決定論的打切り対照強化後の限定科学レビュー

- **日付**：2026-10-16（JST）
- **レビュー状態**：完了（限定された科学的解釈と次の検証選定）。全候補の実コンパイル費用・厳密誤差保証の認定ではない。
- **位置付け**：GPTによるLevel 2の詳細科学レビュー。ユーザーによる「レビューを開始して」を開始承認とする。
- **対象**：`HIROMU1015/Partially-Randomized-Trotter`、H6、固定DF Hamiltonian／state、V1のPhase A・Phase B・再会計。
- **実施範囲**：GitHub公開済み固定commitの一次報告、比較CSV/JSON、軸費用receipt、Phase A/Bの技術検証、旧対照結果、既存Claude Level 1レビューを照合し、数理的・科学的な解釈と最小追加証拠を検討。
- **実施していないこと**：新SCF/DF/state、signal/reference、trajectory、build/compile、保存値の新規再会計・再ランキング・数値最適化、repositoryの変更。独立した全raw再現、全4,320行の外部再実行、母平均の認証も実施していない。
- **権限**：研究判断の提案のみ。新たな科学計算、grant発行、STOP解除、正式科学registry変更、原稿改訂は承認しない。

## 1．要旨と今回の判断

**主結論：決定論的打切り対照のprefix格子を細分化した結果、旧登録集合でのPR優位の大きさは過大に見えていた。中間精度の固定モデル条件付き点費用ではPRに約31–35%の利益が残る。一方、緩い精度ε=0.05ではB0+Iが優位に転じ、厳しい精度ε=0.0005のPR利益は約10%に縮小した。**

この結果は「PR一般が不利」「PR一般の優位が証明された」のいずれでもない。研究の中心を、**有限時間complex signalにおいて、強い決定論的打切り対照とPRの優劣がどう切り替わり、その費用を回路圧縮・normalization・bias headroomがどう決めるか**、という条件付き適用領域の評価に置くことを支持する。

科学的に新しいのはPRアルゴリズムそのものではなく、**比較対照の探索分解能とbias/shot会計が実compile資源上の結論を左右するという帰属・検証**である。最終論文で新規性を主張するなら、この条件と限界を明示することが必要。

**主張の確度**：

1. **高い**：登録したsignal、Phase Aの数値一致、Phase Bの登録S4実測4軸での局所2%未満PASS、保存比較規則の機械的整合、強化による条件付き点費用の変化。
2. **条件付きで支持**：追加候補のモデル予測費用とPR32標本平均を組み合わせた中間精度の約31–35%利益、ε=0.05の逆転、ε=0.0005の約10%利益。
3. **未認定**：最良の新決定論的候補の実compile費用そのもの、全候補でのモデル誤差保証、bias非相殺条件での全domain winner、母平均PR winner、truth-free候補選択、一般系・長時間・FT/QPE資源への拡張。

## 2．レビュー対象の固定条件と費用の意味

| 項目 | 固定条件 |
|---|---|
| 物理系 | linear H6、隣接距離1.00 Å、STO-3G、12 spin modes、物理sector400 |
| Hamiltonian | 保存されたtol-only DF、rank19、signed generation order、保存state／reference |
| 時間 | T=0.8 a.u.、q=1,2,4,8 |
| 要求精度 | 複素有限時間signal `z(T)=〈ψ|exp(−i H_DF T)|ψ〉` のε=.05/.01/.005/.001/.0005。Ha単位のenergy精度ではない |
| 費用 | P=0、cosine/sine各軸の解析的十分shot数 `N_a` × 各軸のcompiled measured full-wrapper RZ命令数 `C_a` の和 |
| 規則 | 各軸α=.025、経験的allowance factor1/10、axis headroom、既存整数shot算術順序 |
| compiler | Qiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、symmetric_directional、固定hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` |
| PR側 | 既存確認union8、各32 paired trajectoriesから選ぶ**標本点最良**。探索72候補×8は独立した別scope |
| 決定論側 | B0、B0+I、full S2、full S4。旧32候補→S2細格子96候補→S4付き136候補。適格候補の最小 |
| 費用の来歴 | 旧取得済みとPhase BのS4計4軸は`MEASURED_COMPLETE`。その他の追加候補は固定モデルの`VALIDATED_MODEL_ESTIMATE`。PRは`SAVED_PAIRED_SAMPLE_MEAN` |

記号として、通常の軸別十分shot数は概略 `N_a = ceil(2 B_upper² log(80) / h_a²)`、`h_a=ε/√2−|bias_a|−u_a`。したがって単純な回路単価だけでなく、**biasの符号・相殺とnormalization Bがshot費用を変える**。モデル推定されたRZ命令数は、実コンパイル値と同義ではない。

## 3．確認できた一次資料・出典

**GitHub上で実際に取得できた固定資料**：

- [S1：V1最終再会計結果／commit `25bbdd33fe758a0e730138cb91ddac8c41b0a298`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25bbdd33fe758a0e730138cb91ddac8c41b0a298/docs/research/track_a_v1_reaccount_results_20261016.md)。報告、結果scope、数値的限界、保全、STOP。
- [S2：V1全比較CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25bbdd33fe758a0e730138cb91ddac8c41b0a298/artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/comparisons.csv)、[比較JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25bbdd33fe758a0e730138cb91ddac8c41b0a298/artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/comparisons.json)、[axis費用・実測/モデル区分](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25bbdd33fe758a0e730138cb91ddac8c41b0a298/artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/axis_costs.json)。公開値を参照し、全計算を新たに再実行していない。
- [S3：Phase Aの104 signal・8診断結果／commit `bf865de1efae51ce1663c878ad1e9daa5bb915c2`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf865de1efae51ce1663c878ad1e9daa5bb915c2/docs/research/track_a_v1_phase_a_run02_results_20261016.md)、[保存bias・適格性CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf865de1efae51ce1663c878ad1e9daa5bb915c2/artifacts/resource_applicability/track_a_v1_phase_a_run02/2026-10-16/signal_bias_eligibility_v1.csv)。
- [S4：Phase Bの登録S4全4軸実測／最新V1 commitから](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25bbdd33fe758a0e730138cb91ddac8c41b0a298/docs/research/track_a_v1_phase_b_run01_results_20261016.md)。
- [S5：V1再会計実行前の比較・技術契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25bbdd33fe758a0e730138cb91ddac8c41b0a298/docs/research/track_a_v1_reaccount_readiness_20261016.md)。
- [S6：旧H6 B0+I full family結果／commit `34efc914b2f485ec694c7d6016025d5c76829be6`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)。
- [S7：既存H6/H8理論費用モデルA/B結果／commit `7e41670565d9284da9aded94063be591fdcaa3dd`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e41670565d9284da9aded94063be591fdcaa3dd/docs/research/track_a_native_df_theory_evaluation_20261016.md)。これにはV1細格子の新候補は含まれない。
- [S8：2026-10-15のR10統合レビュー／commit `3d910ee1de4597b8895e8506500553dbad5a5085`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)。
- S9：ユーザー提供の添付 `レビューpr.md`（Claude独立Level 1レビュー、2026-10-16）。第4・8・10・11節と付録B7/B8を読んだ。GitHub上の固定commitに保存済みとは認定していない。

**未実施・未取得の区別**：新しい決定論的最良候補の実コンパイルは未実施。元のprivate raw・全軸回路・全環境の独立外部再現は未取得であり、保存成果に対する主担当検証を独立再現と呼ばない。費用以外の全6指標とP感度について、新しい細格子全体の比較結果は今回の資料から認定しない。

## 4．主要結果：対照を強化すると何が変わったか

### 4.1 5精度の点費用削減率

[S1, S2]で報告されたallowance factor1、通常signed signal bias、`100×(1−G_PR/G_det)`をそのまま引用する。

| ε | 旧32候補 | S2細格子96候補 | S4追加後136候補 |
|---:|---:|---:|---:|
| 0.05 | +29.35% | −13.04% | **−13.04%** |
| 0.01 | +49.69% | +30.89% | **+30.89%** |
| 0.005 | +53.27% | +34.97% | **+34.97%** |
| 0.001 | +56.14% | +31.39% | **+31.17%** |
| 0.0005 | +50.37% | +15.09% | **+10.03%** |

**科学的含意**：今回の変化の主因は**打切りS4ではなく、決定論側のS2 prefix格子の細分化**。S4追加による最良決定論費用のさらなる削減は、ε=.05/.01/.005では0%、ε=.001で0.322%、ε=.0005で5.617%と報告される。初期の大幅なPR利益を、そのまま「十分強い決定論対照に対する優位」として使えない。

allowance factor10でも優劣の符号と最良候補は一致した。ただし、これは経験的uの一種類の感度であり、厳密な誤差保証ではない。

### 4.2 新しい最良決定論的候補

| ε | 新最良候補 | G_RZ（公開丸め値） | 費用来歴 |
|---:|---|---:|---|
| .05 | B0+I p7/q1 S2 | 3.56889×10^8 | **モデル推定** |
| .01 | B0 p11/q2 S2 | 2.82872×10^10 | **モデル推定** |
| .005 | B0 p11/q4 S2 | 1.56588×10^11 | **モデル推定** |
| .001 | B0+I p11/q2 S4 | 7.47720×10^12 | **モデル推定** |
| .0005 | B0+I p11/q4 S4 | 3.62659×10^13 | **モデル推定** |

PR最良は旧確認union8/n32から変わらず、軸費用は保存済みcompiled標本平均。**両者の列を「双方実測」の比較表として提示することは不可**。決定論側は新しいsignalを実測数値計算で取得したが、回路費用は5候補とも未実測である。

## 5．費用逆転の数理的解釈

### 5.1 ε=.05の逆転は何を意味するか

[S2]の同一比較レコードと軸費用から、次の**保存値**が確認できる。

| ε=.05の候補 | 1軸RZ/回路 | 両軸合計shots | 総RZ費用 | 1軸RZの来歴 |
|---|---:|---:|---:|---|
| B0+I p7/q1 S2 | 14,524.2 | 24,572 | 356,888,642.4 | 固定モデル推定 |
| PR B2 p5/q1 r4/K4 | 13,270.5 | 30,400 | 403,423,200 | 32 trajectoryの保存compiled標本平均 |

**PRの回路単価は低いが、必要shotsは高い。** したがって、単回路の圧縮率が低費用を保証するわけではない。決定論側はB=1であり、bias/headroomに応じてshot数が変化する。この逆転は、`G=N×C`の費用定義と整合する。

新しいp7の保存signed axis biasはreal約−0.006065、imag約−0.010647 [S3]。こうした固定時間・固定stateでの誤差は有効なbenchmarkのデータであるが、複数候補から小さいbiasを持つ点を後から選ぶと、別geometry・別timeでも同じ相殺が起きるという一般性はない。

旧B0+I p10/q1では保存shotsが28,141で、新p7/q1の保存shotsは24,572 [S2]。**前回Claudeレビューの「候補を細分化しても決定論shot数は下がらない」という仮定は、今回の保存結果では成立していない。**

### 5.2 前回Claudeレビューの条件付き予測との関係

[S9]第4.4節の「中間精度で少なくとも32～40%程度」という見積もりは、`p*=11`と決定論shot数非減少等の前提を置いた**条件付き算術**であり、定理・保証下界ではない。V1後の数値はε=.01で30.89%、ε=.005で34.97%、ε=.001で31.17%であり、その値を下回る。

この差をClaudeの分析全体の否定とする必要はない。むしろ、Claudeが指摘した「prefix格子と符号付きbias相殺が未解決」という科学的問題が、V1で重要だったことを示す。ただし、**32～40%を保証付き下限として引用することは取りやめる**べきである。回路圧縮とnormalizationの近似式はbias headroomの補正が小さい範囲にのみ使い、一般式では軸別のheadroomを省略しない。

### 5.3 格子・bias・shotの三者を区別する

Claudeの「約2 fragment差でPRが有利」という断片換算閾値は、同程度のbias・shot headroomという近似条件下での目安である。今回のp7とPR p5の差は2断片だが、shotsが異なるため、実費用の符号をこの簡約閾値だけでは判断できない。

より適切な科学的整理は：

- **回路圧縮**：prefixを短くしたPRの1回路RZ費用の低減。
- **有限RTEのnormalization**：PRの`B`がshot数を押し上げる効果。
- **誤差相殺とheadroom**：決定論打切りと外側PFのsigned biasが候補ごとに変わり、必要shot数と最良選択を左右する効果。

この3つを混合せず、既存のaxis別NとRZの関係から説明する。精度範囲の式を作る場合も、まず元の軸別費用定義を正本とする。

## 6．信頼性と未解決の科学的問題

### 6.1 Phase A/Bの技術的な根拠

Phase A [S3]：104追加候補のsignal・bias・適格性とPR外側bias8診断を取得。native/independent signalの最大差は約1.53×10^-13、primitive誤差はより小さい。これは保存モデルに対する数値検査であり、全体の近似誤差に対する厳密certificateではない。

Phase B [S4]：S4のB0 p11/q1とB0+I p15/q8でcosine/sine計4軸をcompile。固定modelとの相対RZ差は、それぞれ約0.098%と0%で、登録した各軸2% STOP基準を通過。

**重要な区別：Phase Bの4軸PASSは、ε=.05の新最良であるB0+I p7/q1 S2、また他のp11/q2・q4最良候補の実コンパイル検証ではない。** S2 p7は固定モデルのp10/p15 anchorより小さいprefixへの外挿である。旧p5 pilotが線形式と近いという既存参考値はあるが、SOURCE/旧profileの完全同値・未測定候補の2%上界を認定しない。

### 6.2 符号付きbias相殺と『保守的比較』

[S1]の経験的非相殺診断では、各軸のerror成分の絶対値和を使う。同じB0+I p7/q1で、通常biasの総費用約3.57×10^8に対し、非相殺診断は約4.88×10^9と報告されている。

これは「通常bias結果が偽」という意味ではない。**指定したHamiltonian・state・timeにおけるsigned誤差相殺は現実の数値結果**である。しかし、別条件へ一般化する場合や、referenceのexact signalを使えない実践的な方法選択では、相殺に依存するwinnerは不安定になり得る。

非相殺診断は厳密な数値誤差certificateではなく、旧B0+Iの12候補でsigned分解が未保存であるため、**全比較domainの最良値はUNRESOLVED**。未解決の候補を除いた最小を『保守的最良』と呼ぶことは不可。十分な証拠がないのに「保守的計算でもPRが勝つ」と主張しない。

### 6.3 限られた候補探索と統計

旧PR側は確認8候補・各32 trajectory、決定論側は細格子136候補を比較している。PRの新prefix p3/p4は**exact-tailの外側S2 bias診断のみ**であり、有限RTE law・normalization・費用は未計算。つまり、この比較は**より強い決定論的対照で既存PRを試す保守的benchmark**であって、両手法の全パラメータ最適化比較ではない。

PRの標本点最良は母平均最良ではない。探索時に使った8標本と確認32標本を混合していないことは長所だが、確認8候補間で同じ標本を使ってwinnerを選んだ影響は残る。保存標本SEだけで全候補同時の保証を得たことにはならない。希少Taylor次数の未観測費用の厳密保証もない。

### 6.4 Truth-assisted selectionと実運用

今回の各候補の適格性・最良費用選択には、保存full-H referenceとのsignal誤差を使っている。**これはbenchmark上の事後的なtruth-assisted選択**であり、実機で真のsignalが既知でないときの実装可能な候補選択規則ではない。『同じ精度へ揃えた回路資源benchmark』としての価値を損なわないが、『truth-freeに方法を選択できる』という別の主張は支えない。

### 6.5 SOURCE、compiler、指標、外挿

- H6の旧B0保存費用とB0+I/新nativeの比較にはsource/profileの差が残り、matched B0対B0+Iの純粋な因果的費用差を測定したとは言えない。
- V1の再会計は**P=0、RZ**が主対象。旧gridで複数費用指標・P感度を評価済みでも、細格子の新最良で6指標すべてや同じP境界が成立するとは未証明。
- 理論モデルA/B [S7]の832行は旧H6/H8登録候補についてであり、今回追加されたV1細格子の費用を独立に認証していない。理論費用とcompiled費用はそもそも同一の量ではない。
- H4 geometry、H5/H7限定開殻pilot、H8元native結果を無条件に同一分子サイズ傾向へpoolしない。H8は同種の細格子V1をまだ受けていない。
- finite-time signalのεをenergy/QPE/RPEやFT logical T/Toffoli費用と混同しない。

## 7．今後の主張と論文化方針

### 7.1 今回採用できる中心的な結果表現

> 線形H6・1.00 Å・STO-3Gの保存DF rank19を対象とする、T=0.8の有限時間complex signal推定で、決定論的打切りのprefix格子とS4候補を強化した。登録PR確認候補と固定費用モデルに条件付けると、ε=.01～.001の測定込みRZ点費用に約31～35%のPR利益が残った。一方、ε=.05では決定論的B0+Iの点費用が低く、ε=.0005では利益が約10%に縮小した。優劣の変化は、回路単価だけでなく、bias/headroom・normalizationを通じた必要shot数の違いと関係する。

この文は**計算モデル条件付きの保存点推定**の記述である。全候補が実コンパイル測定済み、普遍的な系サイズ傾向、理論的な優劣保証と読める表現は避ける。

### 7.2 採用しない主張

- PRはH6で**すべての精度で**決定論対照より有利。
- 中間精度での削減率32～40%以上は保証下限。
- ε=.05での−13.04%は実測確定した劣位、又はPR一般の劣位。
- ε=.0005での10.03%は誤差・候補・統計不確実性に関係なく頑健。
- B0/B0+Iの任意prefixのcompiled費用はモデルで2%以内に保証される。
- 実機でexact signalなしに最良方式を選択できる。
- RZ命令数がそのままFTのT-count/Toffoliや総energy推定費用である。

### 7.3 研究としての着地点

主軸はPRの**一般的な低費用実証**ではなく、**決定論打切り対照の探索分解能を上げると評価がどう変わるか、PRの利益が残る領域と負担要因は何か**という比較方法論・有限時間資源の適用領域である。V1のnegative側結果は中心的証拠から外さず、予測モデルの限界も記録する。

研究方向の全面変更やPR-S4/q16／大規模系などへの無制限拡張は現時点で不要。H6の条件付き成果を主として整理し、H8等は政策差と未検証対照を注記した補助例に留める。広い比較を中心主張にしたい場合だけ別の科学判断を必要とする。

## 8．最小追加検証の優先順位と判断条件

### 第1優先：ε=.05での新最良S2 p7/q1の実測

**推奨する最小scope：B0+I p7/q1 S2のcosine/sine各1軸、合計2 full-wrapper compile。** 検証したい量は、固定モデルのRZ推定値14,524.2/axisと同じ入力・compiler・controlled phaseでの実際のRZ。新signal、SCF、DF、state、PR samplingは不要となる設計が可能か、Codexが既存SOURCEとreceiptで確認する。

- **理由**：今回のε=.05の符号反転を直接支えている新決定論候補が未実測で、S2 p7はモデルfitの主要anchorより短いprefixへの外挿だから。
- **実行前固定**：対象2軸、signal/shot再利用の照合、literal controlled phase、profile/source/Input一致、予測誤差基準、最大call/CPU/RAM/wall/output、one-shot grant、mandatory STOP。
- **判定**：予測モデルと実測が事前設定の2%以上不一致、source/phaseが変わる、cost/axisに欠測が出る場合はSTOP。無断再fit、追加候補のcompile、別のSOURCEでの置換をしない。
- **検証後**：採択済みの元signal/shot規則に従う必要最小限の保存値再会計は、**別途数値再解析の実行承認**を受けて行う。レビュー本体では費用の再計算をしていない。

**この追加計算は今回承認されていない。** これを実測できれば逆転を支持するデータの質が上がる。通常biasを用いたモデル条件付きという留保や、別geometryへの一般化限界は別問題として残る。

### 第2優先：中間精度と厳密側の代表winnerの任意検証

論文の主結果で「30.89/34.97/31.17/10.03%」を純粋なcompiled実測値として主張したい場合、モデル推定のままの次候補も別途対象となる。

- 中間精度：B0 p11/q2 S2（ε=.01）、B0 p11/q4 S2（ε=.005）。
- より厳しい領域：B0+I p11/q2 S4（ε=.001）、B0+I p11/q4 S4（ε=.0005）。

ただし初回から全候補を無差別に再compileする必要はない。まず第1優先の結果で費用モデルの妥当性を確認し、**論文でどの精度域の数値を実測値として主張するか**に対応して、対象と新grantを個別決定する。大きな変化が出れば科学的判断へ戻す。費用モデルの精度を理由にtruth-free候補選択を主張しない。

### 条件付き：bias非相殺診断の欠測・H8の対照格子

- 旧B0+I 12候補のsigned分解が既存raw/receiptから回収できるかを**読み取り専用で技術調査**する価値はある。存在しなければ新数値処理の実行承認を別途受けるか、全domain保守的比較を未解決のまま明記して完結する。
- H6で緩い精度の優劣が反転したため、H8の旧粗格子結果を無条件に強化対照後の結果とみなせない。**H8でも強い適用領域主張を行う場合に限り**、H8向けの絞った対照検証を別設計で検討する。H8の全面V1を自動開始しない。
- V2の既存理論832行を新候補のcompiled validationと取り違えない。モデルBはPF/RTEのnative混合reuseをそのまま実装したものではない。V2の残差解析は独立した別承認・優先順位判断とする。

## 9．GO/STOPと次の担当

| 項目 | 推奨 |
|---|---|
| 今回の科学的解釈 | **限定的にGO**。中間精度の条件付き優位、ε=.05の条件付き逆転、強化対照の有用性を報告可 |
| 全5精度での実測勝敗の確定 | **保留**。新winnerのcompiled費用はモデル値 |
| 最優先の次の担当 | Codex：p7/q1 S2 B0+Iの2軸compileの必要性・実行契約準備。新科学計算開始にはユーザー承認 |
| 次の判断 | ユーザー：実測を論文の中心に必要とするか、モデル条件付きのまま完結させるか。実行scope/grantを確認して別途承認 |
| V2 | 既存104候補の理論費用は再計算不要。追加残差解析は別判断であり、V1の結果解釈を妨げない |
| その他 | H5/H7、PR-S4/q16、追加sampling、新Tなどへの自動拡張なし |

今回のレビューを理由に既存科学SOURCE、raw結果、grant/ledger/markerを変更しない。各one-shot後のmandatory STOP、失敗時の自動retry禁止、fresh provenance・資源確認、remote SHA/bytes確認を維持する。

## 10．旧レビューから更新した判断と残存論点

| 従来の見解 | 本レビューの扱い | 更新理由 |
|---|---|---|
| H6の旧登録域でPRが全5精度で有利 | **新対照域ではε=.05で優位でない点推定** | p7 B0+Iを含む細格子で分母が更新された [S1/S2] |
| 中間精度で32～40%以上の削減が残る見込み | **条件付き見積もりに戻し、保証下限ではない** | p11等のshotsが旧候補から変化し、観測値が推定範囲を下回る [S2/S9] |
| 厳しい精度でも50%程度のPR利益 | **ε=.0005では10%程度のモデル条件付き利益に縮小** | S2細格子と打切りS4の追加 [S1] |
| 正しいnative/Qiskit接続は重要 | **維持** | Phase A/Bでの局所照合・実測成功 [S3/S4] |
| 理論費用A/Bは独立した研究の中心ではない | **維持** | 旧登録候補への接続済みだが、今回V1新候補の実compiled値や各winnerの誤差保証は与えない [S7] |

**未解決の中心論点**：最良対照の新compiler実測、bias相殺の一般性・全domain非相殺費用、標本母平均、truth-free候補選択、他系/時間/FTへの外挿。これらすべてを解決しなければ今回の有限H6 benchmarkが無価値になるわけではない。必要な主張の強さに応じて実測・資料を最小化すべきである。

---

**レビュー終了状態**：`SCIENTIFIC_REVIEW_COMPLETE_WITH_DECLARED_LIMITS`。本書は科学的評価と次作業の提案であり、新しい数値結果・計算承認・自動STOP解除ではない。
