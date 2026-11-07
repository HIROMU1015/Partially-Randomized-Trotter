# O2研究成果統合：Sampling・Return Aggregation・測定推定器

2026-11-08 JSTまで。既存の研究報告・理論整理を改善機構ごとに統合した資料である。新しい科学レビュー、数値再解析、手法のランキング、科学的GO/NO-GOの採択ではない。固定SOURCE・結果・REPORTへの入口は[証拠索引](O2_EVIDENCE_INDEX.md)、今回の文書作業の範囲は[REPORT](REPORT.md)。以下の小数・割合はすべて元報告の表示値の引用で、再計算していない。

## 1. O2で何を研究し、どこまで分かったか

O2は、PR-PFの同じ測定taskに対して **必要測定数、1回当たり回路資源、その取得・学習費** を減らすアルゴリズムを扱う。主な操作は、sampling law/weight、finite RTEのreturn集約・parent配分、層別化/CV、量子測定プロトコルの変更である。Hamiltonian表現・分割はO1、PF構成・合成・processingはO3、一般PR/RPE検証と全round資源評価は別担当。過去のTrack B分類を、現在のO2担当範囲と機械的に同一視しない。

得られた成果は二層ある。**形式returnの局所生成、有限bit補正、strict位相を保つnative接続、H4測定pilot、LCU測定アクセスという技術的蓄積** がある。また、固定synthetic条件ではClosed P5やParent＋samplingに限定的なT費用上の利益が記録されている。しかし **実H4で強い層別対照を上回る学習費込みの期待総資源改善、全PR＋RPEでの最終優位は未確立** である。

後続の負結果は、前段の成果を取り消さない。G9のClosed P5を認めるとGeneral Fullの独立した追加利益は限定され、S2では単価低下が十分N増加で相殺された。H4の752-shotはINCONCLUSIVE、SCV224/448では今回の学習費を回収する明確な材料がなく、A2の単純LCUは成功率補正後の測定noiseを減らせなかった。いずれも各条件・機構の限定結果で、O2全般の不可能性ではない。

## 2. 証拠を読む共通規則

最終目的は資源aごとの `Σ_(round,axis) N E[C_full,no-prep^(a)]`。T、CX、compiled RZ、depth、古典秒、memoryを根拠なく合算しない。state preparation費の除外は研究上の範囲であり、候補固有のdilationや追加処理まで無料にする仮定ではない。

| 証拠の層 | 何を意味するか | 今回の主な例 |
|---|---|---|
| 数学的構成 | 明示したclass・仮定内のmean/phase/weight等 | R0/G1/G6、Parent、A1対層別、A2 block |
| 技術的実装 | SOURCE/API/保存/生成経路が存在 | ReturnKernel、B3、selected engine、LCU adapter |
| 人工correctness | 創作した小入力の新接口を確認 | G6形式test、B3受入、A2の9件 |
| 実入力での数値correctness | 登録された入力のoperator/state/保存値確認 | G9/G10 synthetic、C1の選択6event、N1/N2 |
| 有限統計pilot | 固定条件の有限測定recordに基づく診断 | 752、SCV224/448。実機backend測定ではない |
| 誤差保証・十分shot数 | 記載仮定の下でのbias/confidence/予算 | G7–G10/Parent/S2の十分N、旧A1 engine認証 |
| 局所native価格 | 指定provider・取得sequenceのliteral資源 | G9/G10/D/Parent/S2 |
| Full-wrapper費用 | D/basis/control/readoutを含む指定scopeの価格 | H4 pilotのselected full no-prep compiled RZ |
| 全PR＋RPE資源改善 | 同じ最終精度/成功確率で全段を比較 | **今回統合した成果では未確立** |

これらは単純な合格段階表ではない。例えば十分Nのforecastは、その数の実測を完了した証拠ではない。H4 pilotの実現RZは生成した各測定recordへのcompiled価格で、実機でそのgateを実行した支出ではない。source-bound local evidenceを外部独立再現やimmutable CIと呼ばない。[既存総合地図][MAP]・[O2調査][SURVEY]

## 3. 改善機構の系譜

| 機構 | 変更する操作・狙い | 主な系譜 | 到達した問いの境界 |
|---|---|---|---|
| 確率・weightの再配分 | 安いevent/pathを増やし単価と測定負担を交換 | canonical PR、IS/Parent、S2、sampling自由度調査 | 同targetでもM2/range/取得費が増え得る |
| 有限辞書・return集約 | 同finite meanを少ない/安い回転へ再構成 | R0–G5、G6–G10、Closed P3/P5、Parent | strong Closed/CTS、合成価格・normalizerアクセスが重要 |
| whole-pathの価格接続 | event和で失う境界相殺・非加法価格を扱う | D1–D3、S2、J1、C1 whole-path | 局所return queryは価格score部分和を返さない |
| 古典推定器・配分 | 層間/層内の変動を減らす | B1-O/B1-OL、A1、SCV224/448 | Born noise、学習誤差・費用、希少層が残る |
| 量子測定そのものの変更 | fine/coarse差分をcoherentに測る | A2 augmented-state→LCU | 成功率/scale/失敗費を戻すとnoise利益が消えた構成 |

### 3.1 通常PR、IS、cost-aware/whole-path sampling

通常PRは有限RTEのorderとlabelからphase付きunitaryを選ぶ。cutoff Kは最大even orderで、paired polynomial degreeはK+1。finite meanと無限Taylor/厳密時間発展を区別する。同じpath familyのlawをPからqへ変える場合、元weightを含む正しいlikelihood補正でmeanを保つ。実outcomeをY∈{−1,+1}とすると、固定係数aでは `w=a/q`、`M2=Σ a²/q`、rangeはmax|w|である。

既知の `q∝a/√C` は固定辞書の価格×二次momentのleading proxyを扱う。Bernsteinのrange項、bias控除後の残余幅、整数ceil、全stage費、normalizerとonline生成を同時最適化したことにはならない。T=0のeventに正価格用の式を機械的に適用しない。有限bitでは理想確率と実qを分け、support・丸め・weightを実法則へ戻す。[SURVEY][SURVEY]・[既存自由度照合][FREEDOM]

この機構はIS/Euler/線形辞書最適化などの既知原理の応用が中心である。`NO_DISTINCT_FREEDOM` は調べた変更案から別の実装可能機構を見いだせなかったという限定調査で、任意lawのno-goではない。今後は新しい取得構造や測定outcomeの変更を伴わない同じISの再探索を繰り返さない。

### 3.2 初期R0–G5：隣接degree再配分と固定辞書

**目的・仕組み**：同じfinite first operator meanのeven/odd係数を再配分し、normalizationと回転価格を交換する。R0の理想恒等式、R1のnative/十分N地図、G1の7 prototype/6頂点構造、G2/G3の有限law、G4の限定分離/CTS対照、G5のclass-wide保存証拠という異なる成果を保持する。[R0/R1][R0]・[全体claim map][G5MAP]

**条件と結果**：2-system-qubit distinct-basis synthetic、p=(3/4,1/4)、V=exp(−iπXX/16)、finite P3、既知x=1/8,1/4、controlled、登録native精度のliteral価格。分子geometry/basis/DF/L_Dは非該当。R1は126key/264rowを完了したが、一律優越ではなくnative/shot間のtrade-off。G4-Aの指定lawによる限定B2分離は保持する一方、G4-Bのmatched CTSは同じ指定lawを下回った。G5はx=1/4の指定six-vertex digital classを同一保存CTS lawが下回ると確認した。[G4][G4]・[G5][G5]

**状態・限界**：同toy・同辞書の実用優位を求める主線は既に区切られた。R0/G1/G4-Aの数学成果は失効しない。旧RA-D0のtechnical inconclusiveを後続成功で塗り替えず、G5のclass外のLCU・別confidence・物理必要shot下界・分子改善へ広げない。再検討には旧classと異なる具体的操作/情報アクセスが必要で、parameter追加だけでは足りない。

### 3.3 G6–G10：ReturnKernel、General Full、Closed fast path

**目的・仕組み**：Q_i²=Iによる形式returnを集約し、even parentとchildの回転へ組み直す。G6はGreen/first-passage系列とordinary envelopeを利用し、全word宇宙や真のglobal B_newを入力せず局所生成する。既知のfree-product Green、Euler pairing、zero-fill等を組み合わせた構成で、一般部品の新規発明とはしない。16 focused形式test、有限bit局所packet、mean/moment/rangeとbiasの区別を実装した。[G6][G6]

**G7/G8の成果**：P3 p=(3/7,4/7),x=2/5 とP5 p=(1/5,3/10,1/2),x=5/7のdevelopment入力、axis精度1/200、strict RZ精度10⁻⁶、H160/root256。G7ではexact controlled-Q query仮定の下でP5 Fullの条件付き期待Tが登録ordinary/partial/P3対照を下回り、P3ではClosed対照が強かった。未知provider/prep単価はaffine項のまま残す。G8は2048 interface trialと20 unique positive keyのon-demand/cache経路を確認したが、短trace・同backend/seedのsequence一致は外部独立再現や全support保証ではない。[G7][G7]・[G8][G8]

**G9/G10の強い対照**：3-system-qubit synthetic Q0=Z0、V1=R_XX01(π/4)、V2=R_XX12(π/4)R_ZZ01(π/4)、Qi=Vi†ZiVi、p=(1/5,3/10,1/2)、x=5/7、有限P_m、strict RZ10⁻⁶、H160/root256、literal direct native。分子条件は非該当。axis精度1/200、G9のaxis α=49/22000、G10は49/34000、resource failure込みの各familywise1/20。G9 v2は11row/22axis、G10 v3はm=3/5/7内の17row/34axisを完了した。異degreeを同exponential精度として比較しない。

| G9 v2・同P5 directの引用 | N/axis（十分forecast） | 二axis T intercept | native CX | 限界 |
|---|---:|---:|---:|---|
| ordinary | 1,246,046 | 348,759,216.423 | 20,586,846.721 | 共通prep/readoutを別係数で保持 |
| General Full | 1,076,085 | 257,506,695.137 | 15,748,232.374 | zero-fill/保守的budget |
| Closed P5 | 917,129 | 255,860,636.626 | 15,647,565.043 | 同ideal Full ensembleのfast path |
| matched CTS | 1,596,220 | 364,311,583.986 | 9,766,516.587 | CXでは強い対照 |

出典：[G9][G9]。Closed P5は登録direct表のTとprep/readout slopeで有利で、同CTSに対するT intercept低下は既報約29.768734%。一方CTSはCXが低く、全資源の支配ではない。Closed P5とGeneral Fullは同じ理想return ensembleで、native event当たりTも同じになる記録がある。生成・normalizer・finite-bit・acceptance・十分Nの違いを別の量子改善原理と二重計上しない。

G10ではm3でClosed P3、m5でClosed P5、m7でClosedP5Tailなどを認めると、Full一律T優位は支持されない。m5の6row/19keyはG9からの再利用＋新confidenceでのrebudgetで、独立実験の追加ではない。[G10][G10]

**状態・再検討条件**：各run完了・STOP。既知low-order fast pathは再利用すべき成果。全supportのnative取得、非列挙の生成と価格取得の両立、他provider/分子/full-wrapperで増分が残る構造は未確立。既存Closed/CTSにない具体構造が先に必要で、同じG6–G10の再実行は引継ぎ方針に含めない。

### 3.4 Parent Allocation：辞書変更とsamplingの交換

**目的・仕組み**：parentのidentity係数aをchildへz_i≥0,Σz_i=aと配分し、c_i=√(z_i²+b_i²)と角度を変える。理想mean/phaseを保つが、common配分はこのfamily内のL1最小で、noncommonはnormalizationの無条件改善ではない。Euler＋線形分割と既知IS/KKTの応用である。[Parent][PARENT]・[終了総括][PCLOSE]

**条件と結果**：既知3-qubit synthetic、p=(1/5,3/10,1/2)、state000、τ=x=5/7、r1、finite P5、strict pygridsynth2.0.0/seed0/primitive10⁻⁶、axis1/200・α1/40。正式集合は50 control＋訂正S4の20law。旧raw-C KKT proxyの20lawと訂正√C proxyを混ぜない。

元総括のh_T=0におけるbest-common対best-noncommon比較では、ordinaryで約14%、P5で約19%の期待総T低下が記録されたが、N/CXは増加した。proposalも異なるのでallocationだけの効果ではない。同proposalの訂正ordinary KKT/halfwayはT改善を示さず、P5 KKTの小差とhalfwayの不利を別に保持する。全70lawのh依存包絡線、連続native最適性、online古典費は未取得。[PCLOSE][PCLOSE]

**状態・再検討条件**：限定成果を保持して同developmentの数値探索を区切るD＋Eは既採択。これはParent一般の無効性ではない。新しい非列挙取得や具体的PR taskの成立条件が必要で、同じr/precision/proposalの再探索で延長しない。

### 3.5 D1–D3、S2、J1：非加法whole-path価格

**目的・仕組み**：stepごとの価格和で失う境界相殺をwhole-path価格へ戻し、同じmeanを保つsamplingへ接続する。D1は六表現/r1・r2、D2は全方式共通のliteral隣接inverse相殺、D3は保証と合成価格の影響を分けた。D2は219,303 ordered positive pairを扱う既存成果で、新しいcompiler最適化ではない。[D1][D1]・[D2][D2]・[D1–D3総括][D3]

条件は同3-qubit synthetic、p=(1/5,3/10,1/2)、τ=5/7、state000、finite P5、r1/2。固定rでは同targetだがrを変えると有限多項式は異なる。D3では保証/離散価格により十分N、選ぶr、P3+tail/P5の保存順位が変わった。P3+tailは同finite P5の表現で、低精度P3への変更ではない。これは既知の費用×統計原理の具体的な限定証拠である。

**S2の新sampling利益は残らなかった。** D2 Closed P5、r2、x=5/14、固定bias/axis誤差1/200・α49/24000、root256/probability160、strict primitive10⁻⁶の3,969 pair表で、元law P0と加法価格IS qA、境界価格IS qBを比較した。[S2][S2]

| 同境界価格B・二axisの引用 | 十分N/axis | 既知期待T費用 |
|---|---:|---:|
| P0/B | 773,966 | 427,870,692.302325 |
| qA/B | 774,100 | 427,931,933.535182 |
| qB/B | 774,093 | 427,926,178.861741 |

qBはqAより既報約0.00134476%低いがP0より約0.01296807%高い。単価低下をM2/rangeによるN増加が上回った。未知共通費hを0とせずaffine項を保持し、このTの向きは共通非負h_Tでも変わらないという元報告の結論を引用する。古典online費は未測定、分子/全RPEの結果ではなく、Sampling一般のNO-GOでもない。

**J1の取得障害**：ReturnKernelは局所係数、ClosedP5Tailは群law、Parentは外部価格付き有限tableを扱う。これらから全path価格scoreのnormalizerや条件付き生成を安く得られる追加構造は特定できなかった。同価格groupingは全pair表取得後で、非列挙取得の証拠ではない。[A2/J1][A2J1]

**状態・再検討条件**：S2完了・STOP、J1は `NO_NEW_CHEAP_ACQUISITION_STRUCTURE_IDENTIFIED`。小さいprice summary、閉じた遷移、score部分和と実conditional lawが同時に取得できるなどの新しい具体性が必要。O2-R1も `NO_SUPPORTED_R1_CANDIDATE` を維持し、同じpair表・Parent・Closed構造を再評価しない。[R1選別][R1]

### 3.6 分子接続C1と古典生成APIの到達範囲

G6 live APIには512 correctnessを含む16,896件の限定古典workload完了がある。これは量子Hadamard測定pilotやIID/一般API互換性の保証ではない。productionの非列挙化、offline受入、全native取得、全RPE改善を同じ到達点としない。[B3][B3]

C1はlinear H4/1.00 Å/STO-3G/legacy DF12/generation-prefix L_D3/8system、T4/5,q1,r4,m5,h1/5の **tail-only＋scalar**。新R1 dyadic model、13basis/324labelを保存し、state/reference修復後の選択6eventでcontrolled phase/adjoint/X/Yを確認した。stateの厳密ground性やrigorous physical model bindingは未認証で、後のA1 N1/N2とは別identityである。[C1 correctness][C1R]

Ordinary/Fullは各1024 logical path完走、旧P5停止後にP5-onlyの別SOURCEで1024 path完走。3方式でzero回避を観測せず、4 child rotations/pathが残ったという既報結果が凍結理由の一部。logical差、計時/cache条件の違いを、strict native・認証N・full-wrapperの改善へ変えない。**C1追加研究は正式一時凍結**、G6方法論は維持される。[C1完走][C1P]・[C1総括][C1]

再開にはstrong Closed/CTS後の具体的改善構造とnative/full-wrapper/共通confidence/取得費の有限scopeが必要。未知項が残ることだけを追加計算の理由にしない。今回の統合は凍結変更ではない。

## 4. H4の層別化・CV：target固定からpilotまで

### 4.1 N1/N2/engineはCV改善を示す実験ではない

以下のH4-A1系列の共通条件は、保存linear H4/1.00 Å/STO-3G/legacy DF12/8system/全4電子sector/L_D3、m0,q1,δ=.02,r1,K2、coefficient_atol=0.0。N1 actual finite-grid Q・selected full、N2保証ground、旧exact Fraction engine、理想gate・独立uniform53-bit wordモデル。PRNGの数学的独立性、未近似分子、実機noise、最終energy保証を認証したものではない。

旧opt2 identity不一致・来歴不足のSTOPを経て、新しい保存DF有限targetを固定した。N1は完全payload、actual CDF/Q、324 symbolic label中232 positive-grid、全5正層とcontrol平均を取得。g1≈0.04983、g2≈3.39×10⁻⁶、g3≈1.89×10⁻⁹は正supportを示すが、無条件の少数trainで学習可能とは言えない。[N1][N1]

N2は全4電子70次元sectorのground分離とstate誤差を保証したが、当初total engineはUNKNOWNでSTOP。後続の局所gate一様資格で、固定旧exact engine・上記wordモデル下の座標誤差表示上界 `b_engine≤7.8079008×10⁻⁷` を取得した。新しいA2 gate群への移送保証ではない。[N2][N2]・[engine][ENGINE]

### 4.2 B1-OLと3-control A1の関係

**目的・仕組み**：B0はraw、B1-Oはorder、B1-OLはorder/labelの全5正層で既知質量を用いて合成する。A1は同じ測定bit/path metadataに、既知平均νを持つ3controlの補正を付ける。独立train→係数凍結→fresh評価、補正rangeと全学習費が必要で、最適係数・真signalを無料に使わない。

既存理論整理では、現在のcontrolは各B1-OL層内で定数。層内条件付き分散を追加削減できず、理想連続最適B1-OLに固有の分散優位はない。一方、有限quota/probe/整数配分/学習費が異なる実装比較は別で、有限pilotの優劣をこのoracle関係だけで断定しない。[A1理論][A1THEORY]

### 4.3 旧N3の中断と、752-shot完走は別の記録

旧4方式N3は座標ε_X=ε_Y=1/8、同時α=1/20、calibration512/評価単位、production上限8192/方式/軸を採択していた。これらの保証付き本検証予算を、後の752-shot探索へ無条件に継承したわけではない。

| 実験 | 保存bit/価格 | 既報compiled RZ | 終了・比較の意味 |
|---|---:|---:|---|
| 旧4方式N3 | 397 / 397 | 1,935,700（保存bit分） | 未完了77trial、親/worker消失の原因・exitはUNKNOWN。cal/N/prod未到達 |
| 新B campaign | 1,339 / 1,318 | 6,435,609（既知分） | 方針縮小の正規停止。価格欠測21・bit未保存35、campaign総費用UNKNOWN |
| 752 pilot attempt01 | 0 / 0 | 0 | 資源admissionで最初のtrial前STOP。比較結果なし |
| 752 pilot attempt02 | 752 / 752 | 3,632,115 | 完走。X/YともINCONCLUSIVE |

出典：[旧N3][N3]・[新B停止][N3STOP]・[752最終結果][752]。原因不明中断をOOM等へ断定せず、正規停止を科学的失敗にしない。未着手方式の費用0は優位性の根拠でない。旧397件、新B1339件・係数・乱数/cacheを新pilotの学習へ混入していない。表を新しい研究累計へ合算しない。

### 4.4 752-shot：学習可能性を確認、比較はINCONCLUSIVE

同H4-A1条件、seed20261108。A1は5層×16/axisの160学習、B1-OLは5層×8/axisの80probe、両方式各axis128の512独立評価。A1 X/Y rank4、range幅は約3.125/2.5。B1-OL quotaは両軸(9,107,4,4,4)、全5正層を保持した。[752][752]

| 方式・軸 | 推定平均（表示） | Vhat(mean)点指標 | 評価RZ | zero-variance cell |
|---|---:|---:|---:|---|
| A1 X | 0.985917681 | 0.00024328269 | 613,714 | なし |
| B1-OL X | 0.999897955 | 1.0475128e−9 | 615,321 | 0,1,4 |
| A1 Y | −0.077939674 | 0.0078391053 | 613,613 | なし |
| B1-OL Y | 0.072048329 | 0.0087119666 | 615,412 | 4 |

Xのpoint scoreはB1-OL、YはA1が小さい方向だったが、全bit同値の層のsample variance0は母分散0を保証しない。契約どおり **X/Y/全体INCONCLUSIVE**。保証付きcalibration/productionを実施したことにはならない。

| 方式 | 学習/probe RZ | 評価RZ | 全段階RZ |
|---|---:|---:|---:|
| A1 | 782,346 | 1,227,327 | 2,009,673 |
| B1-OL | 391,709 | 1,230,733 | 1,622,442 |

全価格はQiskit1.3.0、(rz,sx,x,cx)、opt2、seed17、selected full no-prep、cache hitも全価格を課金。実行約70分はこの一campaignの古典elapsedで、期待量子費用の比ではない。**状態は完了・STOP、同精度の期待総費用優位は未確定**。再検討には新しい層内変動/測定機構や有限配分上の具体的根拠が必要で、同3controlの追加shotへ自動移行しない。

## 5. 層内SCV：非定数controlと学習費込み比較

### 5.1 機構・既知法との差

5層を保持し、層内で変化するmetadata uとexact条件付き平均ν_hを使い、凍結λ_hによる `Σ p_h[Ȳ_h−λ_h(ū_h−ν_h)]` を測る既知の層別＋CVの応用である。Fragment群indicator、signed supportは追加量子測定なしに同pathから取得できるが、ν/moment計算とfresh trainは有料。層内非定数性・rank/rangeは相関や利益の十分条件ではない。[Signed metadata][SMETA]・[SCV統合][SCVSUM]

両pilotは同H4-A1条件/同旧engine、h1だけ学習、他層λ0、全5正層の同じfresh評価をB1-OL/SCVで共有するpaired設計。pilot間でcontrol・学習量・quotaが異なるため、絶対Vhatからcontrolを新ランキングしない。

### 5.2 既報結果と価格

| 項目 | Fragment群indicator 224 | Signed Support 448 |
|---|---|---|
| control | S={df-fragment-3}のrotation群indicator | rotation labelのsigned support metadata |
| fresh train / paired eval | 64 / 160 | 128 / 320 |
| 各axis評価quota | (4,64,4,4,4) | (8,128,8,8,8) |
| rank/guard | X/Y rank2、D≤4 | X/Y rank2、signed幅≤6 |
| XのVhat傾向 | 表示上実質不変。h1 all-equalを保持 | 差極小、退化/疎flagを保持 |
| YのVhat B1-OL→SCV | 0.0151001791303→0.0150964477577 | 0.00737790271621→0.00740702201167 |
| 学習RZ | 306,990 | 613,350 |
| 共通評価RZ | 772,251 | 1,543,455 |
| physical record分の全RZ | 1,079,241 | 2,156,805 |
| 既報elapsed | 約24.26分 | 約66.09分 |

出典：[224][224]・[448][448]。この「physical」は重複しない測定recordの会計単位で、量子backend実験ではない。224のY点低下は既報約0.024711%だが、両軸の学習込みpoint scoreは不利だった。448も追加の分散点改善・学習費回収の兆候は得られなかった。pairedの共通評価RZをB1-OL参照とSCV参照から二重に合算しない。

**支持すること**：exact中心化・独立train/係数凍結・全層paired評価・保存bit/価格・range/rankの技術接続は完了。**支持しないこと**：母分散差/相関の厳密判定、統計的有意性、最適B1-OLへの優越、期待総RZ改善、SCV一般の無効性。

状態は両campaign完了・STOP。旧SCV統合の「現2control一区切り・追加科学計算保留」は当時NOT_ADOPTEDの提案として保持し、後続のA2への移行と混同しない。今回の統合はProduct一致/別controlの再開・採択を行わない。Born noise支配、弱い相関、有限学習損失の寄与はUNKNOWN。再検討には追加測定費込みで異なる機構が残る根拠が必要で、control名の変更やtrain増量だけで救済しない。[SCVSUM][SCVSUM]

## 6. A2：coherent Multilevel、augmented-state、LCU

**目的**：metadata CVが残す測定内noiseへ作用する測定構造を調べた。通常のfine/coarse path共有では別々のHadamard測定noiseが残るため、真path平均の小さい差だけではshot削減にならない。PRのX/Y信号をHadamard状態上のobservableへ対応させる代数的整理と、物理アクセスの完成を分けた。[A2/J1][A2J1]

最新A2は既知2-unitary LCUにより、fine/coarse明示gate列、追加block/heraldの2ancilla、PREP/SELECT/逆PREP、制御RY、X/Y readout、全attempt zero-fillを実装。state逆準備/reflection・coherent sampler・未知norm oracleは使わない。人工1/2system qubitでsuccess/failure block・phase・success probability・推定値を確認し **新規9件PASS**。[A2最新][A2]

既報の記号は `χ=(ζ(Ψ_f−Ψ_c),Ψ_c)`、`d²=||Ψ_f−Ψ_c||²`、`S=1+ζ²d²`、`β=ζ+√(1+ζ²)`。successはχ/β、p=S/β²。成功時の測定値zにβ²を掛け、失敗0を含めて全attemptで平均する。既存のscale付きobservableの小normだけで比較せず、成功率と失敗prefix費を戻した。

\[
 E[T^2\mid path]=\beta^2(d^2+\zeta^{-2})>4.
\]

これは今回の特定LCUの既報の会計式。simple coherent分岐は同じ差分meanでM2=4であり、当該LCUに測定noise削減の利益は残らない。fine=coarseでもβ²/ζ²のnoise床がある。任意dilation/quantum protocolの不可能性へ拡張しない。

**状態・未取得**：`EXECUTABLE_WITH_MINIMAL_DELTA` と `COST_ADVANTAGE_UNSUPPORTED`。実H4のfine/coarse binding、追加gate/scale/joint-grid数値誤差、native/compiled価格、必要NはUNKNOWN。旧single-control engine認証は追加gate群へ移送しない。既存K2→n=0 pushforwardの二level候補も未採択で、実H4で計算していない。

前段 `ACCESS_MODEL_INCOMPLETE` の履歴は、最新の具体LCUアクセスで技術的に進んだ。資源改善の候補は残さず、新科学実行の一時保留を技術提案として記録した。再検討には既知操作で達成可能な有利なscale/全attempt moment・価格が必要。今回の統合で新しい理論や別dilationを設計しない。

## 7. O2全体に共通する知見とUNKNOWN

| 既存成果が支持する知見 | それでも残る境界 |
|---|---|
| sampling law、normalization、weightは必要Nと不可分 | 単価低下だけでは同精度総費用を下げない。S2が具体例 |
| 低次数Closed/CTSは必須の強い対照 | General Fullの生成能力を追加量子利益と二重計上しない |
| exact有限bit補正・strict phase・全trial分母が必要 | 理想law/meanと実q、channelとcoherent signalを混同しない |
| 価格/normalizer/conditional lawの取得が研究対象 | 全path表や未知価格を無料oracle化しない。J1の障害 |
| metadata CVはpath間の説明可能成分へ作用 | 残るBorn noiseと学習費、有限係数精度がある。真の寄与は未確定 |
| rank/非定数性・人工PASSは重要な技術成果 | 実outcomeとの相関、精度保証、資源改善の証拠とは別 |
| coherent差分の測定は具体化可能 | LCU scale/success/失敗費が利益を打ち消し得る |
| 実現RZ・十分N forecast・期待費用保証は別 | 中断・欠測・未測定古典費を0補完しない |

O2から新たに「全PR＋RPEの最終資源評価を完了した」とは言えない。保存model外、長round、他分子、他compiler/精度への一般化、全量子/古典費を含む優位、新規性・論文化可能性は未確立のまま保持する。

## 8. 今後の新規O2研究へ引き継ぐ探索領域

現候補を同条件で再実行する方針ではなく、**同じO2領域で、過去の障害を越える新しい操作的性質を探す**。以下は探索領域の整理であり、科学的順位・候補の採択・実装開始指示ではない。

| 探索領域 | 避けるべき既存障害 | 再利用資産 | 新しく示すべき性質 |
|---|---|---|---|
| sampling lawの設計 | 同じp/√C、同じParent/classの再探索 | finite-bit/weight、固定辞書・全path会計 | 安い生成/normalizerとrange/M2負担を同時に改善する操作差 |
| 測定プロトコル再設計 | 独立noise床、単純LCUのscale負担 | controlled phase/X/Y、A2 flag/decoder | 無料state oracleなしで全attempt noise/価格に利益が残るアクセス |
| 推定器の改善 | 層内定数control、train費、未知ν/母分散 | 層別/CV、独立fit/freeze、既存bounded confidence | baseline後にも残る相関または有限取得上の新性質と有料の学習法 |
| returnの新しい利用 | Closed/CTSへの還元、child回転が残る | ReturnKernel、ClosedP5Tail、native IR | 既存fast pathに含まれない具体return/context構造 |
| ランダムeventの部分的解析処理 | products恒等化だけではfull signalが既知にならない | scalar/identity/zero分類、full wrapper意味論 | 未知state/Bornなしで寄与と質量を評価できる認識可能部分 |
| 適応sampling・測定資源配分 | IID保証の無断流用、pilot費、希少層除外 | conditional取得、独立stream、既存confidence接続 | predictableなlaw/weight、妥当な停止保証、全取得費と正層coverage |
| samplingとreturnの新しい結合 | J1の価格summary/部分和不足 | G6/Parent/D2の接口 | 小さい閉じたsummaryから価格scoreとconditional生成を取得する構造 |

次の提案では最初に、既存手法でできない操作、使える情報、保存するtarget、強いbaseline、最小の判別材料を示す。科学実行・新規実装・重要レビューを開始する指示は今回含まない。旧SOURCE・結果・STOP・C1凍結を維持し、本統合の公開照合でmandatory STOPする。

## 固定引用先

[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/docs/research/o2_sampling_return_technical_survey_20261106/REPORT.md
[R0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/24bfeb84a4ce87b56985d174dd98d1d5e1702a2b/docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md
[G1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g1_one_shot_result_validation_20261009.md
[G2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md
[G3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md
[G4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md
[G5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md
[G5MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g5_claim_evidence_map_20261010.md
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/docs/tracks/algorithm_codesign/parent_allocation_s4_supplement_results_and_gpt_handoff_20261015.md
[PCLOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md
[C1R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/handoff.md
[C1P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/handoff.md
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md
[FREEDOM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a776aa84aefc128cc317e33075415140916013a0/docs/research/research_c_sampling_law_freedom_survey_20261030.md
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/docs/research/o2_r1_server_candidate_screening_20261106/REPORT.md
[N1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/docs/research/o2_a1_new_target_n1_20261106/REPORT.md
[N2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5e9bd08068738cdf1070bdc43bee8194b113ae33/docs/research/o2_a1_new_target_n2_20261106/REPORT.md
[ENGINE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdf00cee3c87d804845497afc89ac6a71f3b3f59/docs/research/o2_a1_engine_qualification_20261106/REPORT.md
[N3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d97e916526f431280e687815d3e970bbf03a077e/docs/research/o2_a1_n3_science_20261106/REPORT.md
[N3STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/36ee9c6244eeedc875b5ddfafffb4c4cf76c8b8b/docs/research/o2_a1_n3_pilot_preparation_20261107/REPORT.md
[752]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c8e55a2728f86a1eda17a2e0105050a637e7c86/docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md
[A1THEORY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8d8a290fa4d4cf24f968aea27953ad5422e418b3/docs/research/a1_b1ol_theory_20261107/REPORT.md
[224]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/REPORT.md
[SMETA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4512675ae1d650858256547d359210361d77f68/docs/research/scv_signed_support_h4_metadata_science_20261107/REPORT.md
[448]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/scv_signed_support_448_science_20261107/REPORT.md
[SCVSUM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62c450a6c22fc6cf228bd2fbaabda3c36a7cf169/docs/research/o2_scv_two_pilot_synthesis_20261107/REPORT.md
[A2J1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e610af3c0a0b9745cff63b39b57ee1f97657cf6/docs/research/o2_a2_j1_mechanism_20261108/O2_NEXT_ALGORITHM_REPORT.md
[A2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89facf729535901a79c111721e5ab1714bd9a4be/docs/research/o2_a2_executable_access_20261108/REPORT.md
