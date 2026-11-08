# 次回の最小人工判別案（未採択・未実行）

次の担当はユーザー／GPT。この文書は新しい科学実行の許可ではない。今回の結論案C1継続を採る場合に、**head出口→tail最初のfull-frame run入口**という一種類だけの局所relative Gaussian橋を、人工条件で判別する提案である。

## 再利用と最小差分

- 既存C7-2 `context` の2-mode/3-native-blockとfull-frame Provider、独立exterior-minor lift、C3のmixed_head/physical_segment、既存Reference gadgetを使う。
- frame端部・実回路gauge・wire順を明示して接続し、既存 `StrongBuilder.transition`を呼ぶ薄いopt-in接口だけを追加する案。現行builderのdefaultを変えず、gate列の末端を名前だけで推測して削る実装を避ける。閉じたhead/tailの端部を科学的に識別できる最小の表現を用いる。
- 抜いたG_HとG_T†の代わりに、同じsource/input/rotationとanalytic gaugeを持つG_T†G_Hを入れる。head/tail中心、tail内run共有、既存step境界融合、scalar/identity/χ、global cutを変えない。tail出口→head方向、tail異frame run一般、全mode中心合成は初回に追加しない。
- StrongとReferenceの各physical segmentへ同じ適格性規則を適用する。Aは書換え後のactual inverseを使い、CSWAPを跨いで融合しない。未適格接口は元の列のまま保持し、別法へ切り替えない。

## 人工fixtureと反例

第一案は固定C7-2 CASESの**q2_mixed、cut1**をそのまま使う。2 mode、3 blockのframe角0.23/−0.37/0.41、scalar .17、extracted identity .04、h=.1/τ=.2、K0、steps=(plus,zz)→(minus,plus)。最初のhead出口block2→tail先頭frame1は異frame、次stepのhead出口block2→tail先頭frame2は同frameで、同じ一fixture内にrelative化候補と相殺/P0重複の対照がある。これは既存人工入力の提案上の指定で、今回は生成・復元・実行しない。geometry/AO basis/分子DF rank/分子LDは非該当。

結果を見て別のframe角やpathへ切り替えない。このcaseで追加効果がなくても保存し、そのまま停止する。

negative checkは小さな局所検査として、(a) 正しいG_T†G_Hの代わりにG_H†G_T（逆向きの橋）を使った列、(b) analytic gauge補正の欠落、(c) ordinary tailをdirectional等へ誤変換した列、(d) 同じbasis IDでもrotation/wire/gaugeが違う入力、を固定する案。vacuum energyが全て零の既存fixtureだけではgauge欠落が検出不能な場合があるので、**局所phase witnessをG_Hのdeclared global phaseにη=π/7を加えたcopyとして事前固定**する案を置く。single-particle rotation descriptorは同じでも真空位相と橋の全unitaryは変わるので、既存vacuum_phaseから得た補正の保持と欠落を区別できる。これは局所検査専用であり、固定whole-pathのeventやlawは変えない。phase witnessを通すための事後調整はしない。間に非可換system gateがある局所caseも固定して橋の適用を拒否する。明らかな入力不適格は拒否、誤書換えは全unitary/coherenceの不一致検出を期待する。

## 判別順序・必要な費用

1. まず局所橋の全unitaryをphase込みで比較し、逆順/gauge/modeのnegativeを検出する。新しいGaussian builder、ground solve、samplerは不要。
2. 固定人工whole-pathのS₀/S₁/R₀/R₁について、独立V、AB=V、actual A†、A/B vacuum ray、analytic χ、Strongの両branchまたはW0†W1、全system入力X/Y/coherence/reference leakageを検証する。全4 system入力、Referenceは既存5-qubit小規模checker。component1e−10・observable/signal2e−10・rtol0・phase fitなしという既存規約を実行前固定し、不一致後に緩和しない。
3. rawがPASSした場合だけ、**4構成×X/Y、最大8 full-wrapper**を共通P0で比較する案。Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/declared phase保持。既存C7-2のS₀/R₀ compiledは、SOURCE/profile/full raw IR/phase identityが完全に適合する場合だけ再利用し、再利用不能なら新しいbaselineも同予算内で一回compileする。今回そのidentity判定はしない。
4. compile後も全unitary・X/Y/coherence/phase/leakageを照合し、7資源のS₁−S₀、R₁−R₀、前後のR−Sを別々に記録する。低い局所gate価格を足して利益を作らない。適用接口と不適格接口の処理も記録し、Strongを弱体化しない。

人工phase witnessのSOURCE・call数は次の実行許可の範囲を固定するときに必要十分に確定する。大きなgrant/認証基盤や多数の未採択契約は作らない。

## STOP・未確認のまま残すもの

SOURCE/input/phase/全入力correctness/対照公平性の不一致で依存処理をSTOP。無断retry、fixture/cut変更、精度緩和をしない。P0増分が0または悪化でもその結果で完了し、別case探索へ進まない。結果・原本・旧STOPを保全し、公開照合後mandatory STOP。

人工利益はH4への証拠でも全law価格でもない。実H4の保存接口分類・QPY価格解析・build/compileが必要なら、人工結果を受けて別の限定実行許可を得る。law/PF変更C2、N、全13-round更新、q64進行中結果、詳細科学レビューは含めない。
