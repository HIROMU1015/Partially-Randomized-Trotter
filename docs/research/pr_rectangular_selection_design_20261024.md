# Rectangular Givens候補A：構造による選択規則は未確立

**Gaussian singleton選択最適化を、独立した研究方向としては一旦終了することを推奨する。**
元support・frame・基底変換の境界は説明に役立つが、compile前に候補Aを選べばRZが減るという簡潔で一般化可能な規則は、現資料から確定できない。
再開するなら、下記の構造proxyを未検証仮説として固定し、既知の「両方式をcompileして選択する」対照との資源regretと古典費用を評価する必要がある。
今回の結論は技術的な継続価値の提案であり、研究終了の正式採択・科学的GO・新規性の判定はユーザー／GPTへ戻す。

## 根拠と今回の境界

[PS48結果](pr_rectangular_givens_ps48_execution_20261024.md)のcommitは
`1e20458d742a8fac62ce0d0264188b1a7772ee28`、候補AのSOURCEは
`770e53addda63a03fa8c6c5d5168f7062067c572`、[F2寄与分析](pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)は
`41c4e79f1c58af0f2be634f46dbeba8dcd36acc5`。
対象は4 mode・2粒子・occupied(0,1)、L_D=1、残差2fragmentの固定合成Hamiltonian。
分子geometry／AO basis／分子DF rankは非該当、rank探索なし。
T=δ=1/5の一点、q=1、r=2、h=1/10、K=0、固定F2 compiler
（Qiskit1.3.0／Nature0.7.2、opt1、seed17、rz/sx/x/cx、topology-free、位相保持）でのPS48に限る。
ここでF1は残差fragment F1。

今回はSOURCE・保存frame/atom metadata・公開済み結果の読取りと設計だけ。
以下の数値は既存報告の引用であり、新しい集計・oracle値・予測一致率・ランキングは計算していない。
Givens列の新生成、QPY decode、行列生成、build／compile、選択器実装、科学test、新grantは0件。

## 説明できる構造と、まだ説明できないcompiled費用

元frameは **U=R₂₃ R₀₂ R₀₁**（右端が先に作用）。既存方式Bは元support Sの選択列を固定し、
標準ベクトルの二重再直交化で補空間を埋め、正方行列を分解する。
候補Aは選択列Vだけのrectangular分解を用い、内部supportをT={0,…,|S|−1}へ固定する。
両者の演算子はPS48で全件同値だが、同じ部分空間を異なる補完・wire配置・Givens列で実現している。

| 演算子 | SOURCE／保存metadataから分かる特徴 | 保存PS48の観測と解釈の限界 |
|---|---|---|
| index20、S=(3)、IIIZ | R₀₁・R₀₂はZ₃と可換な内側suffixで、共役から除ける。保存Uの第3列はmode0/1でexact zero。既存completionはこの局所性を保持できる一方、AはT=(0)から対象部分空間へ変換する | 全8件でRZ/CX/depth増。単純な既存演算子にcanonical Tへの変換を課す不利と整合する。ただし各Givens／compiler passの増分は未分離 |
| index21、S=(2)、IIZI | 内側R₀₁はZ₂と可換だが、R₀₂は選択modeを外へ混ぜ、R₂₃も作用する。既存の第2列保持とAのT=(0)からの合成は異なるrecipeになる | 全8件で3指標改善。ただし「support1ならA」「Tから近いほど安い」は20／23との比較だけでも十分な説明にならない |
| index25、S=(1,2)、IZZI | 選択2列のjoint parityは選択空間内U(2)回転に不変。Aはこの自由度を使い、列を個別固定するcompletionを避ける。元Sは連続modeだが、T=(0,1)と同じではない | 全8件で3指標改善。U(2)自由度は22／24にもあるので、それだけでは選択条件にならない |
| index24、S=(1,3)、IZIZ | 同じrank2でも元Sは非連続でmode3を含む。第1列はframeの広い混合を受け、第3列には局所性がある。既存completionとcanonical Tへの分解の有利さは25と同じではない | 全8件でRZ/CX増。非連続supportやmode配置は説明候補だが、一般的な拒否条件としては未検証 |

これはoperatorの構造と観測の対応であり、gate単独の因果的費用を測ったものではない。
R₂₃が元support内の回転だから、という理由で任意の位置から削除してはいけない。
中央と可換な**隣接suffix**は除けるが、途中で選択空間と補空間を混ぜる回転を通った後は元Sへの所属だけでは判断できない。
ここではその性質を説明用の特徴として使い、既存回路をpruneする別候補は設計・実装しない。

**同じatomのB/A比較では、中心controlled回転のsupportサイズ・角度・符号は同じで、中心部の命令数は変わらない。**
変わるのはS/Tのwire、Gのrecipe、前後の基底・D・partnerとの境界である。
Natureの正方分解はorbital RZとXXPlusYYを、rectangular helperのGaussian suffixはXXPlusYYを返す。
rawのRZ個数やGivens個数だけでは、XXPlusYYのnative分解・特殊角・位相・cancellationを含むcompiled RZを比較できない。
両helperの`np.isclose`分岐もあるため、for-loopの最大段数をactual列長として扱わない。

### Partnerと順序はatomだけの選択を崩す

既存報告のindex23では、identity partnerのΔRZは+4、Pauli `IXXI` partnerでは−3または−2。
index24のPauli partnerではoperator順序でΔdepthが−6／+6となる。
atom、元support、T、Gaussian frameが同じでもfull-wrapperでの最適化が変わる。
compile前に使えるcontextは、partnerのPauli word・H/S基底、元basis keyの共有関係、event順序、
実際の中心target、角度・符号、Dとprep/readoutの境界まで含む。
同frame共有runは今回のAの対象外で、既存full basis共有を保持する。

## 実装前に考えられる規則と採択判断

| 案 | compile前の情報と判断 | 現時点の扱い |
|---|---|---|
| supportサイズ／位置／連続性だけで選択 | |S|、SとTの位置関係、frameの零構造 | 安い特徴だが費用差の符号を決められない。採択しない |
| index21／25、または同じframe/supportのlookup | 観測済みcostのラベルを記憶 | 同一入力のcacheとしては使えるが、事後のcase記憶。一般的な構造選択・独立新規性とはしない |
| **分解と境界を含むRZ proxy** | 既存二方式のclassical Gaussian recipeから、共通native templateでRZ・CXを見積もり、中央部・partner／D境界の明示的な簡約だけを反映。予測差が不明ならBへ戻す | 再開する場合の一候補。ただし未実装・未評価。genericな合成／compilation heuristicであり、新しいPRアルゴリズムの採択根拠にはしない |

最後の案でも、「Givensが少なければA」のような固定重みを根拠なく与えない。
特徴は元S・選択列の零構造・補完の規則・T・両recipeのnative template・boundaryとする。
候補AのGivens列を得るには数値分解が必要で、`BasisProvider.basis`を呼べば回路も生成される。
従って、これは入力ラベルだけの無料な判定ではなく、将来の承認済み分解／接続作業と古典費用の測定が必要な案である。
誤差帯や安全marginは未取得で、任意の閾値を「保証済み」として採択しない。

採択できる保守的な運用条件は、**singleton以外はB、特徴不明・境界不明ならB**というfallbackまで。
このfallbackだけではAを積極的に選べず、提案proxyによる選択全体のcompiled RZ無悪化も保証しない。
今回、実装へ進めるほど裏付けられた正の選択条件は得られていない。

## 強い対照：両full-wrapperをcompileして選ぶ

依頼で指定された既知のcost-aware compilationを強い対照とする。
同じ物理operator・角度・順序・law・wrapper・compilerでB/Aを作り、科学gateに合格した候補から
**RZを第一、同値ならCX、さらに同値ならdepth**で選択する案とする。RZ/CX/depthの3値も別に保存する。
この辞書式優先順位も、将来実行前に正式採択する条件であり、今回実行契約にはしていない。
既定にないCX/depthの増加を許容する重みや制約は勝手に作らず、必要ならユーザーが評価方針を別採択する。
depthが減ってもRZが増えれば、RZ主評価の勝利とはしない。

この対照は、固定二recipeと同じ評価規則の範囲ではactual RZの下側包絡を与える。
構造規則が同じ候補集合でこの対照より量子費用を下げたと主張することはできず、意義は主に
**不要な合成・compileを避ける古典費用の削減と、その代償となる資源regretの小ささ**にある。
basis単独で安い方を選ぶ対照ではなく、full-wrapperで比較する。
既存の`gates.compiled_wrapper`、完全IR/角度/profileを含むcacheとcase別の科学gateを再利用できる。
両方式をcompileして選ぶ処理そのものを、新しい独立アルゴリズムとは認定しない。

既存PS48の実transpile36件に要したwallの合計は1.876秒、全検査・起動込みwhole-job wallは21.93秒と報告済み。
これは二方式を最初からcompileした費用や大規模系でのspeedup測定ではないが、現状で複雑な選択器を作る実益は未立証である。
同一caseの両費用が既に保存されていれば再compile不要なcache運用も強い対照になる。
今回、PS48の包絡・選択後の平均費用・regretは**計算していない**。

## 独立検証を行う場合の最小案（未承認・未登録）

PS48は規則の考案に使ったdevelopment資料で、同じ48件の一致をholdout PASSとはしない。
Gaussian基底は6種類であり、X/Y・partner・両順序があるから独立したGaussian構造を48例得たわけではない。

再開を選ぶ場合だけ、まずユーザー／GPTが一つの規則、未知context時のfallback、RZ優先とCX/depthの扱い、
cost-aware対照への許容regret・古典費用の採否基準を**結果を見る前に固定**する。
その後、別の実装承認と科学実行承認、新SOURCE／専用one-shot／outputが必要。

| 検査軸 | costを見ずに選ぶ既存入力 | 提案例数と分かる範囲 |
|---|---|---|
| 未使用partnerへの境界移送 | PSの同じ6 atom、保存index3 `P:collect:IYYI`。既存partner2とsupportは同じで、局所基底が異なる | 6×1 partner×両順序×X/Y＝24 candidate wrapper要求。候補Aでは未測定のcontextへの移送を検査できるが、frameの一般化ではない |
| 未使用Gaussian frameへの移送 | 同じphysical fixtureのSP/F0から元word `ZIII` と `ZZII` を構造で先に指定し、保存identity partner、両順序・X/Y | 2×1×2×2＝8 candidate wrapper要求。異frameの最小診断。2基底だけで一般的保証とはしない。identityの両順序は重複し得るためlogical件数とunique compile数を区別 |

これは合計32 logical wrapperという**将来の案**で、今回case登録・law生成・科学処理はしていない。
元のPS48で得た6基底、保存B、元F2の対応baseline QPY／独立reference／費用は、SOURCE・角度・入力のbindingを確認して再利用する。
SPの二つのF0基底は新しいAの数値検査が必要で、その生成とprojector／actual adjoint検査も将来契約に含める。
baseline費用は元F2で測定済みであり、「全ての結果が世界で未観測のblind実験」とは呼ばない。
規則設計には新caseのA費用を開封・使用せず、予測を固定してから評価する。

always-B、always-A、構造規則、二方式compileの対照を同じcaseで比べる。
予測がBでも対照評価にはAの費用が必要なので、評価用Aのbuild／compileを無断で省いて採点しない。
既存baselineの再build／compileは不要。対応reference不足や角度binding不成立ならSTOPし、無断再計算しない。
全16／controlled32次元、basis/operator/unitarity 1e-10、raw signal 2e-10、compiled operator/signal 5e-10、rtol0、位相fitなしを維持する。
品質を測るための両方式評価費用と、運用時に選択器が省ける費用は別に計測する。
caseごとのRZ regret・CX/depth差と、feature抽出・分解・選択・compileを含む古典費用を記録する。
PS/SPの別armや未使用frameへの適用は旧PS48 grantの延長でなく、別の明示承認が必要である。

## 継続価値・引継ぎ・STOP

**現時点の推奨は一旦終了。** sourceから説明できる不利条件はあるが、Aを選ぶ一般的な正条件・予測誤差の根拠はない。
より詳しいproxyは既知compilation heuristicに近づき、既存の小規模transpile費用・cache対照に勝る実用価値も未確認。
index21/25の改善だけを理由に一般規則を作り、同じ48件でGOにすることは避ける。
operator同値性の成果と固定条件で改善したcaseは保全し、否定しない。

再開に意味があるのは、未使用contextでも小さいregretを保ち、十分な規模で古典費用を削減する必要性を別途示せる場合。
これは未取得の条件であり、今回新しい大規模計算・対照・selector実装を始める理由にはしない。
次の担当は**ユーザー／GPT**。この研究方向を一旦終了する提案を採否判断し、継続する場合のみ規則・比較基準と限定実装／科学実行の範囲を別承認する。

科学SOURCE、PS48結果CSV／JSON、raw、grant、ledger、marker、STOPと元workspaceは保全する。
今回追加するのは本設計メモと入口・履歴・文書参照だけで、既存の科学的PASSや統計未資格状態は変更しない。
**公開・保全確認後mandatory STOP。追加実装・再集計・build／compile・全F2・新fixture・γ／N・詳細科学レビューへ移行しない。**

SOURCEの対応箇所：
[completion・基底共有・中心部・wrapper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/770e53addda63a03fa8c6c5d5168f7062067c572/src/trottertracks/pr_dictionary_validation/circuits.py)、
[候補AとTの規約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/770e53addda63a03fa8c6c5d5168f7062067c572/src/trottertracks/pr_dictionary_validation/rectangular.py)、
[frameのfactor仕様](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/770e53addda63a03fa8c6c5d5168f7062067c572/src/trottertracks/pr_dictionary_validation/fixture.py)、
[位相保持compiler・cache](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/770e53addda63a03fa8c6c5d5168f7062067c572/src/trottertracks/pr_dictionary_validation/gates.py)。
固定Nature0.7.2の[rectangular helper](https://github.com/qiskit-community/qiskit-nature/blob/0.7.2/qiskit_nature/second_q/circuit/library/initial_states/utils/givens_rotations.py#L27-L81)と
[正方Gaussian分解](https://github.com/qiskit-community/qiskit-nature/blob/0.7.2/qiskit_nature/second_q/circuit/library/bogoliubov_transform.py#L198-L259)はlocalの固定SOURCE本文も静的確認した。
