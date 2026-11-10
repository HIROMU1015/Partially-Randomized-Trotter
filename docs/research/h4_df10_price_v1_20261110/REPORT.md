# H4 DF10価格モデルV1：保存回路の構造計数・価格残差

2026-11-10 JST。リソース評価系列。**指定した保存PF/PRの評価を完了。固定した構造モデルM1は、全留保68軸回路でcompiled RZとの整数一致に至らなかった。** 新fit・sampling・実回路build/compile・signal・N・N×C・全RPE費用は0。これは限定モデルの数値検証であり、詳細科学レビューではない。

[実行前PLAN](PLAN.md)／[開発結果](DEVELOPMENT.json)／[留保結果](HOLDOUT.json)／[全88軸行の残差CSV](PATH_RESIDUALS.csv)／[入力順序修正と初回停止](TECHNICAL_NOTE.md)。

## 1. 何を検証したか

Linear H4、隣接1.00Å、STO-3G、保存tol-only 1e−8 DF10/generation順、Nα=Nβ=2、保存DF-Hのgroundエネルギーをtargetとする既存の数値state条件付き評価、P=0。全RPE条件のεE=.00159360010199040 Ha、αtotal=.05/方式、βtotal=.4 rad、h=.1、13roundは維持し、今回N・signal・energy資格は再評価しない。

- PF：B0/S2/prefix7、δ=.1。q8/T.8を開発、q16/T1.6を留保。B0は切捨てPFでありFull PFではない。
- PR：S2/prefix1/q8/T.8/R_outer4/K2、δ=.1。探索9本を開発、確認33本を留保。cohortをpoolしない。
- 1 full Hadamard interrogation、X=cosine/Y=sine、measurement込み、state preparationなし。Qiskit1.3.0、rz/sx/x/cx、opt1/seed17、元provider/wrapper規約、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

価格の正解照合データは全て保存compiled値。新しい実回路を作って取得していない。PF2回路とPR42 trajectory、各2軸で88行。留保はPF1回路＋PR33 trajectoryの68行であり、独立68標本ではない。

## 2. SOURCE・入力・再利用

| 固定版 | 今回の用途 |
|---|---|
| [設計10132af4a][DESIGN] | §6のV1計画。既存branch `codex/resource-eval-h4-df10-longtime-price-design`を継続 |
| [元価格155529e2][H46] | scientific_provenanceと保存価格表。PF q8 `49709717a6c3d405c60d`、PF q16 `8e50c5560bb8f47f026f`、PR `14bd81a375400c9b0305`。元SOURCE/input/raw/価格は変更なし |
| provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe` | 元`diag_hermitian`、one-body/DF diagonal primitive関数を本文のまま再利用。元`h46_new2_native.gaussian_operations`も155529e2から再利用 |
| **モデルSOURCE `43f748655c64756c2c2a21d754870c2f7b321a77`** | M1、元runner、PLAN、人工テストを実データ評価前にcommit。開発後・留保後のモデル変更なし |
| **継続execution SOURCE `5a2e6edf0cc2f7c00be491baffb86e4f139222c3`** | 元リストの完了順をpath番号順へ対応付ける薄いadapterだけ。モデル・元runner・既存counterの全bytesは43f748655と一致 |
| [D1 ffe8a71da][D1]／[m3会計9fe5e6952][M3] | 完了済みとして参照。N・m3 N×C・順位・費用比の再計算なし |

保存NPZ/receiptのbytes/hashとH/DF/rank bindingを確認し、one-bodyと保存最初の7 DF fragmentから価格用の局所記述子を取り出した。同じNumPy1.26.4/Qiskit1.3.0環境を読取使用した。保存行列の8×8局所対角化とGaussian記述子取得であり、新SCF・DF fit・分子ground solveではない。state vectorを再解析していない。

PR全42本でcandidate/cohort/path index、preparation hash、PASS/conditionalフラグ、eventのordered basis descriptor hashを照合した。各cohortの保存path価格からの平均が元aggregate欄と一致した。生event・seed・NPZ・QPYを新たに公開せず、入力identityと原本hashだけを結果へ記録した。PF q8の旧再利用価格を新しい独立標本として数えていない。

主な再利用コードは[NativeCounter](../../../src/trottertracks/track_d_native_resources.py)。Track DのP0/1Q bridgeは再synthesisを含み、元DF10 compilerとの同一性も未資格なので今回呼ばない。Track Cの開発/留保分離、旧DF12のdirect/model/外挿の区別を継承したが、係数・価格・共分散は移植しない。

## 3. 固定した構造的RZ会計

モデル名は`DF10_SOURCE_NATIVE_BASIS_CANCEL_ZFOLD_V1`。[実装](../../../src/trottertracks/h4_df10_price_v1.py)は価格・保存平均・cohortを予測入力に取らず、係数をfitしない。

1. 元builderのuncontrolled forward/head、directional reverse/headをδ/2で配置。PRだけ保存tailを間に置く。blockのbasis逆変換→対角→basis復帰を保持する。
2. tailは保存application順・符号・rotation/product・identity・relative phaseと、元の同basis共有を会計する。identityはactive basisを閉じる。R_outer4の8 occurrenceをそのまま使う。
3. P/XX+YY・対角RZ/RZZ・controlled Z/ZZ・scalar・H/Sdg/measurementを、事前に選んだ解析的rz/sx/x/cx/measure展開へ対応付ける。scalar/抽出identityは元の規則で一回、X/Y wrapperも一回。
4. 直に隣接する同basisの逆変換対だけを消す。wire上で他gateに遮られないRZを加算し、binary64で厳密0だけを消す。近い角のsnapやCliffordへの格上げはしない。
5. 展開列を既存counterへstreamとして渡す。一般1Q再synthesis、CX相殺、P0固定点処理は実装しない。全QuantumCircuitやDAGをbuildせず、部分回路を個別compileした価格の単純和でもない。

**M1は、選んだ解析的native展開に対する会計候補であり、元compilerのbasis translatorが選択した出力列の再計数ではない。** 特にRY(t)=SX RZ(−t)SXdg、SXdg=X SXなど、位相付きで正しい局所展開を使った。元compilerが別の等価展開を選べばRZ数は変わる。native展開後の限定的な消去と元postcompiler価格の差を、最初から0と仮定しない。

`structural_rz`を候補予測値とし、未説明のcompiler correctionはnullのまま、**compiled−structural**を整数残差として保存した。`expansion_global_phase`は選択した展開の会計上の全体位相であり、分子signal・energy・RPE位相を新計算した値ではない。

## 4. 開発結果

下表は**各軸の値**。X/Yそれぞれを評価し、保存価格・M1予測・残差は両軸で同じだった。PRは有限cohortの平均であり母平均ではない。

| 対象 | n | 保存compiled RZ | M1予測RZ | 平均残差 | path残差 min…max | 0一致/軸 |
|---|---:|---:|---:|---:|---:|---:|
| PF q8/T.8 | 1 | 44411 | 39557 | +4854 | +4854 | 0/1 |
| PR探索 q8 | 9 | 16857.11111111111 | 15442 | +1415.111111111111 | +756…+1869 | 0/9 |

開発20軸行は全て不一致。保存値に合わせたoffset・傾きの逆算や、残差を見た後のモデル変更は行わず、同じSOURCEを留保へ進めた。

## 5. 留保結果

| 対象 | n | 保存compiled RZ | M1予測RZ | 平均残差 | path残差 min…max | 0一致/軸 |
|---|---:|---:|---:|---:|---:|---:|
| PF q16/T1.6 | 1 | 88515 | 78869 | **+9646** | +9646 | 0/1 |
| PR確認 q8 | 33 | 16947.78787878788 | 15514.545454545454 | **+1433.2424242424242** | **+736…+1960** | 0/33 |

留保68軸行の整数残差0は**0/68**、全て正だった。最大絶対残差はPFで9646、PRで1960。PRの確認値はsum(compiled)=559277、sum(predicted)=511980、差47297を33で割った有限平均である。これらを量子Nで重み付けしたり、探索cohortと合算したりしていない。

PF q16はq方向の一つの留保点、PR33本は同じq8での別trajectoryの照合である。どちらも既存値が既知の保存データであり、完全blind検証とは呼ばない。価格の元averageとpath値の対応照合は通過したが、M1の価格一致判定は通過していない。

## 6. どこまで説明でき、何が未説明か

| 内訳・観測 | 今回わかったこと／限界 |
|---|---|
| head・basis・対角・control・scalar・wrapper | 元SOURCEに沿って分離できた。PF q8のRZ融合前は78569、限定消去後39557、保存compiledは44411。q16では156649→78869、保存88515。native countをcompiled価格と同一視できない |
| source上の明示basis対 | PF q8で15対、q16で31対の隣接逆変換対をモデル内で消去。Track Dの融合定数を移していない。これらはM1の構造内訳であり、P0の全削減機構の認定ではない |
| tail basis共有 | 探索pathではbasis group21…27、共有対6…11。確認ではgroup18…30、共有対2…14。各pathの列構造を区別できたが、これだけでabsolute compiled価格は再現しなかった |
| rare次数 | 探索9本の1本、確認33本の2本にorder-2 eventが各1回。残りのorder-0だけのpathにも全て正の残差があった。したがって不一致をrare eventだけに帰することはできない。未観測support/tailを保証しない |
| 等価分解の選択 | M1のRY/XX+YY/controlled-ZZ展開と、元basis translator・1Q optimizerの選択は同一資格がない。同じunitaryでもRZ数が異なり得るため、残差を「未計上の削減」だけで説明できない |
| 未モデル化compiler/interface | 一般1Q run再synthesis、特殊角の処理、CX相殺によるrun再接続、Gaussian/head-tail/step/readoutをまたぐ最適化を未閉包として残した。各機構へ残差を一意に配賦する証拠は今回取得していない |

全ての保存価格がM1より大きいという有限観測は、M1が一般の価格下界であることを示さない。raw展開値は保存価格より大きく、限定消去後の値は小さい。どちらかへ一定係数を掛ければ十分だという結論も出していない。

## 7. 技術検査・停止の保全

人工テストは計数6種類と順序修正2種類の計8種類PASS。2/4次元の人工入力で、H/RY/XX+YY展開の位相付き作用、RZ融合、basis対消去の範囲、tail共有・identity、scope、保存完了順とpath identityの区別を確認した。実H4の量子作用correctnessや独立科学レビューのPASSではない。

初回留保は、元`trial_files`が0,3,7,2,1,…という完了順だったことを新runnerが取り違え、4軸の予測後にbinding照合で停止した。部分結果は未保存で、全留保完了扱いにはしない。同じ33本の参照集合を番号順に対応付ける修正だけで継続し、モデル・元runner・counterのbytesは不変。旧開発結果も不変。[停止と継続の詳細](TECHNICAL_NOTE.md)を保持する。初回一回で全campaignが完了したとは報告しない。

完了結果は20＋68＝88軸行。初回の未保存4予測だけは継続時に同じ入力で再評価した。fit/bootstrap、trajectory sampling、実QuantumCircuit/full build、transpile、native再synthesis、signal、N、N×C、全G、H6数値は0。Gaussian記述子を取得する実データ処理中も、実QC constructorとtranspileを禁止する検査を通った。

研究moduleのうち数値SOURCEとして用いたのはこの価格計数経路と固定局所関数だけ。旧science runner/authorization/grant関数は実行せず、旧runを再開していない。SOURCE/input/raw/価格/cache/grant/ledger/marker/KEEP/STOP、他worktree/process、共有環境を変更していない。

## 8. 全RPEへ足りないもの・最小の次案

**このM1を全13roundのcompiled価格providerとして採用しない。** 同δのPF q8/q16、PR q8の有限cohortだけでも一致しておらず、q4096・全law期待価格・long-q systematic・covariance/CIは未資格のままである。RPEのαtotal=.05を価格の95%保証へ読み替えない。

次の科学実行が必要なら、まず**追加compileなしの局所会計改善V2**を候補とする。元runtimeのbasis translationと1Q runのRZ数をSOURCEから対応付け、PF q8・PR探索9本の既存入力で、M1の等価分解選択差と未閉包interfaceを切り分ける。保存価格が既にあるため、現段階で新q全回路compileを要求する理由はない。新fitや固定offsetの校正を先行させる必要性も、今回の残差だけでは確定しない。

V2の予測・残差評価や局所再synthesisを行う場合は、処理範囲を具体化して別途科学実行許可を受ける。今回のV1では開始しない。V1の留保残差を使ってV2を設計・選択するなら、同じq16/33本を新モデルの未使用独立holdoutとして再PASS扱いしない。既知データでの再確認と、新しい独立domain検証を区別する。

その後、構造ruleの反復・境界閉包とlaw期待rewardが扱える場合は、長い全回路をcompileせず価格を延長できる可能性がある。閉包が得られない場合に限り、元設計§7のq1/q64等の最小反証案の情報価値を再判断する。今回の残差やD/DF12の寄与率からDF10の全費用順位や最適な追加qを決めない。N・α/β・候補最適化は別問題として維持する。

## 9. 最後の質問への回答とSTOP

**「H4 DF10の構造的価格モデルは、保存済みPF/PR回路のcompiled RZ countをどこまで再現できたか。全RPEへの拡張には何が不足するか。」**

元SOURCE・保存eventから、head、basis共有、対角/control、scalar、wrapper、限定した境界消去の会計までは実装・評価できた。しかし、今回固定した解析native展開＋限定RZ融合のM1は、開発20軸・留保68軸の全てでcompiled RZに一致せず、過小予測だった。PF q16の残差は各軸+9646、PR確認33本は各軸+736…+1960である。

不足するのは、**元compilerが実際に選ぶnative分解と、1Q/CXを含むinterface最適化の価格会計の資格**である。さらに全RPEには、長qでの閉包、PRのlaw期待価格とrare-event寄与、外挿systematic・相関の扱いが必要。本結果はその不足を有限データで示したもので、全lawや全RPEの価格保証ではない。

同じbranchへ成果をcommit/pushし、remote SHA・主要公開成果物を照合して**mandatory STOP**。次の担当は**ユーザー**：この限定モデルの不一致結果で区切るか、追加compileなしのV2を具体化・実行するかを判断する。**Codex**：新たな明示許可までV2、補正fit、追加compile、長q価格、全G、Main Claim変更、詳細科学レビューへ進まない。

[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10132af4a67ce112f79c539a9552f5ca0e83e516/docs/research/h4_df10_longtime_price_design_20261110/REPORT.md
[H46]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ffe8a71da98bec330ce1af109a9606f14e1fc598/docs/research/h4_df10_rpe_d1_20261110/REPORT.md
[M3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9fe5e6952c9fec576574560f0029e491c36bea0e/docs/research/h4_df10_rpe_cost_m3_validation_20261110/REPORT.md
