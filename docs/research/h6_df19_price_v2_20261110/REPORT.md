# H6 DF19価格モデルV2：保存補正・局所差分・固定δの独立q検証

**保存q1系のV1誤差は少数補正で大きく減ったが、その定数補正は固定δのq4へは移らなかった。q1/q2で凍結した簡潔なAffineはq4でPFを丸め誤差程度、PRの8本平均を約372 RZのbiasで予測した。全RPEへ使うには、q>4、未観測order2等の価格、有限cohortとsystematicの資格が不足する。** 保存確認点はV1で使用済みなので探索的比較、新q4は係数固定後の独立検証と区別する。

## 条件と範囲

Linear H6/隣接1.00 Å/STO-3G、保存tol-only 1e-8 DF19、weighted-Hermitian completion、12 system mode/Nα=Nβ=3・sector400。targetは保存DF-Hの固定Nα=Nβ=3 sectorでの基底状態エネルギー。保存Hamiltonianと数値stateを保持し、P=0。数値stateを厳密基底状態と認定しない。価格は測定付き完成Hadamard wrapperの1 interrogation・各X/Yのcompiled RZ。qは**qPF**で、RPEの2^mとは別である。

m5_bestの元順11 signed piece、元fragment順、scalar/identity規則、PRのfinite K2 lawを保持。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directional、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。h=.1、m0…12、εE=0.00159360010199040 Ha、αtotal=.05/方式、βtotal=.4・βstat=.2は変更・再評価していない。signal/N/G、ground-state solve、全RPE成功率、Main Claim、詳細科学レビューは対象外。

元成果と新作業の固定SOURCE・入力hash・private rawの対応は[SOURCE索引](SOURCE_INDEX.md)。同目的の既存branch `codex/resource-eval-h6-df19-price-design` を継続し、H4担当およびH6 signal/N担当のbranch/worktree/processへ書き込んでいない。

## A. 保存値による少数補正

V1 carrier `a4d743cb5` の保存予測・直接価格を使用し、旧174軸の構造計数を繰り返していない。H4 S1のpaired評価関数とC7-4のmultiresponse fitを再利用したが、H4/Track Dの科学係数は移植していない。

- M0: Ĉ=C1。
- Moffset: Ĉ_s=C1+b_s。qが変わってもbを定数として適用する。
- Mbasis: Ĉ_s=C1+b_s+c_s B（PRのみ、B=tail_basis_groups）。

PF Full19/q1/T.8の1本、PR_A p5/R32/K2/q1/T.8の探索9本、PR_B p7/R16/K2/q1/T1.6の探索9本で計5fit。X/Yは2応答として対応を保った。以下の係数は両軸で同じ（順にb, c）。

|族|モデル|係数 RZ|rank/列数|残差DOF|
|---|---|---:|---:|---:|
|PF_FULL19|Moffset|73236.00000000|1/1|0|
|PR_A|Moffset|23967.66666667|1/1|8|
|PR_A|Mbasis|24528.78110048, -27.90071770|2/2|7|
|PR_B|Moffset|28795.44444444|1/1|8|
|PR_B|Mbasis|31324.22222222, -172.41666667|2/2|7|

保存確認33本はV1で使用済みであり、V2の未使用holdoutではない。PF既知点もq1/T1.6で、開発点T.8とはδが異なる。既知データの最大軸RMSEが最小のものを探索的に選び、同点時は少数parameterを優先した。PF=Moffset、PR_A=Mbasis、PR_B=Moffset。新しい1/2/5%等の採用閾値は設けていない。

表はX。YのRZ指標は一致しており、独立標本としてpoolしていない。PR各33本、PF1本。残差は直接値−予測値。

|族|モデル|直接C平均|予測C平均|bias|RMSE|相対RMSE|最大絶対誤差|
|---|---|---:|---:|---:|---:|---:|---:|
|PF_FULL19|M0|428918.000|354760.000|74158.000|74158.000|17.2896%|74158.000|
|PF_FULL19|Moffset|428918.000|427996.000|922.000|922.000|0.2150%|922.000|
|PR_A|M0|141173.212|117206.455|23966.758|23979.475|16.9893%|25340.000|
|PR_A|Mbasis|141173.212|141196.667|-23.455|761.797|0.5384%|2269.065|
|PR_A|Moffset|141173.212|141174.121|-0.909|780.874|0.5521%|2349.667|
|PR_B|M0|179579.364|150948.182|28631.182|28636.568|15.9451%|29756.000|
|PR_B|Mbasis|179579.364|179707.053|-127.689|597.604|0.3333%|1469.389|
|PR_B|Moffset|179579.364|179743.626|-164.263|579.184|0.3229%|1354.444|

MAE、全相対誤差、path間残差SD、係数のnominal OLS共分散は[SAVED_CORRECTIONS.json](SAVED_CORRECTIONS.json)、各pathは[SAVED_PATH_RESIDUALS.csv](SAVED_PATH_RESIDUALS.csv)。共分散はIID・等分散を仮定した診断で、全systematicや有限bit law誤差の上界/総CIではない。PFの1点fitにはtrajectory SEを付けない。各designはfull rankだが、q1だけでは定数補正とq比例補正を識別できず、q方向傾きの検証にはならない。Bの係数を因果的なcompiler補正とは解釈しない。

旧PF q2/T.8/δ.4の補助点も別枠で適用したが、固定δ=1.6のq外挿証拠としては使用していない。旧V1の不一致判定・出力・STOPは不変。

## B. 局所構造とcompiler差分

保存one-bodyの最初のGaussian pairを固定し、2/3-qubitの4 fragmentを診断した。1Q再合成・interfaceに差が生じる具体例で、全回路の残差をこれらだけに帰属していない。

|局所fragment|V1解析kernel RZ|Qiskit RZ|
|---|---:|---:|
|XX+YY pair|6|8|
|pair＋元の2 phase|7|9|
|pair＋inverse pair|10|0|
|pair＋CX interface|6|8|

最後から2番目はprimitive列の診断で、V1のwhole-basis inverse相殺を無効と判定するテストではない。解析分解の行列差は約1e-15。compiler側にはraw行列norm差2.828/4があり、調べたfragmentではglobal phase i/−1による差として切り分けられた。raw不一致を消さず保存し、phase-only残差を別指標とした。生のvector一致のPASSや全wrapperの新しい正しさ保証へ格上げせず、compiler出力のphase補正も行っていない。[LOCAL_DIAGNOSTICS.json](LOCAL_DIAGNOSTICS.json)を参照。

既存の2/3-event境界補正は短stepのIID条件・制御構成・校正係数が異なり、H6の11 signed piece・piece別時間/分布・K2 lawへ数値係数を直接転用できない。native count・Z fold・literal basis cancellationはV1から再利用した。追加自由係数やcompiler全体の再実装、残差0を目的としたrule変更は行わなかった。局所差分だけでは固定δのq法則は確立せず、承認済みの限定pilotへ進んだ。

## C. 同一δ=1.6のq1/q2/q4

既存6価格表には同一候補q2/q4の直接価格がなかった。各新wrapperのfingerprintとcompiler hashが一致する場合だけ旧cacheを読取再利用する規則とし、共有cacheを変更していない。

|qPF|T|PF Full19|PR prefix7/R16/K2|用途|
|---:|---:|---|---|---|
|1|1.6|保存deterministic 1本|保存EXPLORATION 9本|既知基準点。確認33とはpoolしない|
|2|3.2|`3f813a46e17461bfe484`・各軸1|`72255c43facdfb8b4bee`・独立8本×2軸|追加校正|
|4|6.4|`201a8e365ab6f4af6ebc`・各軸1|`ef41b65942d56977d6cd`・別独立8本×2軸|新qの独立検証|

q4の対応候補は既存m6でN側のideal条件付きPASS済みだが、これは価格モデルの成功を保証しない。N/signalは再計算していない。

[PILOT_INPUTS.json](PILOT_INPUTS.json)で候補・cohort・seed規則・元application順・R16 allocationをsampling前に固定。新528 seed streamは相互非重複で、照合した旧H6 cohortとの重複0（全履歴の完全coverageやPRNG独立性の証明ではない）。同じwhole-pathをX/Yで共有し、量子shot数Nと古典価格標本数nを区別する。

q2完了後、保存q1とq2の直接価格から2応答Affine Ĉ=a+b qを校正した。係数・V2選択・解析script hash・比較指標を[Q4_FROZEN_MODELS.json](Q4_FROZEN_MODELS.json)に保存し、commit `b85837960c80d619a421c46235baf5ef2523d217` としてq4価格を見る前に固定した。PF: a=870, b=428048。PR_B: a=923.666666667, b=178659.666666667（両軸）。q4予測はそれぞれ1,713,062と715,562.333333333。q4による再fitは行わない。

新規H6 full-wrapper compileは36、旧cache hitは0。すべて予定内で完了し、全workerをjoinした。表はXで、YのRZ指標は一致。q2は校正データであり、そのAffineの平均一致は独立検証ではない。q4のみ新q検証点である。

|qPF|族|モデル|直接C平均|予測C平均|bias RZ|path RMSE|path相対RMSE|
|---:|---|---|---:|---:|---:|---:|---:|
|2|PF_FULL19|M0|856966.000|708732.000|148234|148234|17.2975%|
|2|PF_FULL19|V2|856966.000|781968.000|74998|74998|8.7516%|
|2|PF_FULL19|Affine|856966.000|856966.000|1.16415e-10|1.16415e-10|0.0000%|
|2|PR_B|M0|358243.000|300636.625|57606.4|57626.9|16.0828%|
|2|PR_B|V2|358243.000|329432.069|28810.9|28851.9|8.0502%|
|2|PR_B|Affine|358243.000|358243.000|0|1782.49|0.4975%|
|4|PF_FULL19|M0|1713062.000|1416676.000|296386|296386|17.3015%|
|4|PF_FULL19|V2|1713062.000|1489912.000|223150|223150|13.0264%|
|4|PF_FULL19|Affine|1713062.000|1713062.000|2.32831e-10|2.32831e-10|0.0000%|
|4|PR_B|M0|715934.125|601129.500|114805|114810|16.0362%|
|4|PR_B|V2|715934.125|629924.944|86009.2|86016.6|12.0144%|
|4|PR_B|Affine|715934.125|715562.333|371.792|1025.06|0.1431%|

q4の定数V2補正はPFで223,150 RZ（相対RMSE13.0264%）、PR_Bで平均86,009.181 RZ（相対RMSE12.0144%）の過小予測。q1だけの補正を長いqへ直接移すことは支持されない。一方、凍結AffineはPFの1,713,062 RZを数値丸め程度の誤差で再現し、PR_Bは予測715,562.333に対し8本平均715,934.125、bias371.792 RZだった。これは当該q4点の観測で、q法則の証明やq>4の保証ではない。PFの約2.33e-10 RZはOLS浮動小数点丸めで、残差0を採用条件にはしていない。

機械可読の[Q2直接価格](Q2_DIRECT.json)、[Q4直接価格](Q4_DIRECT.json)、[誤差・対応軸共分散](Q_VALIDATION.json)、[全path残差](Q_PATH_RESIDUALS.csv)を保存した。Affineのpath RMSEを個別pathの予測能力と解釈しない。


Affineはcohort平均の予測であり、各pathを予測するV1/V2とは単位が異なる。Affineのpath RMSEには個々の価格変動が含まれる。q1/q2の2点はa,bを決めるが、反復構造・端部・角度依存の全要因を識別するものではない。PRの係数・q4予測平均の共分散は、独立IID cohortを仮定した有限標本のnominal量として凍結ファイルへ記録した。PFの確定価格にtrajectory SEを付けていない。

## 資格、実施範囲、次の作業

保存回路での改善、新qでの観測、全RPEへの資格を分ける。8 trajectoryは限定反証pilotで、全sampling lawの高精度期待価格・希少event価格・qPF>4のsystematic上界を保証しない。未知のlaw/実装誤差や共分散を0に置かず、総CI・全RPE費用・全13roundのPF/PR順位は未確定とする。Normalization Bを価格へ再乗算していない。

[COMPLETION.json](COMPLETION.json)に実施範囲とprivate rawのSOURCE対応を記録した。実施は保存補正5fit、q1/q2の2応答Affine 2fit、H6の新36 full-wrapper compile、PR 16 whole-path（528 signed-piece stream、768 event）、限定局所compiler 6呼出し、人工parallel regression 4 compile。V1旧計数の再実行、bootstrap、新signal/N/G、ground-state solve、SCF/DF/state再生成、q>4価格取得、q4再fitは0。

新pilotの768 eventは全てorder0で、order2は0回。**order2の価格を0とは置かない。** PR_Aの旧order2例をPR_Bへ移植せず、全K2 lawへの期待価格保証は未資格のまま維持する。新full-wrapper statevector検証を実施したとは扱わない。

最小の次作業（未実行）：

1. **ユーザー**：同一δ=1.6のqPF=8/T=12.8への追加価格検証を実施するか判断する。候補案は同じFull19 PF各軸1、同じPR7/R16/K2の事前固定8 whole-path各2軸、最大18価格。新N/Gや候補探索は含めない。
2. **Codex**：許可後に既存同条件価格/cacheを先に照合し、不足分だけで固定Affineを新qへ検証する。未観測order2の寄与は、価格上界または限定診断の設計を別枠で示し、今回のcohortから全law保証を作らない。

追加compileが将来必要になる理由は、直接検証済みのq1/2/4だけではq>4のsystematicを拘束できないためである。このq4 pilotだけを根拠に、全13roundの検証済み価格providerとして採用することは正当化できない。PR_A p5/R32の長q検証も今回実施していない。

技術チェックは[TECHNICAL_CHECKS.md](TECHNICAL_CHECKS.md)。主担当自身の確認で、独立レビューではない。全wrapperの新statevector/物理signal検証を行ったとは記載していない。元の検証済みbuilder・sampler・provider・compilerを再利用した価格取得である。既存SOURCE/input/raw/result/grant/ledger/marker/KEEP/STOPを維持する。

公開成果のremote SHA/tree・主要blob照合後にmandatory STOP。新q、cohort追加、再sampling、N/G、全RPE費用、H4/H6他担当の処理へ自動進行しない。
