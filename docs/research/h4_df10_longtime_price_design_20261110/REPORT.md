# H4 DF10：長時間RPE compiled価格モデルの技術設計

2026-11-10 JST。リソース評価系列。**静的調査・未実行の検証計画。新fit・価格予測・残差再解析・sampling・実回路build/compile・N/G評価は0。** [固定資料・関数・候補の索引](SOURCE_INDEX.md)。数値価格は既存欄の引用だけである。

## 1. 結論

**第一候補は、DF10の実builderに沿った構造的RZ会計と、必要な場合だけDF10固有のcompiler差分モデルを組み合わせる方法。最初の科学検証は保存データだけで行い、新compile・fitを必須にしない。** Track Dの計数器・特徴量・共分散処理、Track Cのデータ分離を再利用できる。別研究の係数・P0互換資格は移せない。

- 同じδ=.1の直接価格は、固定PF B0/p7のqPF=8,16、固定PR p1/R4/K2のqPF=8に限られる。PRの探索9本・独立確認33本のresult/event原本は既知保存先に全て存在した。保存42本の構造モデル検査には新samplingが不要。
- PFの2点だけでAffineの2係数を決めても独立q検証は残らない。PRのq8だけでは、標本数を増やしてもq方向の傾きを識別できない。別δ・prefix・R/Kを説明変数なしに混ぜない。
- DF10向けの全13round compiled価格・長q誤差上界・同時95%価格区間は、今回の固定資料では未資格。既存m3の有限PR平均、構造的native count、全law期待compiled価格を区別する。
- 保存値による最初の反証検査を§6に具体化した。追加compileはその結果と目的に応じた§7の選択肢であり、現在の実行予定ではない。別Codexのm3 N×C検証は実施・取り込みしない。

## 2. 対象と、q・T・回路構造の対応

Linear H4、隣接1.00 Å、STO-3G、保存tol-only 1e−8 DF10/generation順、Nα=Nβ=2、保存DF-Hの同sector groundエネルギーをtargetとする。数値stateの既存条件付き性を保持し、準備費P=0。化学精度 .00159360010199040 Ha、αtotal=.05/方式、βtotal=.4 rad、h=.1、m=0…12を固定する。N、α/β、候補の再最適化は対象外。[D1][CONNECTION]

\[
q_{\rm RPE}=2^m,\quad T_m=.1\,q_{\rm RPE},\quad
q_{\rm PF}=\text{一回のinterrogation内のPF分割数},\quad
\delta_m=T_m/q_{\rm PF}.
\]

検討の基準族はPF **B0/S2/prefix7**とPR **S2/prefix1/R_outer4/K2**。PFは切捨てB0でありFull PFではない。δ=h=.1を固定する基準族ではqPF=qRPEになるが、これは任意候補の恒等式ではない。後半でもこの族がsignal/energy条件を満たすとは認定しない。将来の候補変更は価格domainも変更する。

元`h46_m5_plan.schedule`は、公式のsigned係数w_jに対してpiece時間w_j T/qPFを生成する。PRは`ABS_TIME_PER_OUTER`でR_outerを整数配分し、各piece最低1 step、1 outerの総RTE event数をR_outerに保つ。S2は1 piece、総event数はqPF R_outer。高次公式はpiece数・符号・整数配分を含む別族で、単にqやTを掛けるモデルに統合しない。

元`h46_long_time_native.compose`は各pieceをq=1として組み立て、順に連結する。PFはdeterministic native、PRは同head＋明示tailを使う。scalarは全Tに対して一回だけancillaのPへ集約し、PR/B0Iだけ抽出tail identityを加える。DF10のheadはblockごとのbasis逆変換→対角→basis復帰。PR tailでは既存basis planによる共有がある。Track D固有のhead相対融合・q−1境界融合がこのbuilderにも同じ形で実装済みだと仮定しない。連結後compilerが除去する境界を別に調べる。

一shot価格の必要キーは、保存H/DF/snapshot、fragment順・prefix、PF公式とsigned schedule、T/qPF/δ、R_outer/K・piece配分、finite-bit sampling law、実event順序とbasis plan、scalar/identity・control、軸・測定・準備費scope、builder/provider、compilerである。state/E0は価格算術の入力でなくても、将来Nとjoinするscientific bindingとして保持する。X=cosine、Y=sine、control-0 reference、H→evolution→(YのみSdg)→H→measurement。

## 3. 現在のcoverageと、別構成データの用途

下表は[305439c01の既存coverage][COVERAGE]を継承する。直接値があることは、全RPE成功や母平均価格の保証を意味しない。

| RPE域（δ=.1の基準族） | 固定PF p7 | 固定PR p1/R4/K2 | 今使える証拠 |
|---|---|---|---|
| m0…2、T=.1… .4、qPF=1,2,4 | 同時刻の直接値未確認 | 同左 | 短round端部の価格資格が必要 |
| m3、T=.8、qPF=8 | 直接RZ 44411/軸 | 探索平均16857.11111111111、n9／確認平均16947.78787878788、n33 | 当該wrapper・cohortの保存値。モデル不要 |
| m4、T=1.6、qPF=16 | 直接値あり | 同構成価格なし | PFは直接採用可能。PR q32/δ=.05で代用しない |
| m5…12、T=3.2…409.6、qPF=32…4096 | 同条件直接値・適格モデル未確認 | 同左 | 構造閉包または独立domain検証が不足 |

T=.8の同PFにはqPF=1,2,4,8、T=1.2には4,6,8,16、T=1.6には2,8,16,32、T=2には4,8,10,16,32の直接価格がある。固定PRにはT=.8/q8、1.2/q16、1.6/q32の価格がある。**これらはδが異なる。** 同Tでのq掃引をδ固定の長時間q掃引へ置換しない。

| 保存データの区分 | 校正・検証への具体的な用途 | 制限 |
|---|---|---|
| PF p7、δ=.1：q8/T.8とq16/T1.6 | source由来モデルの2点照合。Affineに使うなら2点校正 | 2点fit後の独立q holdoutなし |
| PR p1/R4/K2、δ=.1：q8探索n9と確認n33 | 前者を局所rule/補正の開発、後者を凍結後の標本外照合へ分離可能 | 同じqのcohort検証。長qへの外挿検証にはならない |
| PF p7、δ=.2：q4/T.8、q8/T1.6、q10/T2 | 別δでの反復・境界仮説を2点校正＋1点留保で検査可能 | target δ=.1の係数・直接価格ではない。T1.2/q6の保存δ=.19999999999999998は無条件に同一cache keyへ丸めない |
| PF p7のその他δ、PR p1/R4/K2のδ=.075/.05 | 角度・端部・law変化を明示した構造ruleのストレス検査。PR q32/δ=.05は探索n9と確認n33あり | δ変更でPRの次数確率も変わる。qだけの回帰へ混ぜない |
| B0I p7／Full p10、δ=.1、q8とq16 | scalar/identityやhead構成差へのrule転用を検査する補助族 | PF B0 p7の直接値、PRのtail価格へ代入不可 |
| その他prefix・R/K・高次公式の有限時間価格 | 明示した特徴量モデルが各差を表現できるかを族ごとに留保して検査 | 母集団を混ぜたq回帰、候補間の無根拠なpairing不可 |

元6価格表のmetadata照合では、S2/δ=.1のT.8/q8とT1.6/q16の両点が揃うのはB0 p7、B0I p7、Full p10だった。PRのこの2時刻対は揃わなかった。全私有資産の不存在を示す調査ではない。候補ID・selectorは索引に示す。旧alias価格は独立標本に数えない。

## 4. 再利用できる実装とcompiler条件

**元価格のproviderは `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。** 公開carrier155529e2のtreeだけでは実import元を表さない。scientific_provenanceのprovider_SOURCE、native/extension/continuationの各SOURCEを保持する。今回の基点treeにある同名moduleを無条件で実行版と扱わず、providerの関数本文も確認した。

DF10のcompilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。保存compiler hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。元関数はこれらを`transpile`へ渡し、その出力のRZを数える。**Track DのP0 pass構成・initially-zero等の追加条件は、共通hashやバージョン名だけでDF10へ同一認定しない。** DF12 opt2、Track C/newA compiler Aも別domainである。

| 既存資産 | 再利用単位・今回の判断 |
|---|---|
| `h46_long_time_native.compose/sample_requests/compile_cached`、providerの`ax2b_native_df_v5`、`df_rte_qiskit` | 科学構造と元compilerの正本。既存保存値・明示eventから辿る設計に使う。今回import/実行なし。旧cost runnerはsampling・build・simulation・compileを含み静的調査に呼べない |
| 元`compiled_cache`、canonical fingerprint再利用 | cache directoryの存在を確認。wrapperの角度・wire・順序・global phase・測定を含む実fingerprintが一致する保存価格を再利用できる。新q/Tは通常別wrapper。局所hash再利用は価格の加法性を証明しない |
| `track_d_native_resources.NativeCounter/NativeInstruction`、transfer/max-plus | 確定rz/sx/x/cx/measure列の整数countと三depthを再利用可能。zero RZも入力列にあれば数える。未lowered DF10を渡せず、P0後価格へ自動昇格しない |
| Track D native emitter・stream bridge | 1Q/CX界面、phase、境界ruleの参考。保存CDF8/K0のq4/q8での有限一致はDF10/K2の資格ではない。全列走査・O(L)履歴が残り、長q高速化の完成品として移植しない |
| C7-4 `event_features/design/fit/residual_summary`、C6aのpaired統計 | 全path単位の分割、絶対価格＋軸間共分散、低次元回帰の方法。D専用labels/descriptor/loaderをそのままPF/PRへ呼ばず、適合する小さい関数を再利用する。fit/bootstrapは今回0 |
| Track C `paired_paths`、価格MC・shared-cell会計 | 訓練と最終holdoutを分け、共通cellの係数を合算してから共分散を取る方法。modular加法価格やserial depth上界をDF10 whole-compile値と混同しない |
| 新A V3/V5の構造会計・回転分類 | D/basis/event/boundary/scalar/readoutの内訳と全wire frontierを参考にする。厳密zero・special angle・binary64由来を保持。RZをTへ換算しない |
| DF12 V3/R1 | 同一tupleのdirect/model/holdout表、共有anchor・systematic感度と情報価値判断。既存残差・寄与率・係数をDF10へ移植せず、再実行もしない |
| `h4_df10_rpe_connection.aggregate_supplied_axes`、cost connectionの`account_subset` | 将来の軸別N×Cと部分/全round区別を再利用。今回の調査で会計器の追加実装は不要 |

元`compile_cached`は保存領域内でfingerprint名を使い、missなら実compileへ進む関数である。**cacheの有無確認にこの関数を呼ばない。** 将来は保存metadataから読取選択し、compiler・scope・fingerprintとの対応が確定した既存値だけを再利用する。元cacheへ新profileを混在させない。全cache/QPYの照合完了や任意qのhitは未確認。

## 5. モデル候補と成立条件

### A：構造的会計、B：Affine、H：構造＋必要最小限の補正

構造案の記法は、方式s・軸a・path ωについて

\[
 C^A_{s,a}(q,\omega)=W_{s,a}(T)+qD_s(\delta)
 +\sum_i R_s(e_i)+\sum_j B_s(\text{interface}_j)
 +E_{s,a}(\text{ends},T).
\]

これは**設計上の分解**であり、既に正しいcompiled式が得られたという主張ではない。重複するscalar・wrapper・basis共有・step融合を一度だけ会計し、非局所最適化を局所境界項へ押し込めない。まずlocal primitiveの非零角、Gaussian/basis列、event次数・product列、head/tail・step境界と端部を分ける。count用の加法transferは適格な反復単位なら圧縮可能だが、P0が境界外へ影響しない閉包が別途必要。max-plusはdepth用で、RZ加法性の証明ではない。

δ/R/Kとlawが固定され、compiler後価格が同じ有限範囲のbulk/境界rewardに閉じる場合、定常な内部期待rewardの和はqやq−1に比例し、Affineの構造的根拠となる。必要な界面長より短いqは別の端部式を要する。scalar P(−cT)・X/Y readoutの端部角分類は別に評価する。PFでも端部1Q最適化、zero/special angle、fusionの到達範囲が変われば単一Affineは厳密ではない。PRでは固定lawの期待値と、指定event列の価格をさらに分ける。

H案は`C^H=C^A+x^Tθ_DF10`、B案は方式・軸別`C^B=a+b qPF`を比較候補とする。Hの補正が不要ならfitを省略する。bulkとq−1境界を全て自由係数にすると定数/qと共線になり、別々の因果係数は識別できない。C7-4の同じ問題を再実装しない。

| 検証設計の項目 | A：構造 | B：Affine等 | H：組合せ（優先候補） |
|---|---|---|---|
| 1 入力・特徴 | source由来primitive、角分類、basis/run、境界、端部、明示events | 固定族のqPF。δ/R/K変更は別fitまたは明示特徴 | A＋未説明interfaceの少数特徴 |
| 2 予測資源 | まずRZ。nativeとpostcompilerを別出力 | PF/PRそれぞれの絶対compiled RZ、各軸 | 同左。paired差だけの予測にしない |
| 3 適用条件 | 同H/DF・実lowerer・compilerと界面条件 | 同scientific binding・δ・law・compiler | 同左＋補正の検証済み特徴範囲 |
| 4 直接値対応 | 保存whole-wrapper価格とpathごとに照合 | 同tuple/cohortの保存価格平均 | 構造残差を同じpathで照合 |
| 5 校正 | source由来ruleならfitなし。価格からruleを調整した点は校正扱い | PF q8/16は2係数校正可能。PR q8だけでは不可 | PR探索9本など。q一定ではq係数を自由にしない |
| 6 留保 | PF q16を未調整のrule検査に、PR確認33本を標本外へ | PFの追加独立qが必要。PRは別q校正とdomain holdoutが必要 | 同左。確認cohortを調整に使ったら留保資格を失う |
| 7 保存データだけの範囲 | 明示eventsとsourceの局所会計・既存価格比較 | 他δの補助族での反証、targetの不足確認 | 同qのpath差・軸差・局所残差の検査 |
| 8 新実行が必要な範囲 | DF10入力からの新計数/価格予測自体も別の科学許可。原本不足なら取得 | 新fit/残差評価。必要なら別q直接値 | 新特徴・補正fit。必要な時だけ直接compile |
| 9 誤差・依存 | 未閉包界面、zero/角度、未観測次数を明示 | 校正変動、外挿bias、共有係数、law差を分離 | joint coefficient/path/axis共分散を保持 |
| 10 Gへの接続 | 認定済み価格域だけ使う | 同N固定で価格誤差を加重 | 既存会計へdirect/modelと資格域を明示して渡す |

`sqrt(q)`はDの二lane depth揺らぎを動機とする探索案で、DF10の絶対RZへの第一候補にはしない。basis-runは共有変換を表す有力な特徴だが、未来の特徴lawが必要。compiled DAGから逆算した特徴を予測入力に使わない。depthへ拡張する場合もE[max]をmax(E)へ置換しない。

## 6. 最初の最小モデル検証案 V1：保存値だけ、fit/compileなし

**提案・未承認・未実行。** 次に実行を選ぶなら、全q直接compileやAffine fitより先に、構造案を反証できる以下の一campaignを推奨する。現在の依頼はこの設計までを許可している。

1. **入力固定**：155529e2の保存H/DF/snapshot/preparation metadata、実provider c31ef424、元native/schedule、同compiler、PF B0/S2/p7、PR S2/p1/R4/K2。PF q8/T.8とq16/T1.6、PR q8/T.8探索9・確認33を使う。private event/result原本の存在は今回確認済み。NPZ内容・全QPYの再検算は未実施であり、必要原本が不足すれば代替生成せず不足を報告する。
2. **source由来ruleを固定**：既存counterに渡すコンパクトなDF10 primitive/境界descriptorを、実builderから定める。実QuantumCircuitの全build・再compileをせず、保存eventsと保存prepared入力に沿った局所会計を行う。この実データ計数・予測は新科学処理なので明示許可後だけ実施する。既存emitterを無理にD専用入口へ偽装せず、必要な薄い接続に限る。
3. **開発と留保**：PF q8とPR探索9本だけを開発用とする。PF q16とPR確認33本はrule固定後に評価し、見た後のrule変更を同じholdoutの再PASSにしない。既存の集計値は公開され今回も既知なので、完全な盲検・前向き事前登録とは呼ばない。独立seedの旧cohortを使う外部照合であり、新q検証とは区別する。
4. **指標**：PF/PR各方式・X/Yの絶対RZ、保存価格との差の符号付き整数、最大path差、native→postcompilerの差分と出所を記録する。PRは同path X/Yを一vectorとして扱う。既存cohortとの価格一致が目的で、N×C・N solverは呼ばない。price値を見て補正を逆算した行は校正に移す。
5. **判定・STOP**：厳密なbridge一致を主張するには全留保回路の整数差0が必要。ただし有限一致だけでは全q/全lawの定理にならない。差があればnativeとcompiledのどこで生じるかを記録し、exact価格としての採用を止める。fit・閾値変更・新sampling・compileへ自動拡張しない。原本/条件不一致、再生成の必要、反映できない角度/界面があっても当該移送を未資格とする。
6. **成果物**：小さい照合表、SOURCE/入力参照、ruleと留保の役割、未解決界面、次の判断だけ。fit0、sampling0、実回路build/compile0、signal/N/G0で公開照合後STOP。全lawの厳密期待価格を得たとは記載しない。

このV1はm3のN×Cとは別処理である。保存価格の再取得や既存会計テストの反復は不要。元cacheや科学SOURCEを変更せず、原本から計数用の読取接続を作る。

## 7. V1で不足が残る場合だけ検討する追加科学計算

**以下も未承認・未開始。V1と一緒に自動実行しない。**

| 必要な判断 | 最小の追加候補と情報価値 | 限界・STOP |
|---|---|---|
| 局所補正で十分か | 保存探索側の構造残差だけで必要な少数係数をfitし、留保側を一回評価。DF10の絶対PF/PR応答を保持 | q8のみでPRのq傾きをfitしない。留保後に変更したら新検証が必要。改善しなければ単純なモデルとして区切る |
| 初期端部と長q外挿を反証できるか | δ=.1固定でPF q1/T.1・q64/T6.4各1回路、PR同2点各8 whole-path、各X/Y。PF4＋PR32＝**最大36 wrapper compile**。保存q8/q16は再compileしない | n8は構造・外挿のpilotで、高精度母平均保証の標本数ではない。q64適合でもq128…4096は未検証。費用順位を確定する実験とはしない |
| PR Affineを比較するか | 上の新q1と保存探索q8だけを校正へ使い、同q8確認33と新q64を留保。PFは保存q8/16校正、新q1/64留保。仕様と役割は取得前固定 | q64で選び直さない。選択に使えばholdoutではなくなる。q1のpath数でprecision不十分なら未確定のままSTOPし、取得後top-upを自動実行しない |
| 最大qまでの誤差を制限したいか | 有限界面の閉包とlaw期待値を解析的に評価できるならfull compile不要。できなければ、固定Nがある段階で高情報価値の後半qを一つ選び、DF10固有の反証計画を別途具体化 | q64の残差や他研究の寄与率を全域の上界へ移さない。q4096を含む全候補直接compileを既定にしない |

36という値は上記scopeを選んだ場合の回路数で、資源利用上限ではない。新seed/cohortを既存seedから分離し、両軸を同じpathで取得する。PFとPRは異なる回路・lawなので、既存seed番号が等しいだけでpaired価格差の標本にはしない。必要な正しさ確認と元compiler条件を保ち、別Hamiltonian・R/K探索・signal・shot・全Gは追加しない。科学実行SOURCEと具体的入力・既存STOPとの関係を定め、**ユーザーの明示許可を受けた範囲だけ**実施する。新しい承認証明基盤は不要。

## 8. 信頼水準・全law期待値・総費用への接続

RZの一shot期待価格はPRでは対象lawに関するE[C]であり、固定cohortの標本平均とは別である。保存PCG64/NumPy float64 CDFとideal finite-RTE lawの差、未観測order-2等の条件付き費用を保持する。稀な次数の寄与は「確率×条件付き価格」で別に扱い、未取得条件付き価格・tail boundを0としない。解析期待値を出せるlocal rewardならsamplingを減らせるが、その導出・新数値評価も今回行っていない。

| 証拠水準 | 現在／将来言えること |
|---|---|
| 既存直接価格 | PFは指定compilerの指定回路の整数count。PRは指定n本の保存平均。sampling精度や全law保証を同一視しない |
| V1が整数一致した場合 | 指定PF点・有限PR留保cohortでの構造モデル支持。全13roundや未観測eventへの普遍保証なし |
| DF10固有fit＋独立q検証 | 検証domain内の経験的proxy。whole-path標本/共有係数の変動を評価できても、長q systematicは別に必要 |
| 全13roundの95%価格評価 | model bias・law差・rare tailを含む根拠が必要。現時点では未取得。正規近似/ bootstrap幅だけで認定しない |

価格の名目95%統計診断を将来採る場合は、価格用のfamily・信頼配分を別に固定する。RPEのαtotal=.05はenergy測定の失敗確率であり、そのまま価格モデルの信頼水準ではない。n9/33や上のn8を根拠なく95%精度へ換算しない。

Nが将来揃えば、既存adapterへ各方式の`m/axis/qPF/qRPE/T/δ/prefix/R/K/law/compiler/scope/N_origin/C_direct_origin/model_origin/domain/uncertainty`を渡す。直接値を優先する採用規則と、modelの校正域・独立検証点・外挿域を別列に残す。有限cohort proxyを使う場合はその条件を明記する。未知N/Cは0で埋めず、全13roundが揃うまで全費用をnullにする。

\[
\widehat G_s=\sum_{m,a}N_{s,m,a}\widehat C_{s,m,a},\qquad
\Delta G_{s,m}=\sum_aN_{s,m,a}(C_{s,m,a}-\widehat C_{s,m,a}).
\]

統計部分は固定Nのvector wについて`Var(Ĝ)=wᵀΣw`。同じpathのX/Y、同じ校正係数を共有するround間、直接anchorとfitの交差共分散を保持する。不明なcross-covarianceを独立0にしない。比較で共分散が不明なら条件別成分を示し、total CIはnullとする。

検証済みの一様価格誤差幅bが本当にあれば`|ΔG|≤Σ N b`で費用影響へ接続できる。短qの最大残差は未検証長qのbではない。後半検証の優先度は将来のN、価格感度、共有誤差の方向と未資格domainから決める。N未取得の今はq1の端部とq64の反復長を選ぶ構造上の理由だけを示し、DF10の寄与率・順位・最適な次qは計算しない。D/DF12の後半寄与率やR1の許容誤差率を移植しない。

## 9. 最後の質問への回答・公開とSTOP

**「H4 DF10の全RPEラウンドのcompiled価格を、どのモデル・既存データ・最小追加検証で、どの信頼水準まで評価できるか」**

現時点で直接利用できるのは上記の疎なDF10価格で、全13roundの期待compiled価格を95%等の水準で保証できるモデルはない。最も有力なのは、**DF10実builder由来の構造会計＋必要な時だけ局所compiler差分の補正**。最初は保存PR探索9/確認33とPF q8/16を使うV1を、新fit・新compile・samplingなしで行う。これにより有限cohortでの適合と未閉包界面を判定できる。純AffineはPRの同δ別qが不足するため、今すぐfit可能な解として提示しない。

V1と構造閉包で足りる部分に追加compileを要求しない。足りない場合だけq1/q64の最大36 wrapperという反証pilotを選択肢にするが、それだけではq4096までの一様誤差・全law・同時CIは得られない。全域の強い評価には、境界閉包とlaw期待値の根拠、またはNが揃った後の情報価値に沿った後半domainの検証が別途必要である。

本調査は既存branch/worktreeを確認した上で、並行m3作業との競合回避と科学成果の分離のため、固定305439c01を親とする`codex/resource-eval-h4-df10-longtime-price-design`を新設した。D/C/新Aの別枝は固定Git objectで読み、merge/削除/履歴変更は行わない。新成果物は本REPORT・SOURCE_INDEXと入口リンクだけ。科学コード・入力・raw・旧結果・grant・ledger・marker・KEEP・STOPを変更しない。

今回実施した確認は、コード本文・保存欄の読取、candidate/条件のmetadata対応、既知42 result/eventファイルとcache directoryの存在確認、成果物の参照/差分/公開照合である。新しい科学数値評価・科学test・研究module import・QPY decode・独立科学レビューは行わない。文書のみのため重複する人工テストや価格基盤の実装を追加しない。

**GitHubへcommit/pushし、remote SHA・公開成果物を照合してmandatory STOP。** 次は**ユーザー**がV1の実データ計数・価格予測・照合の実行可否を判断する。**Codex**はその明示許可後だけV1を実施し、fit・新compile・全RPE費用・Main Claim変更・詳細科学レビューへ自動進行しない。

[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ffe8a71da98bec330ce1af109a9606f14e1fc598/docs/research/h4_df10_rpe_d1_20261110/REPORT.md
[CONNECTION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/305439c01b6e9d8ec264045585ad77b36630275c/docs/research/h4_df10_rpe_cost_connection_20261110/REPORT.md
[COVERAGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/305439c01b6e9d8ec264045585ad77b36630275c/docs/research/h4_df10_rpe_cost_connection_20261110/COVERAGE.json
