# B：参照AによるC-N限定価格cohort

基準`3258a89f225d9530fba38e9d2343bc0165dd6d7d`、SERVER 2、2026-11-06。
共通H4条件・保全は[共通REPORT](REPORT.md)を参照。

**予定39セル・2,307標本を完了。S2の2,304標本すべてで、full wrapperと強baselineはRZ/CX/depth/sizeの全ベクトルが一致した。今回の参照Aでは、取得した範囲でfull化の価格利得は観測されなかった。全RPEの価格移送と勝者はUNDETERMINED。**

## 1. 取得範囲

Partial S2 LD3、Full Y4 LD12、h=.02、Qiskit1.3.0・参照compiler A・rz/sx/x/cx・topology指定なし。
旧opt2の価格や未完了セルを流用・再開していない。
seed namespaceを`trackC_AB_3258_20261106`として独立cohortを作った。
full/strongは同じtrajectoryでpair、q8はq1/2と独立である。

| 対象 | 登録recipe・strata | 取得 |
|---|---|---|
| S2 LD3 | r=1,2,4,8,16,32、order2が一度以上ある/ないの2strata | q1/2各stratum32本、q8各stratum128本。36セル、2,304本 |
| Full Y4 LD12 | tail-free r0、決定論的 | q1/2/8各1回、3セル・3点。両軸を評価 |

rare stratumは複数order2も含む条件付き分布で、1個のorder2だけへ置き換えていない。
群確率を戻して平均し、未観測rare群を0補完しない。Bの価格samplerは記録したNumPy float64条件付きsamplerであり、Aの量子shot用integer-CDFとは区別する。
取得実時間は約907秒。科学SOURCE・seed・各標本の価格・時間を[raw bundle][raw]へ保存した。

## 2. 強baselineと資源境界

既存`trajectory_parts`・`d_part`・wrapperを再利用し、旧opt2呼出しを避けるoverrideでSOURCEを生成した。
新process内の旧`native_compile`は誤用時に例外を出すようにし、旧ファイル自体は変更していない。

S2は、R occurrenceを挟む最大D runをまとめてAでcompileした。
q>1の隣接するreverse Dと次forward Dを同じrunとして扱い、half-runの弱い単純和にしていない。
R occurrence内部も全体としてAでcompileする。prep/readoutも同じAを使う。
比較する両側に同じcompilerと、契約で認めた最大D run内の最適化機会を与える。
Full LD12の強baselineは既存定義どおりactual full wrapperなので、fullと同じ価格である。

RZ/CX/sizeは実native回路から数え、depthは**compile済みの各pieceを実際に連結したbaseline回路**のDAG depthを取った。
piece depthの単純和は補助欄`strong_serial_block_depth`に分け、primary depthと混同しない。
full wrapperは測定を含めてcompileする。状態準備費だけを除き、ancilla・basis・scalar・X/Y readoutを含む。
Tへ換算していない。

## 3. paired結果とholdout

[paired検査][paired]は全2,307点の全ベクトル一致を確認した。
S2については独立にcompileしたfullと強baselineを比較した負結果である。
Full Y4の一致はbaseline定義によるもので、独立した性能発見には数えない。
未取得qや別compiler・別最適化までの一般定理としては主張しない。

q1/2でaffine価格を較正し、q8を`7 C2-6 C1`として予測した。
q8を用いた再fitや追加標本は行っていない。

| holdout診断 | S2 LD3 | Full Y4 LD12 |
|---|---:|---:|
| 最大相対点誤差 | 6.0905% | 0 |
| 最大予測相対95%半幅 | 7.2624% | 0 |

既存の5%点誤差・2%予測半幅の目安をS2は通らなかった。
これらは局所・点ごとの診断であり、同時保証や数学的なmodel誤差上界ではない。
FullのMC幅0は決定論的な直接価格のためであり、q>8のmodel誤差まで0にしたものではない。

## 4. 全RPEへ接続した条件付き価格点モデル

primaryは保存shot数S2 8,598、Full 6,912を固定し、同じNをfullと強baselineに掛けた。
資源ごとに`sum_m,b N_m,b E[C_m,b]`を計算した。
同じ較正を複数roundへ再利用する共分散を、round間で独立とみなさず係数を先に合算して伝播した。
full/strong差にはpaired covarianceを使う。

| 資源 | S2 full=strongの条件付きG | Full Y4 full=strongの条件付きG |
|---|---:|---:|
| RZ | 4.52054e12 | 1.18703e13 |
| CX | 1.06840e12 | 3.71567e12 |
| depth費用 | 2.29576e12 | 7.71521e12 |
| size | 6.87215e12 | 1.86445e13 |

S2のRZ Gに対する較正MC標準誤差は約`1.95181e11`。
full/strongのpaired差は0、paired MC標準誤差も0だった。
このSEは標本covarianceからの推定値であり、未観測supportの母分散が0である証明ではない。
各資源の絶対MC標準誤差と局所holdout covarianceは[会計結果][accounting]に保存した。
いずれも未知の長q model誤差を含む総不確かさではない。
`sum N E[depth]`はdepth費用指標であり、量子wall timeや実並列実行時間ではない。

直接価格を持つ実RPE roundはm=0,1,3（q1,2,8、S2のr1）。
その3 roundだけの固定shot RZ小計はS2約`4.99314e7`（MC SE約8,703）、Full約`4.98103e8`。
それ以外を直接取得済みとは表示しない。
全18round表は、各distinct rのanchorからのaffine外挿であり、q131072までのmodel誤差上界はnull。
点値の大小から勝者を選ばない。

Aの新しい十分数はFullが6,914へ増えた。
同じ価格点モデルをそのshotで再重み付けした副次値は[JOINT_STATUS][joint]へ分けた。
FullのRZ点モデルは約`1.19012e13`となるが、長qの未資格は変わらない。
保存6,912をAの新しい証拠で保証済みと扱っていない。

## 5. STOP・次の判断

この参照Aでの有限cohortでは、強対照に対するwhole-wrapper価格利得が消えた。
一方、S2のholdout精度と長q model誤差は未資格であり、Partial対Fullの最終総コスト順位はUNDETERMINED。
科学的な負結果と、未解決の価格移送を分けて記録した。
標本追加・q>8 compile・別PF/stepの追加・C-S/C-P・T合成は行っていない。

Bの価格取得は一度で完了した。
保存rawを使った後処理で、初回会計に絶対MC標準誤差と直接域の小計を追加した。
初回SOURCE・会計もraw archiveへ保持し、価格cohortを再実行したものではない。
新しい管理基盤・承認証明・grant発行は作っていない。

**mandatory STOP。**
次はユーザーがこの負結果でC-Nを停止するか、精度を保つcompiler最適化など別の明確な問いへ範囲を限定するかを判断する。
今回の結果を救うために標本数や長q範囲を自動拡張しない。

[raw]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/B_cn_price/raw_prices.tar.xz
[paired]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/B_cn_price/analysis/PAIRED_CHECK.json
[accounting]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/B_cn_price/analysis/ACCOUNTING.json
[joint]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/JOINT_STATUS.json
