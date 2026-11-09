# Canonical finite-bit producer：SOURCE契約と条件付き分布保証

[実装](../../../src/trottertracks/pr_reference_v6q_producer.py)／[REPORT](REPORT.md)。固定条件は既存H4・Partial S2/LD3/h=.02/K2、保存18round、canonical independent finite-bit law。これは新しいsampling lawではなく、既存部品の接続である。

## 1. 固定SOURCEの再利用

| 部品 | そのまま使う定義 |
|---|---|
| [integer_cdf／draw_cdf](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/track_c_h4_finite_bits.py#L12) | nominal binary64値をFractionへ移し、exact rational正規化、累積floor、最後のCDF値2^b、bisect_right(cdf,getrandbits(b)) |
| [_make_event](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/rte.py#L1594) | 最初のindexがrotation、残りがordered products。回路適用順はproducts→rotation。phase、angle、normalization、basis metadataを既存constructorが決定 |
| [旧events_forのcfg/dist生成](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_qualification.py#L143) | 同じtail・λ、h=.02、保存r、K2、同じtolerance式とmake_rte_config。NumPy条件付き抽選部分は使わない |
| [保存schedule caller](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5fdb68e4fc59f4e4c95270381043d46f074bff35/src/trottertracks/pr_reference_v6q_schedule.py)／[固定cache consumer](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87e9b11a672e380915866a73253be100e75ff427/src/trottertracks/pr_reference_v6q_consumer.py) | m/q/r/bits照合、同一prep、F/G・B1・X/Y、Compiler A／compile後reset |

probabilityを先にfloatで割り直したり、順序をsortしたりしない。入力componentの順序をtupleへ保持し、同じcomponent objectと同じdistributionをconstructorへ渡す。CDFの正のsupport消失は既存helperの例外で止まり、条件付き再抽選や代替precisionへ進まない。

## 2. 単一drawとevent kernelの法則

任意の一方のcategorical分布について、p_iは既存SOURCEのnominal binary64を実数としてexactに解釈した値、s=Σp_i、L=2^b、C_-1=0、C_i=floor(L Σ_(j≤i)p_j/s)、最後C_(d−1)=Lとする。uが{0,…,L−1}の一様整数なら、bisect_rightはC_(i−1)≤u<C_iを満たすiを返し、

\[
\Pr(i)=\frac{C_i-C_{i-1}}{2^b}.
\]

境界は右側categoryへ進み、空のcellは選ばれない。この法則は既存integer_cdf／draw_cdfそのもので、producer側で再実装していない。

order用の質量をa_o、component用をc_iとすると、EventKernel.drawはorder drawを1回、選ばれたn∈{0,2}に対してcomponent drawをn+1回行い、そのindexを一度だけ_make_eventへ渡す。**各wordが過去の履歴を条件としてもfresh uniformである**という固定モデルの下で、

\[
\Pr(o,i_0,\ldots,i_n)=a_o\prod_{j=0}^{n}c_{i_j}.
\]

これはSOURCEのordered選択結果を指定する確率である。同じ演算子へ複数index列が写る場合はその確率を足す。順序交換・commutation・operatorへの再集約を実装へ追加していない。

_make_eventのevent_probability／event_coefficientは**旧nominal SOURCEのmetadata**のままで、上の実際のfinite-bit cell質量へ書き換えない。event_normalizationも旧RTEのBである。これらのmetadataを逆確率weightとして再利用したり、新しいimportance reweightingを導入したりしない。nominal lawとの差は旧finite-bit proposal diskが覆う。

## 3. 列・shot・axis・roundの独立性

1 eventのword数はorder0で2、order2で4。orderで停止長が変わっても、IID word列の未使用suffixは履歴から独立なので、続くeventにも同じkernel法則が成立する。最大4qr drawsという旧budgetと一致する。

CanonicalFiniteBitProducerは外部bit sourceを明示して作る。同じsourceを持ち続け、各supplier呼出しで新しいevent tupleを生成する。event列・乱数値をcacheせず、seed、setstate、rewind、再samplingを呼ばない。component/order CDFはSOURCEの値から作るが、選んだeventを再正規化しない。

| 水準 | 固定モデルで必要なこと／実装の扱い |
|---|---|
| event内 | order／各componentがfresh uniform wordから選ばれる。draw呼出し回数・順序はSOURCEで確認 |
| trajectory内 | 旧q*r順にkernelを繰り返す。前eventを複製しない |
| shot間 | supplierを再呼出しし、継続中のsourceから新しいwordを読む。seedを同じ値へ戻さない |
| X/Y間 | consume_canonical_axisを物理axis-shotごとに呼ぶ。旧consumerが共有coreから作る二つのtemplateのうち一方だけを返し、他方を次shotへ流用しない |
| round間／worker間 | 同じ条件付き一様・独立モデルが必要。再seed、PRNG stateのcopy、重複streamを実行側が導入してはいけない。新しいstream分割方式は今回採択していない |

実装が行うのは新しいgetrandbits呼出しとstateを戻さないこと、返値のinteger/range検査である。**getrandbitsという名前、sourceのclass名、有限テストから一様性・数学的独立性は証明しない。** 具体的なPRNG／entropy sourceは既定で選ばず、新しい科学的乱数標準も採択しない。実供給器の性質と固定確率モデルの仮定を区別する。

物理executorは1 axis-shotごとにproducerを再呼出しする必要がある。一度生成したtrajectoryの回路をbackendのshots=Nで反復し、そのNを独立なPR標本として会計する運用は、この契約に含めない。独立なdrawでも同じ値やevent列が偶然得られることは許し、一致を理由に再抽選しない。独立性は出力の相異性ではなく供給法の条件である。

make_rte_configが返すRTEConfig.seed／prng_typeには旧APIのdefault metadataが残るが、それを使用してRNGを初期化しない。今回の乱数SOURCEは明示されたgetrandbits objectであり、そのtypeをaxis-shot返値へ記録する。旧cfgのPCG64表記やseedを、このproducerの実際の乱数方式と解釈しない。

## 4. 既存保証とconsumerへの移送

既存proposal_budget／[SOURCE保証](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/track_c_source_guarantee.py)は、同じexact-rational正規化・floor CDFとfresh independent bits、最大4qr drawsを前提にfinite-bit proposal diskを計上している。本producerはそのkernelを実装し、bit幅、range、normalization、importance weightを変更しない。新たなdisk数値を計算せず、既存項を二重計上／消去しない。

固定cache／tail・λ／basis registryと適格event文法、旧correctly-rounded binary64 add/sub/remainderモデルを保つため、Q3-Nのℓ=65、normal域、保存M／B_norm、B1のexact和＋最後1回RNへ同じ条件下で接続できる。単一eventのconstructor一致、kernel法則、coherent meanのfinite-bit摂動、全support位相保証、有限shot十分条件は別の論拠であり、人工テスト一つで代用しない。

supplierは既存consume_saved_roundへ実際に接続し、同じeventsを旧assembleへ渡す。m17明示opt-in、他17round legacy、F/G、actual SOURCE inverse、relative P、D/basis、Compiler A／resetは不変更。既存consumerの`NOT_ESTABLISHED_BY_THIS_INTERFACE`は保守的な自己申告の扱いとして残し、qualificationはこのSOURCE対応と仮定から説明する。

consumerは同じcoreからX/Y templateを二つcompileするが、axis-shot helperが返す実行対象は指定した一つだけである。未使用templateの古典compile overheadは今回測定しない。量子費用へ二軸を二重に計上せず、次の物理shotではproducerを再呼出しする。本番shot loopやbackendは新設していない。

同じCompiler A contextではP角以外に新しいgate／passはなく、resetも同じ位置・8本である。従ってnative RZ単価不変性を保ち、V6-Pの保存626 shots/axisと全18round総RZ不等式へ**元の固定モデル内で**移送できるSOURCE上の条件が揃う。実乱数器の無条件IID、実機RPE、CT/T-depth、別cache・compiler等への一般化は認定しない。研究成果の正式採択はGPT／ユーザーが判断する。
