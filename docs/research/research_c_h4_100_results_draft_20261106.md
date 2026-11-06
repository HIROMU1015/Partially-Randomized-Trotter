# 研究C：H4・1.00 Å Results初稿

2026-11-06。**日本語の作業稿**。最終原稿言語・投稿先・中心claimの正式採択は未定である。既存[論文構成案](research_c_paper_outline_figure_plan_20261105.md)の§4〜7へ対応し、根拠と先行研究差分は[証拠対応](research_c_h4_100_paper_evidence_20261106.md)、図の完全条件・候補identityは[caption/図表索引](../../artifacts/research_c_h4_paper/2026-11-06/captions.md)へまとめた。数値は閲覧commit `417fa459b823509be5c2010c5836e58955088282` の保存値を引用する。

## 7.1 Numerical setup

直鎖H4の隣接距離1.00 Å、STO-3G基底を対象とし、number-CDF rank8、元のframe順に基づく決定論prefix L_D=2を固定した。物理精度はNα=Nβ=2の36次元sectorで評価した。PR-aware DFはtensor残差と、実際にサンプルする非identity Z/ZZ atomへの写像後tail massの平滑化二乗を組み合わせる。対照ENTRYは、同じtailのJ-entryに三角上界に対応する重みを与える平滑化L1二乗目的である。一般signed core、frame別直交回転、number→Pauli写像、係数正則化という基盤は既存CDF/RC-DFを踏まえる。両目的の違いは、entryを絶対値化する前に行和の相殺を保持するかという固定tail辞書上の違いであり、一般表現自由度を新たに導入したものではない（Fig.1）。

共通neutral start C0と事前に固定した摂動start P1/P2を用い、各目的に初期正則化寄与.03、.1、.3を与えた。各trajectoryについて16および64受理更新のendpointを比較した。全36 checkpointは共通のDF物理予算.00019875 Haを満たした。J一次基準はすべてMET、U一次基準はすべてNOT_METであり、以下は収束解の能力比較ではなく、登録した有限更新手順の比較である。共通の受理更新数は、実際の古典計算量や到達物理誤差を等しくする条件ではない（Supplement S1）。

下流ではpartial ordered symmetric S2、s1、δ=h=.1、K0、RPE round m=0,...,12を固定し、r={16,32,64}を比較した。候補固有のown-DF ground oracleを仮定し、状態準備費用を除外した。目標精度は.00159 Ha、RPE failure予算は.01である。Native価格はP0（Qiskit1.3.0、optimization level1、seed17）の位相保持compileとfenced serial会計で評価した。主指標は、反復重みW、kernel平均RZ、shell費用を分けた条件付き総RZである。

$$C_{\mathrm{total}}=W\,\overline c_{\mathrm{kernel}}+S_{\mathrm{shell}}.$$

共通r32では各armの3強度から選び、公平retuningでは双方に3強度×3rの同じ選択機会を与えた。各適格policyの選択用native32で候補を固定した後、独立した最終native64を評価した。108 policyのうち90がnativeへ適格となり、選択用2,880標本、22個の選定(candidate,r)ペアに対する最終1,408標本、同一J/Uの強決定論19候補を取得した。最終標本を選択へ戻さず、旧cohortともpoolしていない。

統計的な主比較はC0・64更新のPR−ENTRY差であり、元の事前登録familyに従う98.75%近似Welch差区間を使用した。他の比較の個別95%区間、同J/U決定論との副比較の個別nominal95%区間とは区別した。近似区間は固定N・IIDモデルに基づき、被覆率や希少tailを検証したものではない。以下の総費用はexact native synthesis、ideal finite-RTE IID信号、finite-bit標本価格をideal-law価格のproxyとする仮定に条件付けられる。

## 7.2 PR-aware versus ENTRY

C0・64更新で、共通r32の条件付き総RZ点推定はPR-awareが93.727×10^9、ENTRYが191.600×10^9となった。保存されたPR/ENTRY点比は.489180で、PR-awareの点推定は51.08%低かった。このregimeでは両armともstrength=.1を選択した。保存されたPR−ENTRY差の98.75%近似Welch区間は[-103.632,-92.114]×10^9 RZであり、登録精度目標を満たした（Fig.2、Table T1）。

両armに同じrとstrengthの選択機会を与えると、PR-awareはstrength=.1/r16、ENTRYはstrength=.03/r64を選んだ。独立final64の総RZはそれぞれ93.007×10^9、170.560×10^9、保存点比は.545306となり、公平retuning後にも45.47%の低い点推定が残った。対応する98.75%差区間は[-82.797,-72.308]×10^9 RZで、保存判定は両regimeとも`NOMINAL_MODEL_PR_LOWER`である。1.15 Åの追加検証を終了した後も、元の主familyに基づく区間を変更していない。

この結果は、固定r32における対照の調整不足だけでは今回の有限比較の差が消えなかったことを支持する。ただしENTRYのstrengthもregime間で変わるため、二つのregimeの差をr変更単独の因果効果と解釈しない。また、これは本研究の近接ENTRY対照との比較であり、RC-DF原法やUWCの最良実装との比較ではない。

## 7.3 Robustness across initialization and finite budgets

公平retuningでは、3初期値×2更新予算の全6比較でPR-awareの総RZ点推定が低かった。保存された削減率はC0で16更新43.71%/64更新45.47%、P1で59.57%/60.96%、P2で53.71%/55.16%である。各セルは独立final64に基づき、C0・64更新以外の成立セルでは個別95%近似Welch差区間の保存判定もPR側が低い結果となった（Fig.3）。

共通r32ではC0とP2の両予算で比較が成立したが、P1の2セルは未成立だった。P1/ENTRYでは、両予算・全3強度において最終round m12が`INFEASIBLE_PHASE`となり、適格なENTRY/r32候補を選べなかった。r64では適格となり、公平retuning比較は成立した。未成立のr32セルをPRの勝利や価格ゼロとして扱っていない（Supplement S2）。

これらはH4・1.00 Å内の三つの固定初期値、二つの有限更新予算、事前に定めたstrength/r集合からの選択後における観測である。初期値分布全体に対する頑健性、すべてのstrengthで独立最終評価した優位、あるいは最適化予算を増やせば差が一定方向へ変化するという収束傾向を意味しない。

## 7.4 Resource tradeoffs

主C0・64更新・共通r32のexact tail λは、PR-awareで.498794143081 Ha、ENTRYで.945068647756 Haであった。同じ保存参照のm12では、attenuationは.852801/.564612、補正RTE operator誤差は.033733/.083891、PF operator誤差は.086558/.136434である。これらは候補固有の数値参照であり、実装全lawの誤差保証ではない（Table T3）。

共通r32では、PR-awareのkernel平均RZは7,484.734375とENTRYの6,599.703125より高い。一方、モデル上のshotsは14,732対19,746、反復重みWは12,522,400対29,031,600であり、小さいWが高いkernel単価と競合した結果、PR-awareの総RZが低くなった。shell費用51,562/69,111 RZも別に計上している（Fig.4、Table T2）。

公平retuning後には交換の形が変わる。PR-aware/r16とENTRY/r64のkernel平均RZは5,492.890625対10,187.875だが、shotsは16,142対16,082、Wは16,932,304対16,741,456で、PR側がshots/Wでも一貫して小さいわけではない。shell費用56,497/56,287 RZを含めると、保存総RZは前節の93.007×10^9対170.560×10^9となる。rの異なるm12のattenuationも.727270/.751770である。

したがって、λの低下、shotsの減少、kernelの単価低下は同義ではない。本結果で直接示しているのは候補固有の保存会計項とその総量であり、λ、行内相殺、U回転、PF非可換性の各因果寄与を独立に切り分けた結果ではない。複数の量を同一軸で無次元化したり、独自の寄与率を新たに推定したりしていない。

## 7.5 Comparison with deterministic PF

PR-awareの選定候補と同じJ/U、同じ候補groundと精度政策を用いる全frame ordered S2/s1を強い決定論対照Dとして評価した。C0・64更新ではDの総RZは104,383,256,064であり、共通r32のPR/D保存点比は.897911、公平retuning後のr16では.891018となった。後者は約10.90%低い点推定で、保存された個別nominal95%比区間は[.874508,.907528]だった（Fig.5）。

P1/P2の64更新retuned PRでも、それぞれ自身のJ/Uに対応するDを分母とした保存点比は.735253/.778293だった。これは候補別の副比較であり、全候補に共通のD価格を流用した結果でも、全セル同時優位の推論でもない。

このDは固定された強い対照だが、すべてのTrotter刻み、分割、順序、compiler、決定論PF政策の最適解ではない。過去の別candidateでのupdate005/r128はPR/D=1.97018542062と不利であり、同update005/r32の独立640件は符号・精度未解決、別update064/r32の独立256件はnominalな支持であった。これらを今回の結果とpoolせず、候補・r・cohortに依存する既存観測として残す（Supplement S3）。

## 7.6 Numerical qualifications and limitations

全36 checkpointでDF物理予算を満たしたが、U一次基準は未達であり、J/U同時一次収束・大域最適性は示していない。適格policyで実行したnative4,288標本については、raw、compiled、実際に構築した両逆回路の全512入力/出力とcomplex X/Yの既存検査を維持した。しかし、有限標本のcorrectness確認は未観測の全law誤差上界を与えない。

全law synthesisとimplemented-law信号誤差の資格がないevidence laneでは、shots/totalとactual_CA_achievedをnullのまま保持した。本文の費用値は理想law等に条件付けたモデル値であり、量子hardwareで化学精度を達成した実測結果ではない。実際の量子shot outcomeは生成していない。近似統計区間の被覆率、希少tail、費用support、状態準備込みend-to-end量子資源も未確立である。

H4・1.15 Åでは初期復元時に整合性検査が失敗し、移送後の最適化・下流評価を完了していない。追加作業はユーザーの判断で終了した。この実行上の停止を、PR-awareの科学的な移送成功または移送失敗と解釈しない。今回のResultsはH4・1.00 Åの有限検証へ限定され、他geometry・分子・basis・rank・分割へ一般化しない。
