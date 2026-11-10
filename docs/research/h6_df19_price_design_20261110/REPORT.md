# H6 DF19 RPE：compiled価格の再利用調査・モデル検証設計

2026-11-10 JST。**設計・読取調査のみ。新しいH6価格予測、残差、fit、sampling、回路build/compile、signal/N、N×Cは0。** [SOURCE索引](SOURCE_INDEX.md)、[保存metadata coverage](COVERAGE.json)。数値は固定報告・保存欄の引用であり、今回の再推定ではない。

現時点では**m3/4の直接価格をそのまま再利用できるが、全13roundの価格はまだ資格できない**。最初の推奨は、H6実builder由来の構造ruleを固定し、保存PF点とPR探索9/独立確認33を用いて絶対RZを照合するV1。新compile/fitを行わず、11 signed pieceと境界の移送が成立するかを先に判別する。未知の長q価格を0や外国係数で埋めない。

## 1. 固定条件とRPE coverage

Linear H6/1.00Å/STO-3G、tol-only1e−8の保存DF19、weighted-Hermitian completion、signed weights/fragment順/scalar、12 system mode・Nα=Nβ=3のsector400を維持する。targetは保存DF-H同sectorのground energy、保存数値state条件付き、P=0。εE=.00159360010199040 Ha、各方式αtotal=.05、βtotal/stat=.4/.2、h=.1、m0…12。厳密ground/actual実装/lawの未資格は価格接続で解消しない。

下表は[D2固定版][D2]のCOVERAGEと[D1][D1]から引き継いだ状態。Nは各軸、再計算なし。qRPE=2^mとqPF（T内の分割数）は別変数である。

| m / T | PF Full19 / PRのm5構成 | 保存N PF / PR | 同candidate直接C |
|---|---|---:|---|
| 0 / .1 | qPF1/δ.1、PR p5/R32/K2 | 705 / 705 | 未取得 |
| 1 / .2 | qPF1/δ.2、PR p5/R32/K2 | 705 / 708 | 未取得 |
| 2 / .4 | qPF1/δ.4、PR p5/R32/K2 | 705 / 719 | 未取得 |
| 3 / .8 | qPF1/δ.8、PR p5/R32/K2 | 705 / 765 | PF直接、PR探索9/確認33 |
| 4 / 1.6 | qPF1/δ1.6、PR p7/R16/K2 | 705 / 721 | PF直接、PR探索9/確認33 |
| 5 / 3.2 | qPF2/δ1.6、PR p7/R16/K2 | 705 / 737 | 未取得 |
| 6 / 6.4 | **計画候補のみ**qPF4/δ1.6、Full19/PR p7/R16/K2 | 別担当、ここでは未判定 | 未取得 |
| 7…12 / 12.8…409.6 | 未採択。m6の構成を自動延長しない | 未評価 | 未取得/適格modelなし |

m0…5のcandidate ID・N origin・SOURCEはCOVERAGE.jsonへ保存欄を転記した。m5はPF `3f813a46e17461bfe484`／PR `72255c43facdfb8b4bee`。m6を評価する担当のSOURCEや条件を変更しない。このcoverageは指定された固定資料内の範囲で、全保存先の不存在証明ではない。

## 2. 直接価格・cohort・cache

[主研究155529e2][MAIN]の6価格表を、Full19/m5、PR p5/R32/K2/m5、PR p7/R16/K2/m5に限定して照合した。

| 固定回路族 | 同族の保存直接価格 T / qPF / δ | 利用と制約 |
|---|---|---|
| Full19/m5 | .8/1/.8、1.2/1/1.2、1.6/1/1.6、2/1/2 | 角度が異なるq1点。q傾きの校正点ではない |
| Full19/m5、旧再利用点 | .8/2/.4、.8/4/.2、.8/8/.1 | 反復境界/角度の構造rule検査候補。同δのq掃引ではない |
| PR p5/R32/K2/m5 | .8/1/.8と1.2/1/1.2、両点とも探索9・確認33 | 同q別δ。lawが変わるため単一q-Affineへ入れない |
| PR p7/R16/K2/m5 | 1.6/1/1.6は探索9・確認33、2/1/2は探索9 | q2/δ1.6のm5直接値はない。q4のm6も未取得 |

RZ主単価の保存例（X/Y同値）：m3 PF427996、PR探索141775.11111111112/確認141173.21212121213、m4 PF428918、PR探索179583.33333333334/確認179579.36363636365。これらを新しい平均・差・比へ加工していない。探索/確認のpool、aliasの二重標本化、他prefix価格の置換をしない。

同じn本でX/Yをpaired取得しているため、標本数を倍にしない。RZ/CXはgate count、sizeはinstruction count、各depthはcircuit layer。N×depthは将来の累積layer費用で実機wall timeではない。peak qubitsはmax幅の別制約、保存coverageが不足する場合はnullとし、Nやround数を掛けない。

元runtimeはPython3.12.3/NumPy1.26.4/Qiskit1.3.0/SciPy1.14.1、compilerはrz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、symmetric_directional、測定付き完成Hadamard wrapper。Track D P0の追加設定やDF12 opt2/Track C compiler Aを同一と見なさない。保存profileはreference-onlyである。

cacheの読取確認では、元compiled_cache directoryの存在と、PR m3/m4の探索9＋確認33それぞれのresult/events_PRIVATE原本の存在を確認した。PR trial resultにはprices/preparation_hash等があるがwrapper fingerprintはなく、event記録もevents/seeds/cohortである。**これだけでは各trial→cache entryの結合は確定しない。** 原本eventsは内容の再計数やQPY decodeをしていない。

PF m3の旧受入receiptにはX/Yのfingerprint・compiled_SHAが残るが、その2 fingerprintのJSONは長時間studyのcacheにはなかった。これは旧保存価格の無効化や全cacheからの消失を意味しない。旧価格は元receiptを優先し、価格確認のために再compileしない。任意新candidateのcache hit、全entry/QPYの整合性確認は未実施。

`compile_cached`はmiss時にlock/file作成と実compileへ進むため、今回呼んでいない。将来は保存canonical fingerprintとcompiler/scopeの対応が確定した既存entryだけを読取選択する。新T/qや角度変更を同じfingerprintと仮定せず、cacheが不明なだけで新compileを要求しない。

## 3. H6 m5の構造descriptor

m5のsigned pieceは保存順に

`[.12547397927357543, .13781643533753404, .1211920365713068, .14181961643121604, .11225428783206336, −.27711271089139133, .11225428783206336, .14181961643121604, .1211920365713068, .13781643533753404, .12547397927357543]`。

これは保存D2のsigned_weightsの転記。各outer内でこの11 pieceを順に、時間w_j T/qPFで適用し、そのouterをqPF回反復する。中央負時間を絶対値へ置換しない。PRのR_outerは11piece全体で分配し、各piece最低1を維持するABS_TIME_PER_OUTERである。保存配分はR32=`[3,3,3,3,2,5,2,3,2,3,3]`、R16=`[1,2,1,2,1,2,1,2,1,2,1]`。Rを各pieceへ重ねて掛けない。今回新しい配分は計算していない。

| 部分 | 実SOURCEから引き継ぐ構造・注意 |
|---|---|
| Full19 PF | one-body＋19 DF block。各pieceのS2を元deterministic_control_planで構成、basis逆変換→signed diagonal→basis正変換。新しい分割やDF順序を導入しない |
| PR prefix5/7 | 各pieceでprefix側のUNCONTROLLED半step、保存basis planに沿うcontrolled tail sequence、逆順DIRECTIONAL半step。tailはdiag(I,U_event_sequence)の元意味を維持 |
| tail event | K2のpaired polynomialと元λ_R/τ、選択product/application順・basis ID/hash/再利用区間・rotation/phaseを保持。1 eventを固定gate価格と仮定しない。負時間・signed λを保つ |
| scalar/identity | piece内builderへのscalarは0。最後にPF c、PR c＋extracted_identityのP(−scalar T)を一回追加。Bは信号のnormalizationで価格乗数ではない |
| directional diagonal | 非零RZZ、RZ、global phase由来ancilla RZを元規則どおり扱う。zero判定や数値πを勝手に近似同値へしない |
| wrapper | control＋12 system mode、H/evolution/(YのみSdg)/H/測定。P0−P1でX=Re(zraw)、Y=Im(zraw)。logical幅は数えるが費用の加算量とはしない |
| 境界 | head/tail、signed piece間、outer step間、scalar/wrapper端部を分離。basis共有、隣接CX/1Qの融合、全体compilerの非局所影響を調べる。S2のq−1界面規則を11pieceへそのまま掛けない |

Full PFのq1価格でもδで単価が変わる保存例があるため、角度非依存と決めつけない。PRはm3→m4でprefix5→7、R32→16、δも変更され、別回路族である。

## 4. モデル候補・再利用技術

| 案 | H6での設計 | 現在の証拠と不足 |
|---|---|---|
| A 構造会計 | primitive/basis/tail、piece/step境界、端部/scalar/wrapperのsource由来reward/transfer。nativeとpostcompilerを別出力 | native counter/max-plusは確定native列の整数会計に再利用可。H6からの新計数/予測は別途科学許可。compiler後の閉包は未証明 |
| B Affine | **同δ・prefix・R/K/law/compiler**の方式/軸別絶対C(qPF)=a+bqPF | 目標PR族は主にq1のみ。δ1.6のq1/2/4列はq1直接だけで傾き未識別。全13roundを一式でfit不可 |
| C 構造＋少数補正 | Aが説明できないH6の局所界面だけに補正。未説明特徴の必要最小自由度を固定 | qのみのbulkとq−1境界を自由係数化すれば共線。探索側で識別できる範囲だけ、留保で固定検証。まずfitなしAを試す |

[H4設計][H4]からはsource先行のprimitive/角度/界面分解、開発/留保分離、絶対価格・paired X/Y、cache miss非呼出しを再利用する。H4 V1の結果待ちを必須にせず、H4担当枝へ書き込まない。H4 S2/DF10の係数・界面ruleをH6 m5へ無条件転用しない。

[既存価格調査][SURVEY]と[同索引][SURVEYINDEX]を再利用する。D C6a〜C8のwhole-path、q別共分散/独立holdout、basis-run/モデル安定性、Cのshared-cell共分散、新Aのnative DAG/角度分類/max-plus、DF12 V3/R1のdirect/model/外挿表は方法・適合する小関数の再利用単位である。CDF8/Strong-Reference比較をH6 PF-PR比較へ読み替えず、数値係数・改善率・Nは移植しない。D q64の後続完了を既存結果として継承し、同じ検証を再実行しない。

max-plusはdepth用で、RZ加法性やE[max]=max(E)を与えない。basis-runを使う場合は未来の特徴lawが必要。postcompiled DAGから逆算した特徴を未compile入力の説明変数としない。構造モデルの一致が崩れたときは、まずnative→compilerの未閉包界面を特定し、大量compileへ自動展開しない。

## 5. 最初のモデル検証 V1（提案・未許可・未実行）

**新compile/fit/samplingなしで、保存価格に対するsource由来構造予測を初めて照合する。** これは実データの新予測・残差計算なので別途明示許可が必要である。

1. provider c31ef424と元native/m5 schedule、保存H6 preparation/eventsを入力に、primitive/境界descriptorと予測ruleを固定する。全QuantumCircuitのbuildはしない。必要なprepared basis原本やevent情報が不足すれば再生成せず不足STOP。実データ計数/予測自体は今回0。
2. 開発用はFull19 m3/q1/T.8、PR m3/p5/R32探索9、および別族PR m4/p7/R16探索9。p5/R32とp7/R16で校正を混ぜず、同じsource規則が両族を説明できるかを見る。自由なq係数はfitしない。
3. rule固定後の留保はFull19 m4/q1/T1.6、PR各族の独立確認33本。探索9/確認33は同q別trajectoryの検証で、新qの検証ではない。PFの旧q2/T.8/δ.4点を反復境界の補助ストレス検査へ留保するが、δ変更を含み、固定δ外挿の検証とはしない。既知の公開集計値があるため完全盲検とは呼ばない。確認を見てruleを直せば、そのcohortは開発へ移る。
4. PF/PR別・X/Y別の絶対RZ、native/postcompiler区別、同pathの符号付き整数差と最大path差、未説明basis/piece/step/端部を記録する。PRはpaired X/Yを一vectorとして扱う。主目的は局所閉包と有限保存cohortへの適合で、母平均精度や全q保証ではない。CX/depth/sizeはdescriptorの適合確認に留め、追加数値指標を行うならscopeへ明記する。
5. compiled整数countの厳密移送を主張するには全留保wrapper差0が必要。差があればexact採用を止め、構造rule/角度/非局所compilerの不足を報告する。fit/新sampling/compile/閾値緩和を自動追加しない。公開照合後STOP。

このV1で利用するPRは2族それぞれ探索9/確認33の既存全path。nをNに読み替えず、同pathの軸を独立標本にしない。古いPF receiptの比較元を確認できなければ、その補助点だけ未資格にし、新compileで穴埋めしない。

## 6. 不足する場合だけの最小追加scope

**V1とは別の未許可案。** 構造が閉じて未compile領域の期待rewardを正当化できれば、以下のcompile自体を省略できる。

- 固定δ1.6のFull19/PR p7/R16/K2/m5について、既存q1/T1.6を再利用し、q2/T3.2（m5候補）を追加校正、q4/T6.4（現在のm6計画候補）を独立q留保とする案。m6のsignal科学資格は別担当の判定を引き継ぎ、ここで採択/変更しない。
- 同candidateの保存cache/原本が後から見つかれば優先再利用。なければPF各1回路・各X/Y、PR新独立cohort各8 whole-path・各X/Y、**最大36 wrapper compile**の構造反証pilotを選択肢とする。n8は高精度母平均保証の標本数ではない。これは科学対象数でCPU/worker上限ではない。
- 新scopeでは元H/state/preparation/compiler/符号/精度と正しさ確認を固定し、価格標本数・seed/cohortと校正/留保を取得前に定める。SOURCEと出力を分離して公開後STOP。取得後top-up、q8以降、m0…2の直接compile、signal/N/Gは自動追加しない。
- m0…2はq1でδ/lawの異なる端部domain。V1のsource/角度/law閉包で判断できない場合にだけ、例えば最初のT.1/PR p5/R32とFull19を一つの反証点として別途提案する。上の固定δpilotで短時間域まで保証しない。

同δで二点校正・一点留保が揃ってもqPF>4の外挿systematic、rare tail、全law期待Cは未資格。m7…12は候補未採択なので必要qの上限すら現段階で固定しない。H4のq64案や他系列の長q寄与率を機械的に採用しない。

## 7. 不確かさと全13roundへの接続

保存有限cohortの単価はE_law[C]の保証と異なる。保存paired統計を優先し、同path X/Y、共通校正係数のround間依存、直接anchorと係数の交差共分散を保持する。未観測eventは確率×条件付き費用で別に扱い、未取得tail boundや母分散を0にしない。bootstrap/正規近似だけで長q model biasやfinite-bit law差は閉じない。価格用の信頼配分とRPE測定αtotal=.05も別物である。

N/Cの結合keyは`H/state/snapshot/completion/rank_policy/fragment順/preparation, candidate_id, formulaとsigned-order, T/qPF/δ/prefix/R_outer/allocation/K/law, compiler/control/full-wrapper/axis, P=0, SOURCE/cohort`。H6の歴史的binding.T=.8は入力reference時刻で、実発展時間はcandidate.Tとm/hで照合する。時刻欄を一括上書きしない。保存Nを再計算せず、direct C/モデルĈ/未取得、校正/留保/外挿、統計変動/model bias/不明covarianceを別欄で渡す。

将来G=ΣNC、固定NのΔG=ΣN(C−Ĉ)、Var(Ĝ)=wᵀΣwを扱う。共有依存を保持し、不明なΣを独立0にしない。DF10やD/DF12の残差をH6の誤差幅へ転用しない。短q最大残差は未検証長qの一様boundではない。N未取得roundを仮定値で埋めず、価格不確かさで順位が変わり得るならPF/PR優位は未確定。今回新G/寄与率/ΔG/順位は計算していない。

## 8. 結論・実施範囲・STOP

**新compileを避けて今すぐ使えるのはm3/4の保存直接価格と、同族の限定補助点。全13roundの期待compiled価格を保証できる既存モデルはない。最初に行うべきことは、H6 m5実SOURCEから定めた構造ruleを保存PF・2族のPR探索/独立確認へ照合するV1である。** その結果で足りない界面だけ補正を検討し、なお固定δのq証拠が必要ならq2/q4の小さい反証pilotを別途選ぶ。全round直接compileやH4完了待ちは既定にしない。

既存H6 D2 branch/worktreeとH4価格branchを読取確認したが、並行m6とH4 V1の競合回避のため、固定D2 carrier95d01b6aを親に`codex/resource-eval-h6-df19-price-design`を新設した。両担当worktreeへ書込み/競合push/制約を加えていない。branch統合・削除・履歴変更なし。

変更は本報告・索引・metadata coverageと必要な入口/証拠状態記録だけ。研究module import、数値モデル/feature計数、QPY decode、新価格/残差/fit/bootstrap/sampling/実build/compile/signal/N/N×C/全13round費用/詳細科学レビューは全て0。文書のみなので既存科学/人工testの重複実行や新価格コードを追加しない。保存原本の限定存在/metadata確認、リンク/selector/文書整合性を技術確認する。

旧SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPを保全する。公開remote SHA/tree・主要成果物一致の確認後**mandatory STOP**。次は**ユーザー**がV1の実データ構造予測・残差照合のscopeを判断し、**Codex**は明示許可後だけ実行する。

[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/95d01b6ab1d820921fcd2521324e41e82389713a/docs/research/h6_df19_rpe_d2_results_20261110/REPORT.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/45607a5086721ea1d21b1ed2d7edfe870549f0c8/docs/research/h6_df19_rpe_d1_results_20261110/REPORT.md
[MAIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[H4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10132af4a67ce112f79c539a9552f5ca0e83e516/docs/research/h4_df10_longtime_price_design_20261110/REPORT.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/REPORT.md
[SURVEYINDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/SOURCE_INDEX.md
