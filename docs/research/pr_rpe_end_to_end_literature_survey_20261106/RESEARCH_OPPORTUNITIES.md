# 研究可能性の残る問いと引継ぎ

2026-11-06 JST。固定基点：`3571477093441ac4852166fe829f8ccc5db5ec4f`。下記O1–O6は整理用の番号で、優先順位・正式候補採択・GO/NO-GO・論文化可能性の判定ではない。具体的なalgorithm仕様、実行契約、証明、新実装は作っていない。

元Hの厳密基底状態、状態準備除外、同じenergy精度・成功確率という目的を全候補とbaselineで共有する。新しい固定数値は未決定。文献ID・固定証拠の入口は [SOURCES.md](SOURCES.md)、費用の因果経路は [END_TO_END_COST_ANALYSIS_MAP.md](END_TO_END_COST_ANALYSIS_MAP.md)。既存STOPを別candidate名で解除しない。

## O1 表現の変更がtail・位相・回路価格を同時に改善する条件

1. **改善機構**：係数massだけでなく、deterministic backbone、実装可能辞書、PFの位相/半径を併せて変える表現の性質を問う。重いtailを少数の安い演算へ移せればN×価格を下げ得るが、成立は未確認。
2. **総費用への接続**：tail→normalization/shot、fragment/basis→full price、fit/PF bias→step幅とround horizonの各経路がある。元H energyへのfit誤差を支払い、stateの対応を保持する。
3. **既知法との差**：UWC L03は表現＋split＋RPE費用を既に結ぶ。RC-DF/BLISS、L07のcheap lexicon、原PRの残差Pauli化も既知。これらを単にPRへ足すだけなら応用実装である。独立した追加の性質はまだ提示されていない。
4. **既存証拠**：RQ-01、MIX-F1/F2、SUP-C-005/064/WORD、Pauli/DF比較を再利用できる。F2のcorrectnessと単回期待費用、研究Cのnominal/固定wordを区別する。研究Cのown-DF groundは今回の元H stateと同じではない。
5. **反証材料**：F2は混在必勝を支持せず、core/pairにも強対照負結果がある。旧UWCはFOUND_UNVERIFIEDで、外部UWC受理を理由に復活できない。近似fitのenergy heuristicと基底変換費が利点を相殺し得る。
6. **未解決の科学的問い**：同情報・同targetでUWC/collected Pauli/DF＋強いPFにも同じ改善機会を与えた後、位相情報・非加法priceを使うことでどの条件に追加利益が残るか。単なるparameter増加ではない性質を特定できるか。
7. **検証可能性・資産**：統合profile `MD-INPUT/CONTROL/STAT`、`DF-CONTROL`、`REP-MECHANISM`、研究Cの固定入力/optimizer/native入口。I-CODEの`normalize_involutory_tail`とtyped basis/phase接口が再利用候補。元版・state・精度への適合は将来別途確認する。
8. **追加作業の区分**：静的確認＝元H/DF/state・候補辞書・保証の差を限定照合。理論＝追加の性質と既知法との差の検討。実装＝採用された場合だけ既存router等の必要差分。科学計算＝同精度全law/全N/全wrapperの検証で、別の実行許可が必要。

## O2 非加法な全path価格を利用できるsamplingアクセスと有限confidence

1. **改善機構**：whole-pathのbasis相殺・control/合成価格とweight統計の関係を使い、実行可能な生成コストで総資源を下げられる条件を問う。新しいIS最適式を作る課題ではない。
2. **総費用への接続**：law変更は期待価格とestimator range/momentを同時に変え、必要N、X/Y配分、round失敗率に効く。normalizer、価格取得、bit精度、online samplerの古典workも記録する。
3. **既知法との差**：L04/L05、whole-path IS、Parent/S4、PRHS/qSHIFTと既知の履歴lawが近い。`p/√c`や`h/C`の再記述だけでは差がない。安いnormalizer・生成アクセス、または有利な適用条件が追加で成立する必要がある。
4. **既存証拠**：RQ-04/05、G6–G10のreturn/finite-bit/native、D1–D3、訂正Parent/S4、SAMPLING-FREEDOMを再利用する。全trial/zero-fill/known returnを区別した既存構成は未実装へ戻さない。
5. **反証材料**：相関marginalだけではcoherent meanは保たれない。L04 II.1は独立・加法条件下で長い積のIS利益が縮むことを述べる。C1はchild費とnative/N不足を残し凍結中。既存NO_DISTINCT_FREEDOMは限定surveyの結果として維持する。
6. **未解決の科学的問い**：非加法price・構造化returnが、必要な古典取得/生成費を払っても同精度全RPEで有利になるクラスはあるか。有限confidence・weight rangeまで含めた性質が既知法へ還元されるかを区別できるか。
7. **検証可能性・資産**：`RA-KERNEL/NATIVE`、`ALLOCATION`、`RTE-BASE`、G6 serializer/provider/return kernel、Parentのmoment/accounting。C1の保存logical pathや過去人工testは既存範囲の参考で、新たな分子sampler実行の証拠ではない。
8. **追加作業の区分**：静的確認＝候補lawの入力情報・全trial分母・実装接口と既知lawの照合。理論＝取得可能性/有限confidenceの追加性質。実装＝採用後の最小sampler差分のみ。科学計算＝sampling/RNG replay/再配分/保存再会計を含み、別許可。旧C1やParentの延長を自動開始しない。

## O3 位相bias・attenuation・native合成を合わせた有利条件

1. **改善機構**：PFのorder/processingや角度合成を変更したとき、位相だけを改善せず、半径・weight負担とbackbone価格まで含めて有利になる構造を問う。一部の高価な合成を残差/測定へ移す機構は既知である。
2. **総費用への接続**：bias予算の余裕はstep/r/K/Nを変え得るが、stage増加・負時間・synthesis error・normalizationが相殺する。合成の効用はprimitive価格でなくfull interrogationのN×価格に現れる必要がある。
3. **既知法との差**：L01 App.Eのrounding-to-residual、SPRINTのprocessing/合成解析、TE-PAI/TETRIS、既知PAI/RUSが近い。層を二つ組み合わせたという名称では差にならない。どの誤差構造や適用条件が追加で成立するかが必要。
4. **既存証拠**：ADD-PF、B-M/compact BCH、研究B、SP/R1、D1–D3、FR、P-Dを継承する。高次PFの実装と保存された条件付き成果はある。SP toyとactual finite-RTEの検証層を分ける。
5. **反証材料**：P-Dは強baselineとの選択差未成立、BMは既知compact対応、PR-3の固定toy外挿はvariance/backboneで停止。正scalar分離だけによる実用認証差も旧FRで得られていない。L02 App.Fには減衰/linewidthの交換がある。
6. **未解決の科学的問い**：原Hの厳密基底状態と取得可能な情報だけで、長roundの位相・半径を両立する有利条件を特定できるか。既知processed/PAIと公平に比較して残る知見は何か。旧FRの未解決問題を本調査で解いたとはしない。
7. **検証可能性・資産**：`PF-ORDER`、`BF-BCH`、`FR`、`SP-R1`、`ALLOCATION`、phase保持builder、finite mean/reference接口、既存native priceの元snapshot。P-A一区間相殺やGaussian singletonを新機構として再開しない。
8. **追加作業の区分**：静的確認＝既知合成のtarget/meanと元契約の適用差。理論＝追加性質と誤差接続の検討。実装＝正式選択後の必要差分。科学計算＝新係数探索、bias評価、native synthesis、compiled価格、再会計は別許可。旧toyの追加探索を今回の必須次段階にしない。

## O4 異なるround構成を同じ元H energyへ接続する性質

1. **改善機構**：時刻ごとの誤差・価格の違いを利用し、表現/split等を変える価値があるかを問う。全roundの構成を増やすだけでは研究機構にならない。
2. **総費用への接続**：安い短roundが初期branchを確定し、高い長roundが精度を決める構造を利用し得る。ただし各signalのeffective phase、半径、alias範囲、N、表現更新費が全体へ影響する。
3. **既知法との差**：UWC L03はround別表現最適化と古典負担を明示し、L11は高powerだけのcontrol置換を議論する。fixed-splitのr/K/β/α最適化もrepositoryにある。独立性を主張するなら共通energy接続や追加の有効条件が必要。
4. **既存証拠**：RQ-06、SUP-RPE-V1/V2/V3R1/ENERGY、M1-S、研究Cを再利用する。条件付き18round成功・energy bridgeを未着手に戻さない。一方、round別表現/splitの新方式は検証済みではない。
5. **反証材料**：L03の最終round支配なら前段だけの削減効果は小さくなり得る。早いroundのbiasでも誤branchを選ぶ可能性がある。既存full RPE modelの点差は不確かさ込みの優越を確定していない。
6. **未解決の科学的問い**：真のE₀/gapを知らずに異構成signalの許容集合を同じenergyへ結び、最適化した固定構成対照より有利な条件を示せるか。既知adaptive/multifidelity運用との差を特定できるか。
7. **検証可能性・資産**：`RPE-COST`、I-CODEのaccounting/provider、後続decoder/enclosure/energy接続の固定SOURCE、保存scheduleとM1-S signal。研究Cのown-DF stateや別rankの結果は適合確認なしに混合しない。
8. **追加作業の区分**：静的確認＝各signalのH/state/phase/decoder契約の対応。理論＝異構成の保証と情報条件。実装＝採択された接続の最小差分のみ。科学計算＝round別最適化・新energy照合・保存値再ranking等は別許可。今回新schedule/契約は作らない。

## O5 reference干渉によるcontrol削減とrandom pathの整合

1. **改善機構**：SE-QPE L11のCSWAP/reference方式で、controlled evolutionの乗算的負担を追加register/干渉の費用へ交換する可能性を問う。既知diagonal-only control後に利益が残る必要がある。
2. **総費用への接続**：回路単価とdepth、追加qubit、reference補正、routing/error detectionが変わる。同じ測定分布を保てる条件が崩れるとNや推定targetも変わる。状態準備除外を追加gate無料へ拡張しない。
3. **既知法との差**：gadget自体と低/高power混在はL11で既知。L20のcompressed controlは別のlocal/translational前提である。PR固有の追加性質があり得るのはpath/mean/referenceの整合条件であり、gadget実装だけは通常の接続課題。
4. **既存証拠**：RQ-03/07のDF/Pauli control、actual adjoint、相対phase、basis共有を再利用する。指定統合資料にはPR pathとSEの共通固有基底・真空reference・Nを結んだ検証を確認できなかった。全repository不存在の主張ではない。
5. **反証材料**：Hamiltonian全体が粒子数を保存しても個々のPauli/RTE pathが真空referenceを保存するとは限らない。独立random halfが共通固有基底を持つことも未確認。CSWAPの定数負担やroutingが短roundの利益を消し得る。
6. **未解決の科学的問い**：同じPR補正平均と元H energyを保ち、無料のtarget-energy情報やcontrolled state-preparation oracleを加えず、reference方式の成立範囲を特定できるか。厳密stateを受け取れることとその制御付き準備/逆演算を持つことは別である。
7. **検証可能性・資産**：`DF-CONTROL/MD-CONTROL/RTE-BASE`、full wrapper/phase reference、既存sampling/word serializer。原著L11のoperator条件とまず静的比較できるが、SE builderが今回既存資産にあるとは主張しない。
8. **追加作業の区分**：静的確認＝number conservation/reference/eigenbasisの対象接口。理論＝成立条件とPR平均への接続。実装＝方式選択後のみ。科学計算＝reference/state生成、simulation、build/compile/routing/price評価は別許可。現時点で新しい回路設計は行っていない。

## O6 signal設計を含む代替推定器とPR oracleの費用境界

1. **改善機構**：QSP-PE L12の既知の感度設計が測定数とquery数を変える機構を、本PRの近似・単価条件で比較できるかという周辺の問い。RPEの改善として採用した方式ではない。
2. **総費用への接続**：query×shotにcontrolled evolutionの実価格、ancilla rotation、近似誤差を接続する。古典signal設計費とpost-processing費も別に記録する。Nだけの比較では不足。
3. **既知法との差**：信号のmax-min設計とunitary oracleでの保証はL12で既知。これを実装するだけでは新法でない。finite-RTE/weighted coherent meanでの保証・総費用が成立する範囲は本調査では未確認。
4. **既存証拠**：既存RPE schedule/accounting、finite-RTE signal、full wrapper priceを比較基盤として再利用できる。QSP-PEの理想query比較はrepositoryのPR総資源検証ではない。
5. **反証材料**：L12のunitary oracle条件とfinite-RTEの非unitary平均は異なる。多重query内のrandomizationを単純な平均置換で扱えるとは確認していない。合成誤差とquery単価が感度の利益を相殺し得る。
6. **未解決の科学的問い**：同じ元H state・情報・energy成功条件で、近似oracleの負担を支払っても優位なsignal classがあるか。RPEのままで研究するか、推定器family変更を許すかはユーザーの判断。
7. **検証可能性・資産**：`RPE-COST`、`RTE-BASE`、旧short-round signal/shot資産。QSP回路や新decoderを今回実装・検証したとはしない。ideal-query baselineの文献対応は追加済み。
8. **追加作業の区分**：静的確認＝oracle access/shift-rescaleと既存接口。理論＝近似oracleに関する追加保証。実装＝推定器familyの別採択後のみ。科学計算＝新signal設計最適化・sampling/回路生成/価格比較は別許可。

## 再提案しない既存経路

P-A interval DP、旧P-C広域予測、P-D/R3の同じscore探索、PR-3の同じ外挿、既知反可換pairの再命名、Gaussian singletonの自動再開、C1/Parentの旧toy延長、旧UWC/失効DF結果の復活を次候補にしていない。上記の問いが関係する場合も元domain・負結果・既知同値・一時凍結を保持する。

O1/O3/O4は旧RQ-01/05/06を中心文献で具体化したもので、新しい7課題を再発行したものではない。O5/O6も既知方式との接続可能性を問う段階である。どれにも現時点の改善率や科学的優先順位は付けない。

## 次の担当・判断事項

ユーザー／GPTが、どの問いを扱う必要があるか、必要なら既知法の応用実装を成果とするか独立した追加性質を問うか、RPE以外の推定器を比較対象に含めるかを判断する。分子、元H/近似Hの区別、精度・失敗確率、主資源指標、compilerは将来の研究設計で決める。

Codexは本6成果物の公開・取得照合までで停止する。追加で依頼された静的照合はその範囲で実施可能だが、重要科学レビュー、非自明な新証明、algorithm設計・実装、新科学計算は現在の許可に含まれない。新科学計算と重要レビューはそれぞれ独立の許可が必要である。自動開始する次の研究作業はない。
