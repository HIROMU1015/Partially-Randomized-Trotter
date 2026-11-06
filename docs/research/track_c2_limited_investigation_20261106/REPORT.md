# Track C2：新研究候補の限定調査

2026-11-06 JST、SERVER 2 `GPUA100-gleap02`。
基準commit：`720416c5b88b227cf76ff7aa729d91a0a98ed58b`。
専用branch：`codex/track-c2-limited-investigation-20261106`。

**3候補とも、候補名が示す一般的な着想には既存研究との重複がある。残るのは、既知法と同じ情報・最適化機会を与えた比較で、共通の元H energy、信号半径、normalization、全roundのshot×full回路価格を両立できるかという未検証の問いである。今回、その成立や新規性は判定していない。**

詳細は [CANDIDATE_COMPARISON.md](CANDIDATE_COMPARISON.md)。本書は静的な判断材料であり、新方式の採択、実行計画の固定、旧研究の再開許可ではない。

## 1. 調査範囲と根拠

実施したのは既存文書・SOURCEの静的読取、関連する一次文献の限定照合、2文書の作成・公開照合である。保存値は原報告を引用し、新しい集計、再ranking、改善率計算は行っていない。SOURCEのimport、runner/testの起動、科学計算、simulation、sampling、回路build/compile、T合成、アルゴリズム実装は行っていない。

適用AGENTS、`PROJECT_MAP.md`、`研究概要・現状.md`全文と後続記録を確認した。共通指示はworktrees親directoryの `research_project_instructions_unified_2026-10-25.md`、特に役割分担・実行許可・資源利用・再利用/適正管理の規定を適用した。サブエージェントは使用していない。共有設定・パッケージ・他ユーザーのファイルやプロセスには変更していない。

| 固定資料 | 今回の利用・確認深度 |
|---|---|
| [3環境統合 / 3571477][integration] | REPORT、gap/reuseの関連profile、台帳のPR-3/P-D/SP/Parent/UWC/BF等の関連記録。全143資産群・全原runの再監査ではない |
| [文献調査 / e8b19ff][survey] | REPORT、PRIOR_ART_COMPARISON、RESEARCH_OPPORTUNITIESのO3/O4、SOURCESの関連文献。以前の確認深度を継承し、全文読了へ格上げしない |
| [Track Cプレフライト / 4b625e5][preflight] | REPORTとREUSE_AND_BASELINESのPF/RTE、phase、SP/ALLOCATION、RPE会計・強対照。ここで整理済みの接口を再利用 |
| [Track C最終A/B / 720416c][latest] | REPORT/A_REPORT/B_REPORT、関連SOURCE。限定価格の一致と条件付き保証、未資格領域を維持 |
| [旧H4 pilot][old-pilot] | C-S/C-Pの負材料とopt2停止を引用。未完了価格セルは取得しない |

前回文献調査の **O3＝bias・attenuation・合成、O4＝異round構成のenergy整合** が今回の直接の出発点である。C2-A/B/Cを、これらとは独立に発見された方向とは扱わない。

## 2. 引き継ぐ目的と証拠の境界

目的は元の物理Hamiltonian Hの厳密基底状態を入力として受け取り、状態準備費を除いたPR＋RPE全体の総回路費用を下げること。厳密状態の供給仮定は、真のE₀、gap、最適bias補正、制御付き状態準備oracleを無料で与える仮定ではない。表現を変更する場合も元state/sectorとの対応を保ち、追加の基底変換・processor等の物理gateは課金する。古典的な情報・係数・価格の取得費は回路指標とは別に記録し、候補だけに無料の情報を与えない。

共通の目的量は既存定義の

\[
G_k=\sum_{m,b}N_{m,b}\,\mathbb E[C^{\mathrm{no\text{-}prep}}_{m,b,k}],
\qquad k\in\{\mathrm{RZ,CX,T,depth,size}\}.
\]

各指標は分ける。RZをTと同一視せず、shotを掛けたdepthも実機の壁時計時間とは呼ばない。full回路のdepthは部品depthの単純和から保証されない。別profile間の値で勝者を決めない。

最新Track Cの参照条件はLinear H4、1.00 Å、STO-3G、DF12、Partial S2 LD3/Full Y4 LD12、h=.02、元H ground、18round、δ₀=.02、energy精度1.5936×10⁻⁴ Ha、失敗確率.05、位相予算.4 radである。**これは過去結果を読むための条件で、C2実験の新しい採択ではない。** C2で変更する対象・主資源指標・誤差割当は未決定。

| 保存結果 | 保持する内容と適用限界 |
|---|---|
| C-N、参照compiler A | 最新BのS2全2,304 paired標本でRZ/CX/depth/sizeのfull/強baseline価格が一致。Fullはtail-free全体を強baselineに与える。観測cohortの負結果であり、任意のcompiler/PFの不可能性ではない。q8から長qへの価格資格と全RPEの勝者はUNDETERMINED |
| 保証A | 固定保存Hermitian行列・理想gate・独立bitsモデルで全18roundの十分条件成立。十分shotはS2 8,598、Full 6,914。元H表現の無条件包含は未資格。価格Bのprimaryは別途固定した8,598/6,912であり、混ぜない |
| C-S | 固定pi/4 coarse・最小残差r・canonical lawの保守会計で巨大normalization/shot負担。安いprimitiveから全費用改善は導けない。全ての非一様な決定論合成精度配分を否定した結果ではない |
| C-P | 指定whole-kernel共役で、丸めを超える信号改善とshot減少は観測されなかった。共役はkernelの固有値を変えない。別の非共役PF設計全般を否定する結果ではない |
| PR-3 | 固定2qubit toyのtail外挿はbias低減よりvariance/backbone費が支配しSTOP。bias低減だけを改善と数えない |
| P-D、BF/BM | P-Dは強い同情報B1b/B2/B4との選択差を支持しなかった。BF元runのINCONCLUSIVEと保存復旧の限定負結果を区別する。BMは既知compact/BCH対応を保持。名称変更や同じscore/gridの延長では解除しない |
| UWC・SP・Parent | 旧UWC raw発見はFOUND_UNVERIFIEDのまま。SPのprimitive trade-offとwrapperのshot負担を分ける。Parent/S4の訂正KKT・common-angleのL1最小性を保持。旧proxy目的や同fixture追加探索へ戻さない |

上記の数値と状態は[最新A/B][latest]、[旧pilot][old-pilot]、[統合台帳][ledger]の記述を引用したもので、本調査の再解析ではない。

## 3. 候補ごとの主要な判断材料

| 候補 | 既に実現・議論されている部分 | 今回残す未検証の問い（仮説） |
|---|---|---|
| C2-A：round依存PF | 既存round別r/K・β/α探索、UWCのround別split/表現最適化、精度に応じたHamiltonian変更という先行案 | PFの次数・step幅・構成そのものを変えた場合にも、全signalが同じ元H energy/alias分枝を示し、最適化済み固定PFを含む強対照よりGが小さくなる条件があるか |
| C2-B：非一様合成精度 | 合成誤差配分と資源最適化は既知。SPのplacement/QPD、Parentのparent-child係数・proposal配分は近接するが、各回転の決定論的合成誤差の選択とは同一の変数ではない | controlled full回路の実際の反復・相殺・誤差感度と離散T価格を用い、既知の最適化済み非一様配分を超える利益が残るか。現時点では具体的な追加機構を確定していない |
| C2-C：基底energy向けPF | 固有値誤差に基づくpartition/step選択、D6/支配位相診断、高次PF、P-D/BF/BMは既存 | 単なる目的関数の変更を超え、位相biasと元H groundの信号半径を同時に保つ非共役PF構成が、finite-RTE・合成・full価格を戻しても有利となる条件があるか |

いずれも一般的な候補名の段階では追加の方法差を確認できない。「未検証」は「世界的に未研究」「新規性あり」の意味ではない。具体的な係数列・optimizer・新理論は今回設計していない。各仮説が既知法の応用接続だけに還元される可能性も残す。

### 限定的な一次文献照合

固定文献調査を優先し、以下だけを関連箇所で照合した。引用は外部文献の事実であり、本SOURCEでの保証や性能ではない。

| 一次資料 | 確認した箇所・本調査への意味 |
|---|---|
| [Kanasugiほか、UWC、arXiv:2603.22778v2](https://arxiv.org/html/2603.22778v2) | §II.2.4 Eqs.17–18はround別splitと全費用、§III.3 Eq.34周辺はround別表現最適化と古典負担を明記。最終round支配の近似も含む。PF次数切替の本課題全体が既に解決済み、という意味ではない |
| [Häner・Roetteler・Svore、Managing Approximation Errors、arXiv:1807.02336v1](https://arxiv.org/html/1807.02336v1) | 今回の限定追加。§I–II、Lemma II.1/Theorem II.2に、誤差の合成と頻度・費用を伴うsubroutine別配分がある。非一様な合成誤差配分一般を新機構としない。controlled PR mean/RPE成功の資格は別 |
| [Mehendaleほか、Lower Eigenvalue Errors、arXiv:2312.13282v3](https://arxiv.org/html/2312.13282v3) | 要旨/§IとApp.A.7。固有値摂動に基づくPF選択、Trotter・PE・合成の誤差配分を扱う。v3の二次PFと初期版の記述を混同しない。実RPE半径・本native full価格の保証ではない |
| [Kivlichan・Granade・Wiebe、Phase estimation with randomized Hamiltonians、arXiv:1907.10070v1](https://arxiv.org/abs/1907.10070v1) | 要旨のみ。位相推定精度に応じたHamiltonian精度の変更とground情報の利用は既知。gap/variance条件の定理を本系へ移したとはしない |

PR原著App.Eの合成残差とround会計、SPRINTの位相/weight/減衰、既知PAI/TETRIS、高次PFは[固定調査][survey]と[プレフライト][preflight]の対応を継承する。原著全件の再読や網羅的検索はしていない。RSC出版社HTML直取得は403だったため、固有値誤差の今回の本文根拠は取得できたarXiv v3に置いた。出版社最終版との全差分は未照合。

## 4. 残る研究機会と共通の落とし穴

最も直接に残るのは、**既知の最適化の後にも残る、共通energyと費用を結ぶ性質の特定**である。Aは異構成間のenergy整合、Bは合成誤差・離散費用とNの結合、Cは固有位相改善と半径維持の両立を切り分けて問える。これは仮説の整理であり、3方向の優先順位や採択判断ではない。

短roundのPASS、摂動係数の低下、T/回転の減少のいずれも、それだけでは全RPE改善を示さない。異なるPFで各roundのeffective energyを別々に追うだけでは共通energyを保証できない。channel誤差だけではcontrolled相対位相を保証できず、scalar補正で位相が保たれてもrange/shot負担は残る。

追加候補を独立に立てるほどの新しい機構は今回特定していない。精度を保持したcompilerの共通最適化は、C-Nの保存結果から直接残る通常の実装・対照改善である。微小角を落とす旧opt2を復活させず、候補とbaselineの双方に与える比較基盤として扱う。これ自体を新研究の新規性とはしない。

## 5. 再利用、並列性、次の判断

既存のPF registry/decomposition、signed finite-RTE、DF phase保持builder、D6/支配位相診断、Parent/SPのmoment会計、参照compiler A、18round error/alias会計、paired full/strong価格providerを利用できる。具体的なSOURCE版・関数・testの対応と適用限界は[比較表](CANDIDATE_COMPARISON.md#再利用するsourceと検証資産)に記す。旧runnerは再起動せず、将来の変更条件に適合する接口だけを使用する。

A/B/Cは共通の固定入力と成功条件を用い、他の二機構を固定すれば独立に検証可能。ただしA/Cで角度・反復構成を変えればBの精度key/費用・compiler誤差inventoryも変わる。結果を見た後で候補を混ぜたものは独立検証ではない。将来の計算プロセスの並列実行と、禁止されているAIサブエージェント委譲は区別する。

次の担当は**GPT／ユーザー**。①既知法の応用接続を成果とするか追加の科学的性質を問うか、②具体的に残す問い、③強対照に共有する情報・最適化機会、④主資源指標と成功保証の範囲を判断する。必要な最小検証と反証時のSTOP材料は比較表の提案であり、固定した実行契約ではない。

**Codex**は、選択された範囲の設計・実装・数値再解析・科学計算・build/compileについて別途明示許可を受けた場合にのみ着手する。重要な科学レビューの許可と科学実行許可は別である。本調査の完成に追加の科学計算は不要。

## 6. 文書検査・保全・終了

変更対象は本directoryの2 Markdownのみ。新しい科学結果・正式採択・既存evidence statusの変更がないため、過去の研究概要・validation manifest・STOPは変更しない。本REPORTを入口とする。固定Git参照/相対リンク、UTF-8、diffの範囲と空白、公開後のremote SHA・2ファイルの取得内容を照合して終了する。これらは文書検査であり、科学的再検証ではない。

旧SOURCE・入力・raw・result・grant・ledger・marker・STOP、旧run01、C-N/C-S/C-P、PR-3/P-D/BF等は保全する。新しい承認証明物・管理基盤は作成していない。

**mandatory STOP：限定調査と2成果物の公開照合が完了した時点で停止。自動的に開始する後続作業なし。** 公開SHAと照合結果は最終チャットに報告する。

[integration]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[ledger]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv
[survey]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[preflight]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/track_c_o3_technical_preflight_20261106/REUSE_AND_BASELINES.md
[latest]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/REPORT.md
[old-pilot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_h4_pilots_20261106/REPORT.md
