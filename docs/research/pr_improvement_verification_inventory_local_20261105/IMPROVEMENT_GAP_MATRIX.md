# PR改善研究の検証範囲とGap（LOCAL）

2026-11-05 JST。静的な証拠棚卸し。研究方向の順位・正式採択、新しい科学的評価は行わない。
基点mainは `546bca7ed2cece59b714b986a264d7b0b0871a01`。
[台帳](IMPROVEMENT_VERIFICATION_LEDGER.csv)、[重複・再利用](DUPLICATION_AND_REUSE_MAP.md)、[残検証](REMAINING_VALIDATION_MAP.md)、[調査報告](REPORT.md)と併読する。

## 読み方

「検証済み」は各保存結果の限定条件内を指す。課題全体の解決、独立新規性、実機成功、最終費用優位ではない。
理論、人工テスト、数値correctness、実回路compile、資源、独立確認を別軸にした。
台帳の `INHERITED_STUDY_INDEX` は旧70カードへの索引で、今回すべてのSOURCE/rawを読み直した意味ではない。
`STAGE_RECORD` は段階・失敗・修正・終了の履歴、`SUPPORT_REFERENCE` は既存baseline等への関連付けである。
行数は実験数でない。

`UNKNOWN` は同定・取得不足、`NOT_REESTABLISHED_IN_THIS_STATIC_INVENTORY` は今回の独立再認定をしていないことを表す。
いずれも元実験の「未実行」を意味しない。`NOT_PERFORMED_OR_NOT_ESTABLISHED` は未実施と未成立を併せた保守的なセルであり、具体的な違いは `saved_result`、`original_status`、`unvalidated` を読む。

## Phase 2Aの7課題

分類の入口は[固定Phase 2A](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbb83680247a4565336ca177278255e32843cfb2/docs/research/pr_algorithm_improvement_phase2a_20261023/OPEN_RESEARCH_PROBLEMS.md)。番号は優先順位でない。

| 課題 | 既存の構成・検証済み範囲 | 部分検証・限定的な負結果・更新 | 未検証／資料不足 | 台帳の入口 |
|---|---|---|---|---|
| RQ-01 表現のnormを測定込み資源へ変換 | grouping/DF基盤、N1/N2/N3、core/pair、H2四方式、Pauli-JW、混在辞書S/Pを検討。F1修正版でH・law・mean・制御位相、F2修正版で実compiled全件と1-shot期待資源を確認 | 混在が必ず最良ではない。固定F2ではPauli端点が低単価。core/pairは強いwhole-Pauli対照で利益を支持せず。N1は誤差課金後exact fallback。研究Cの候補別nominal支持は別系列 | 分子・未使用入力での独立辞書選択則、同精度N加重利益、生成・前処理費込み。旧UWC原runとのidentityは未解消、旧DF screeningは失効維持 | IDX-REP-01〜09、MIX-2C〜CLOSE、SUP-C-*、SUP-PAULI |
| RQ-02 非prefix分割・grouping | prefix PR-2/M1/M2/PM、P-C追跡、P-D、B-F、B-M、R3、one-body再配分を検討。既知compact BCH／同情報対照との対応を整理済み | P-Dの強いB1b/B2/B4は同候補。B-Mはcompact再帰と同値。B-F旧serialization停止と別保存replay負結果を分離。P-C局所支持の後、広域追跡は固有差不足・予測破綻 | 非prefix自由度そのものが同情報の既知partitionを超える証拠。探索・情報取得費を含む差。現在のprefix勝者を新非prefix方式の成功にしない | IDX-OPT-03〜05、IDX-RB-01/02、IDX-QC-05/09、SUP-TRACKA |
| RQ-03 基底共有・control・融合後の生成改善 | WP06-a/bのfull/support policy、P-A DP、Gaussian積、D1〜D3、混在辞書、rectangular Givens PS48。既存diagonal-only control・basis sharing・actual adjoint・境界融合がbaseline | 一律support限定は悪化例あり。P-Aの非退化比較は一区間baselineとplan・metric差なし。Gaussian積は候補固有手順不足。PS48は同値性PASSだが一律RZ/CX削減不支持。その後singleton独立方向を一時終了 | 未使用contextでの選択性能、既存共有後に残る大きい費用の削減、因果分離。PS48の局所改善から全law期待改善を導かない | IDX-CIR-01/02、IDX-RB-12/13、MIX-CAUSE/PS48/SELECT/CLOSE、SUP-C-WORD |
| RQ-04 sampling law・古典生成・有限confidence | R0/R1、RA-D0/G1〜G6の形式構成、zero-fill／全trial会計、G7/G8の局所/on-demand接続、G9/G10のnative会計、finite-bit/live API、C1、M1-S、相関案を検討 | G9/G10は修正版別実行で完了。P5はRA family内部特殊化。C1は分子logical各方式の完走へ進んだが、zero-pathなし・child回転が残る。相関のmarginal一致だけではcoherent mean保存にならない | 一般Fullの物理provider・全N古典費、強いCTS/Closed P5/既知ISとの同条件全task利益、全law実装精度。C1は一時凍結 | IDX-RB-04〜10/15〜18、TB-G6〜G10、SUP-G6-LIVE、C1-* |
| RQ-05 native辞書・allocation・合成位置 | BS/SP、R1、G9/G10、D1〜D3、Parent allocation/S4、混在辞書の実費を取得。T/CX・shot・準備費の交換を記録 | SPの短model利益は長modelで維持しない。S4の訂正KKT・best-to-best差は保持するが、proposalも変わる。CTSはCXで強い。F2の1-shot価格を認証N加重へ格上げしない | 加法primitive会計から全wrapperへの移送、同等cache/selector/取得費、分子でのstrict native＋共通認証N。T/RZ/CX/秒を単一価格へ勝手に変換しない | IDX-RB-03/04/09/10/13/14/17、TB-G9/G10、MIX-F2 |
| RQ-06 round別表現・splitと同じenergyへの接続 | 固定splitでround別r/K・誤差／失敗確率配分、短4段、全18roundモデル会計、V1/V2/V3/R1の後続を実施済み。M1-Sは単時刻のvisibility・phase/整数witnessまで | V2の条件付き18round成功、11/5の固定H4条件付きenergy接続は「未実施」に戻さない。モデル費用順位の不確かさは未解決。M1-Sの減衰rate比はshot比でない | roundごとに表現/rank/splitを変える新方式の同target・branch整合と節約。固定条件の成功証拠から最終費用優位や実装全体成功を導かない | IDX-OPT-01、IDX-RB-15、SUP-RPE-*、SUP-C-* |
| RQ-07 接続性が選択を変える条件 | topology-freeのDF/Pauli/混在・共有・control検証がbaselineとして利用可能 | basis共有等は確認済みでも、routing graphの検証ではない。C1の分子logical接続もhardware connectivity確認ではない | 同graph/layout/routing予算・同control/precisionでの比較、接続性による選択変更とその独立差。この調査では該当する改善実験の証拠を確認できなかった | IDX-CIR-01、IDX-PR-01、IDX-RB-17、MIX-*、SUP-PAULI |

## RQ-01〜07だけに閉じない追加系列

| 追加の整理区分 | 既に検討・実行された範囲 | 到達点と残る境界 | 入口 |
|---|---|---|---|
| ADD-PF：高次PF、PF誤差、BCH、processed PF | Yoshida/Morales/新4次、YP8m8等、D6・支配位相・response、有限時間校正、B-F/B-M、高次PR／長時間 | 実行窓限定の係数・機構・比較はある。既知同値、負結果、修正入力回収失敗を保持。次数の一様優越・H12係数・新selectorは未確立。別PF repository固有研究は旧カード継承 | IDX-PF-01〜07、IDX-EVAL-03、IDX-RB-01/02、SUP-TRACKA |
| ADD-FR：finite-RTE phase/radius | FR-1、scalar補正、非一様FR-R1b、後続technical note | strict witnessと固定予算の実用認証差なしを両立して記録。T3等の未解決を新計算再開理由にしない | IDX-ERR-01 |
| ADD-C：PR-aware Hamiltonian表現 | 研究CのJ/U更新、own-DF下流、強D、候補005/064の独立価格、固定word correctness | 064のnominal支持と005の未解決は異入力。固定word PASSは全word一様保証でない。旧UWC/raw欠落とは別証拠 | SUP-C-U64/005/064/WORD、旧REP/QC索引 |
| ADD-RPE：統計・normalization・energy | 補正平均と実観測平均、B/attenuation、十分shot、branch/alias、条件付きenergy | 固定TのG、全roundモデル、数学的条件付き成功、実backendの各層を区別。全RPEが未着手という表現は失効 | SUP-RPE-*、IDX-OPT-01/RB-15 |
| ADD-ACCESS：表現外の生成・access候補 | bus/CAR、joint-loop/echo、integrable insertion、集団遷移、adaptive subspace、条件付き測定、unknown-gap、P1〜P8等 | 既知還元・具体生成則不足・固定ansatz不成立・保留を区別。性能比較未実施を「未検討」にしない。別repositoryの現状を今回再審査していない | IDX-EXT-01〜08、IDX-QC-01〜14、IDX-PR-03 |

これらは分類上の見出しで、新しい研究課題・アルゴリズムの正式提案ではない。
一つの研究は複数RQに関連し得るため、表のセル数を検証数へ換算しない。

## 後続更新で特に読み替える点

| 以前の状態 | 後続の固定記録 | 現在の限定した読み方 |
|---|---|---|
| F1 metadata失敗・部分結果 | MIX-F1 | 登録correctness未完了は解消。旧runは失敗のまま。F1はK2/q2でF2/K0/q1と別条件 |
| F2 compiled operator失敗・期待資源未完了 | MIX-F2 | 位相保持修正後の新run全件PASS。旧compiled失敗の証拠は残す。認証N未資格は残る |
| PS48が今後の一律改善候補 | MIX-PS48→SELECT→CLOSE | 科学的同値性と一律費用削減不支持を保持し、独立方向の一時終了を反映 |
| G6ではnative未実装 | TB-G7/G8/G9/G10、後続v1.2 | 限定native取得・会計は実施済み。一般分子全task優位は別問題 |
| G9/G10のtechnical STOP | TB-G9、TB-G10 | 別SOURCE修正版の完了がある。旧prefixの利用制限やSTOPは消さない |
| RAはsyntheticのみ | C1-R1/R2/5A/P5R/PAUSE | H4各1024pathのlogical接続は完了。native/N/全wrapperは未取得・追加研究は凍結 |
| 旧RPEの全round成功は未検証 | SUP-RPE-V2/ENERGY | 後続の条件付き成功・energy接続を参照。別の費用未決定を成功未検証へ戻さない |

独立した改善方法差がないという既存判断と、科学的性能で敗れた結果は別である。
今回の棚卸しは、これらの既存判断を新しく科学審査・採択したものではない。
