# H6 Pauli-JW：Hybrid PRと純RTEの探索機会の調査

2026-10-27 JST。固定基点 `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`、専用branch `codex/h6-hybrid-search-fairness-audit-20261027`。
**判断：`LIMITED_EXTENSION_USEFUL`。既存の有限候補比較は報告できるが、Hybridのr上限側に、少数の追加検証で調べる価値がある未探索領域が残る。**
これは保存coverageに基づく検証計画上の判断であり、新しい費用順位や主要科学claimの正式採択ではない。今回の新signal/B/N/G/SE/区間計算、sampling、build/compile、RNG replay、状態作用はすべて0。成果保存・公開照合後mandatory STOP。

## 1. 何が比較でき、何がまだ分からないか

最新保存表のH6では、全5精度で純RTE `H6_p000_q8_r64_K2`（s512、独立確認n48）の条件付き点Gが保存Hybridより低い。この事実と、15条件中14条件で純RTE低Gという[既存総合資料][summary]は変更しない。
一方、H6 Hybridの登録・実行済みrは2/4/8のみ。厳精度の保存最良prefix78/q8/r8/K2はr上限にあり、同じprefix/q/Kでr2→4→8の保存Gは改善している。r16/32/64のHybridのsignal・価格がないため、**「Hybridにも同じ大event数領域を探索させた後でも純RTEが低Gか」は未確定**である。

| 観点 | 今回の判断 | 根拠・留保 |
|---|---|---|
| 共通の物理・測定・会計条件 | 保存family比較として対応する | 同じH6 model/reference、共通kernel、central control、compiler、ρ=0会計。prefixによりtail/law/Bが違うことは設計上の差 |
| 既存有限集合の実測点費用の記述 | 維持できる | 取得済みcohortを比較している。新候補の結果は元集合の記述を無効にしない |
| 両familyの探索機会 | 少数の補完が有用 | 純RTEはr64確認・r128探索、Hybridは改善中のr8で止まる |
| 未探索Hybridの競合可能性 | 排除できないが、逆転の見込みは未定 | 新候補固有B/bias/u/N/C/Gがない。現在の差や単回価格の外挿だけでは除外できない |
| family一般・最適化後の方式優位 | 支持しない | 大域最適性、全候補母平均、実compiled精度は未保証 |

`FAIRNESS_SUFFICIENT`だけで探索の問いまで閉じる根拠は不足する。一方、未探索領域が実際に順位を変えると判明したわけではないため、この限定的保存claimを重大な誤比較と確定する`MATERIAL_ASYMMETRY`ともしていない。coverageと方向性が確認できるので、全体を`INSUFFICIENT_EVIDENCE`として終える必要もない。**未確定なのは改善量・新順位であり、r>8の取得不足自体は確認できた事実**である。

## 2. 固定条件とSOURCEの対応

Linear H6、隣接1.00 Å、STO-3G、中性singlet。保存Pauli-JW918非identity項＋constant、abs係数順/label tie-breakのprefix、同じraw ray/full-H reference/exact norm。S2、T=.8、central-rotation control、cosine/+Im sineのfull Hadamard wrapper、状態準備P=0。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、topology/routing追加指定なし。
Finite paired Taylor RTEのsigned product/rotation/order phase/constant phaseとCDF-grid effective law、Bを一回だけ適用する規約を維持する。K2はorder0/2、K4はorder0/2/4で、KはPFの次数とは違う。

主指標は保存analytic Nと軸別compiled RZ価格Cによる `G=N_cos*C_cos+N_sin*C_sin`。仮想compiled残差ρ=0、ε_sig=.05/.01/.005/.001/.0005、α_axis=.025の既存会計である。Nは量子shotの十分条件による会計、nは古典的な価格標本数。5精度は価格標本を共有する。

[H6公平性SOURCE][fair-source]と[event拡張SOURCE][ext-source]を読み、model/referenceの保存SHAが元H2/H6入力と一致することを確認した。両SOURCEの `pauli_signal_adapter.py`、`rte.py`、`h4_d2_pauli_control.py`、`fixed_signal_evaluation.py`、`pauli_vector_tight_enclosure.py` の5 code SHAも一致した。新しい物理条件や別controlの価格を混ぜていない。
ただしHybridと純RTEはtail Hamiltonian・component mass・event角・B・保持bodyが異なる。q/r/Kを揃えてもsignalや価格を共有できず、prefixだけの純粋な因果効果の測定ではない。

## 3. H6 Hybridの実取得coverage

[H6_HYBRID_COVERAGE.csv](H6_HYBRID_COVERAGE.csv)は**候補×精度×cohort**の表。既存94候補の探索470行＋既存4候補の確認20行＝490行に、提案のみの4候補×5精度＝20行を別statusで付けた。510行は510実験を意味しない。

- 登録94候補すべてに保存signalと探索価格n8がある。rは2/4/8、qは1/2/4/8、Kは2/4。prefixは74/77/78/79/82/230/455/458/459/460/463/684/687/688/689/692。
- 独立確認はp684/q1/r8/K4・K2、p78/q8/r8/K2・K4の4候補、各n48。探索と確認は混ぜない。
- 科学的な登録台帳は[元batch catalog][catalog]、保存値は[元canonical CSV][canonical]と[元candidate results][original-results]、確認は[保存確認results][confirmation]。既存[inventory][inventory]を再利用し、490行の保存G/SE/N/B/bias/u/価格/nを元JSONへ文字列照合した。
- 価格は分子候補に付属し、精度ごとのsignal適格性とは別。厳精度で不適格な候補にも、緩い精度用に取得した価格がある。G=nullを価格なし、不適格、0、∞へ読み替えない。

| ε_sig | 登録・signal済み | 保存guard適格 | 保存bias不適格 | 数値未確定 | 探索価格あり |
|---|---:|---:|---:|---:|---:|
| .05 | 94 | 94 | 0 | 0 | 94 |
| .01 | 94 | 60 | 34 | 0 | 94 |
| .005 | 94 | 60 | 34 | 0 | 94 |
| .001 | 94 | 46 | 48 | 0 | 94 |
| .0005 | 94 | 26 | 68 | 0 | 94 |

新提案4候補は`NO_PROPOSAL_ONLY / NOT_EVALUATED / NO_NOT_ACQUIRED`。保存不適格とは区別する。空の科学値はnull相当で、推定補完していない。
CSVのprice_SOURCE_commitは実際の価格batchまたは確認planのSOURCE。元batch01/02は0880c38e、batch03と確認はa8c7c7dc。signalは保存されたtight L2精度更新後targetを引用するため、価格SOURCEを数値包含の唯一のSOURCEとしない。元archiveのtarget_refined.jsonとprecision_refinement計画へのlocatorを別列に残した。

### 別branch・archiveの確認範囲

指定固定refs、ローカルの関連Pauli/fairness/Hybrid branch refs、GitHubのcodex branch refs、後続Pauli/DF文書整理branchを照合した。後続`6e3436a1…`は4文書追加、[境界調査db1b5bd…][boundary]は静的調査の5ファイル追加で、新H6価格ではない。H4専用PR探索をH6の結果と混ぜなかった。
[元H6 archive member索引][archives]でもHybrid94＋純RTE7の101 candidate IDを照合し、rは2/4/8だけだった。後続[H6 fairness][fair]と[event拡張][extension]の新規候補はL_D=0であり、Hybrid r16/32/64の取得はない。
この範囲で互換な大r Hybrid価格は見つからない。未知のprivate archiveや未提示環境まで不存在を証明したとはしない。索引の検査で足りるため巨大raw/QPYの全復元は行っていない。静的archive全走査の試行は途中で終了し、結果生成・科学run中断・保全証拠変更はない。

## 4. 保存比較と、r上限側を調べる理由

次は[event拡張の最終比較CSV][ext-best]からの**そのままの引用**。全行confirmation n48/n48。GはRZ×shot、SEは保存paired標本SE。新しい比・区間・順位は計算していない。

| ε_sig | 保存Hybrid ID | G Hybrid exact | G SE Hybrid | G RTE exact | G SE RTE |
|---|---|---|---|---|---|
| .05 | PR_p0684_q1_r8_K4 | 1380373939/8 | 29251.435867506367 | 570714023/6 | 355807.7258949337 |
| .01 | PR_p0078_q8_r8_K2 | 23330078720/3 | 8782258.397075232 | 114140561815/48 | 8895018.366266174 |
| .005 | PR_p0078_q8_r8_K2 | 766686975275/24 | 36075977.32174026 | 19023451889/2 | 35580120.073359944 |
| .001 | PR_p0078_q8_r8_K2 | 8036333388915/8 | 1134433962.3786557 | 2853557405885/12 | 889515353.0322384 |
| .0005 | PR_p0078_q8_r8_K2 | 22593289103905/4 | 6378678743.383964 | 45657735316457/48 | 3558125067.408183 |

RTE IDは全行H6_p000_q8_r64_K2。保存報告のHybrid/RTE約1.814–5.938は狭い僅差ではないが、これは**取得済み候補**に関する差である。現cohortの小さいSEで未探索パラメータの改善可能性まで排除できない。

prefix78/q8/K2の同一系列には、次の保存値がある（ε=.0005、探索n8、出典[元inventory][inventory]）。

| r (s) | B exact | N_cos / N_sin | C_cos exact | G exact | G SE |
|---|---|---|---|---|---|
| 2 (16) | 2170075138882155/70368744177664 | 104019783524 / 144535608252 | 10897/4 | 677127026045768 | 1495579854033.6707 |
| 4 (32) | 7410285687380981/1125899906842624 | 4728603541 / 6593650619 | 11445/4 | 32395799715300 | 82891129969.82928 |
| 8 (64) | 2961839000617845/1125899906842624 | 755225928 / 1053563646 | 12513/4 | 11316691969731/2 | 21629302083.223724 |

この系列ではB/Nの低下がCの増加に勝ってGが下がっており、最良周辺を大rへ延ばす理由になる。一方、retained S2の誤差はrだけでは消えると限らない。保存biasが小さく変わったことから極限bias・Gの下界を今回導出せず、r16以降の改善を保証しない。
純RTEのs64→512改善・s1024悪化は[event拡張報告][extension]の別tailの結果。Hybridへ曲線や最適sを転用しない。単回価格も保持/event/compile相殺が非加法的で、r比例の外挿から厳密除外しない。

ε=.05では別の保存最良prefix684/q1/r8/K4を持つ。[保存費用内訳][fair-breakdown]のB約1.027、C_cos=191267/24、N_cos=9916、N_sin=11735は既取得値である。Bが既に1に近いことはr延長のnormalization面の改善余地を検討する材料だが、biasや価格を含む新Gを評価していないので厳密除外の根拠ではない。prefix78/q8だけではこの別の最良近傍を調べたことにならない。

## 5. 実装上の実行可能性と最小の追加案（未実行）

[H6_FAIRNESS_EXTENSION_PLAN.csv](H6_FAIRNESS_EXTENSION_PLAN.csv)に固定提案IDを4件保存した。すべて`execute_enabled=false`、新cohortなし、n未定、科学的実行登録なし。

| 優先度 | 提案ID | s | 目的 |
|---|---|---:|---|
| 1 | PR_p0078_q8_r64_K2 | 512 | 純RTE保存最良とq/r/Kを揃え、Hybrid側の大event領域の機会を直接調べる |
| 2 | PR_p0078_q8_r16_K2 | 128 | 保存r8の隣接倍増。同じprefix/q/Kで改善継続かを見る |
| 2 | PR_p0078_q8_r32_K2 | 256 | r16と64の間を埋め、局所探索機会を確認する |
| 3 | PR_p0684_q1_r16_K4 | 16 | ε=.05の別の保存最良近傍を1点だけ延ばす |

優先度は検証価値の順で、未知Gの予測順位や実行順の契約ではない。初回案はこの4候補のみ。prefix684のr32/64、追加prefix/K/q、純RTEの再延長、H2/H4の新実測は含めない。旧r8と純RTE s512の成功cohortも再取得しない。

静的実装確認の結果：

- `PauliSignalAdapter`はrに応じたtail短時間とq*r eventを処理し、`pauli_vector_tight_enclosure.evaluate`はretained half/reverse-halfとr回のmean-event作用、constant phase、exact raw normを含む。一般kernelにr8という固定上限は見当たらない。
- ただし`h4_pauli_pr_search.candidate_id/config/law_descriptor`の探索validatorはr2/4/8に限定される。既存`pauli_rte_event_extension`はr16以上を許すが**prefix0/q8専用**なので、そのままHybridへ使えない。旧validatorや旧SOURCEを変更せず、新namespaceの最小接続が将来必要。
- 同じsampler、signed product/rotation/phase、D2 central control、wrapper、Qiskit price処理を再利用する設計でよい。prefix78/684を保持する新law/CDF/角/Bは候補固有に評価する。既存r8、純RTE、PR law meanの別候補から転用しない。
- 同じraw state/referenceは再利用できるが、新rのsignal、B、bias、u、Nは未取得。新しいlaw-mean数値評価と同じshot十分条件による会計が必要。signal両軸適格性を先に判定し、どの精度でも適格でない候補の費用compileは省略する。数値未確定は不適格と区別する。
- 科学SOURCE・入力・runtime・対象・call範囲・停止条件を将来batchの開始前に固定する。新しいH6生成は不要。まず候補ごと1回の既存tight L2 mean評価を想定するが、許容精度・作用上限は実行契約で確定し、不確定なら無制限に再作用しない。
- 適格候補の新探索cohortと、新しい独立確認cohortを分離する。現n8/n48は参考値であり、新nを機械的に決めない。保存比較側SE、新候補の価格ばらつき、検出したい費用差、paired軸共分散、希少event、逐次延長・上限、比較側SE floorを考慮する**数値的標本設計そのものも次段階の承認対象**。今回は新SE/区間/break-evenを計算しない。
- 候補ごとのN/Gを得た後にのみ、保存Hybridと純RTEの全5精度へ比較する。探索nominee・確認前固定・確認後minを分け、近接/欠測/数値不確定なら未解決で終了する。改善しないときの追加全格子、純RTE再sampling、旧STOP再開へ自動移行しない。

一般kernelの対応可能性は、新H6候補の実装検証済みやsignal適格性の証拠ではない。今回は新接続実装・人工数値テストを行わず、既存コードの静的調査と非実行設計までで止めた。

## 6. H2/H4の簡潔な再確認

H2は[追加独立確認報告][h2]で、旧13純RTEのうち8候補を新たに確認し、旧1候補と併せた確認集合を得ている。Hybrid51登録、保存最良p1/q1/r4/K4のn96を再利用。純RTEの保存最良はs4、s64/128の新探索は高Gで、探索上限に最良が残った状況ではない。指定資料内で、H6同様に大eventへの改善が続く一方だけを未探索とする重大な兆候は確認しなかった。ただしHybrid全パラメータの最適性を証明したわけではない。

H4は[PR探索][h4]で141候補（Hybrid128、純RTE13）を登録し、prefix/q/r/Kの適応的近傍を比較した。RTE2候補は独立n384、Hybridは元p36の2候補各n48と、直接比較で追加したp1/q8/r8/K4のn48。[event拡張][extension]のs128/256は悪化し、保存最良はs64に留まる。厳精度Hybridのr8/q8境界は残るため「未探索領域なし」とはしないが、今回のH6大r問題と同列に自動追加する根拠はない。確認集合minと元primaryの差、近い費用差、nの不均衡は元資料の留保を維持する。
H2/H4とも現有限集合の記述を維持し、新候補・新統計判定を追加しなかった。

## 7. 再利用・技術照合・公開とSTOP

既存`inventory_pauli_hybrid_rte.py`の候補索引/CSV/null規約を優先し、今回の[inventory.py](inventory.py)は`csvsave`を直接再利用する。元catalog・canonical結果・confirmation JSON・archive member索引を固定Git objectから読む小さい接続だけを追加した。科学runnerをimportしない。新しく導出したのはcandidate/cohortの件数と整数metadata s=q*rだけで、科学量の計算・候補ランキングはない。

固定GitHub資料15件を実取得してGit blob SHAと照合：総合資料/latest15/verification、H6 fairness報告/SOURCE/費用内訳、event拡張報告/SOURCE/精度別比較、既存preparation inventory/verification、直接比較報告、H2報告、H4報告、境界調査報告。別branch確認は観測したrefと指定refに限定する。CSVの元科学値、各候補のq/r/K、SOURCE locatorとリンクを照合し、元科学結果・STOP・manifest・総合14/15資料は無変更。

成果物はこのdirectoryのREPORT、coverage CSV、非実行候補CSV、読取補助scriptの4ファイルのみ。内部絶対path・private seed/rawを追加公開せず、公開済みrepository相対path/固定refだけを参照する。main変更・別研究worktree介入・新grant/承認証明基盤なし。技術照合はprimary Codexによるもので、独立科学レビューではない。

**次の担当：ユーザーが、H6の上記4候補について、限定signal/B/bias/u/N評価、標本数の数値設計、適格候補の探索価格・独立確認・G比較を承認するか判断する。** 承認する場合、Codexが最小Hybrid接続と人工correctnessを確認し、新固定batchとして実行する。既存成功runの再開や新しい全格子は不要。今回の調査完了後はmandatory STOPし、その科学計算・主要claim採択・詳細レビューへ自動移行しない。

[summary]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md
[catalog]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/artifacts/pauli_h2_h6_comparison/2026-10-25/batch_catalog.json
[canonical]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/artifacts/pauli_h2_h6_comparison/2026-10-25/registered_candidate_precision_table.csv
[original-results]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/artifacts/pauli_h2_h6_comparison/2026-10-25/H6/candidate_results.json.gz
[confirmation]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25d248a1f2ed88de62b0e621c2d1cea8532804c4/artifacts/pauli_hybrid_rte_preparation/2026-10-25/evidence/H6/confirmation01/results.json
[inventory]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25d248a1f2ed88de62b0e621c2d1cea8532804c4/artifacts/pauli_hybrid_rte_preparation/2026-10-25/candidate_evidence_inventory.csv
[archives]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/artifacts/pauli_h2_h6_comparison/2026-10-25/data_archives.json
[fair]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/docs/h6_rte_fairness_20261025.md
[fair-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/artifacts/h6_rte_fairness/2026-10-25/SOURCE.json
[fair-breakdown]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/artifacts/h6_rte_fairness/2026-10-25/selected_cost_breakdown.csv
[extension]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
[ext-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json
[ext-best]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/best_by_precision.csv
[h2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/docs/h2_pauli_rte_validation_20261026.md
[h4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md
[boundary]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/db1b5bdf96c05aa9ff23e9f843c2d3101496c8ed/analysis/pr_parameter_boundaries_20261027/REPORT.md
