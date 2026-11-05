# 重複・継続・既存資産の再利用地図

2026-11-05 JST。[台帳](IMPROVEMENT_VERIFICATION_LEDGER.csv)のIDを使用する。
本書はmetadataと既存判断の対応付けであり、算法の新規性を新しく審査するものではない。

## 同定単位と重複排除

同じ科学実験かどうかは、元SOURCE、入力/H/state/sector、lawとfinite target、時間・split・次数・精度、compiler/control、attempt/cohort、原結果の対応で判断する。
文書のcommit、収録先、file数、test数だけでは決めない。SOURCE commitが同じでも新cohortは別の確認になり得る。
反対に、SOURCEが変わっても既存結果の再会計・再掲載だけなら独立実験ではない。

台帳の `identity_group` は索引・記録を辿る補助キー。`LEGACY-*` は研究familyの要約で、単一実験の同一性認定ではない。
MIX-F2/ATTR/CAUSEは同じF2結果を参照する。G10は一部G9資産を含むため、行全体をG9と同一にも完全独立にも分類しない。
`related_records` と `relation_type` を併読する。UNKNOWNのSOURCE・cohortを推測で埋めない。

| 区分 | 確認できた対応 | 数え方・限界 |
|---|---|---|
| 同じ実験の保存コピー | mainのF1/F2/PS48/G6〜G10/C1主報告と固定元文書の同blob対応。11/3〜4の収録索引も使用 | 新実験0。確認は選択文書のbytes/blobであり、private raw全量同一性ではない |
| 同じSOURCE・結果の再掲載 | F2資源報告→寄与分析→SOURCE機構説明→終了handoff。G6原結果→v1.1/v1.2整理。旧70カード→Phase2A→11/1地図→今回索引 | 原結果を再引用。再解析・文書管理を科学的独立確認へ加算しない |
| 同入力での独立確認 | Track Aの記録された別価格cohort、研究Cの005/064各候補の独立価格は関連証拠として参照 | 固定入力・nominee・cohortの範囲。異候補005対064は同入力再現でない。改善方式の全task独立再現を確認した意味ではない |
| 条件を変えた追加検証 | F1 K2/q2→F2 K0/q1、G9 P5→G10 m3/m7、C1選択event→whole-path、F2→PS48 singleton置換 | 条件変種。元の全scopeを再現したとはしない。PS48は元F2の48target/費用を再利用 |
| 技術的失敗後の修正版 | F1 metadata修正、F2 phase保持修正、G9 API修正、G10 v3、C1 R1/R2/P5R | 元failureと新attemptは別。旧grant/marker/STOPを成功へ書換えず、新結果に後続リンク |
| 後続による状態更新 | P-A非退化比較、P-C tracking、P-D強対照、G9/G10完了、C1分子logical完成、singleton一時終了 | 更新された主張・工程だけに適用。既存成功も負結果も削除しない |
| 同一性を判断できない | 旧UWC候補rawと元run、修正済み旧DF入力、未記録SOURCE/cohort、非公開全raw | UNKNOWN/証拠未発見。未実施・性能failureと同一視しない |

## 修正・継続・終了の鎖

| 系列 | 時系列の対応 | 再利用・独立性で注意する点 |
|---|---|---|
| 混在辞書 | MIX-2C→P0→I→B-PRE/BENV→B→F1-OLD→F1→F2-QUAL→F2-OLD→F2 | 2C-R1反可換pair案は既知対応で閉じ、実装された元S/P方式へ混入していない。F2-QUALの統計資格STOPは限定資源契約の完了後も残る |
| F1 | SOURCE `31e8391a56fbb1a0e64f477f78e242655329de99` のmetadata setter停止→`8192530b17d6bc4af195c734ff43a6e78c506942` の全登録PASS | 保存B入力を再利用。旧20科学testを実行し直したという記録ではなく、登録handlerによる検証。人工test PASSとF1数値PASSを分離 |
| F2 | SOURCE `9b1e103102f918bb157dea90c7eaf3acb5d88b6c` のcompiled operator不一致→`9468c7089727f5648d9f59cddbaebe0a42f5f0a0` の位相保持修正版全件PASS | 修正診断の保存3失敗case＋対照と、新しい全登録runを別作業として保持。未評価旧caseを0補完していない |
| PS48 | MIX-F2→ATTR/CAUSE→候補A準備→PS48→SELECT→CLOSE | 48case等重みとF2全law重みは別。選択規則は未実装。既存局所利益は残すが、独立方向の一時終了を尊重 |
| G6〜G8 | G6形式構成→G7 conditional provider IR/native primitive→G8 on-demand trace | G8の20keyとG7のsequence一致は同backend/seed取得。interface trialsを新しい価格推定標本にしない |
| G9 | v1のFraction/API技術停止→別SOURCE v2で11row完了 | General FullとClosed P5は同理想受理ensembleの内部実装差。有限bit全trial lawや費用まで同一ではない |
| G10 | v1 RSS契約停止→v2 JSON key型停止→v3完了 | 旧失敗prefixは科学判断に使用不可。v3のm5六行・19keyはG9から再利用し、confidenceを再配分。新27keyやm3/m7と分ける |
| H4 C1 | 旧partition STOP→別R1→state/hash STOP・reference修正→R2選択correctness→5A部分完走/P5停止→P5R完走→PAUSE | Ordinary/Full各1024pathは旧5A、P5の1024pathは新P5R。旧2方式を再実行していない。cache/worker/計時境界が違い、CPU比を算法速度順位へ変換しない |
| P-A/P-C/P-D/B-F/B-M/R3 | 初期支持・技術中断・保存replay・強対照・既知対応・STOPの旧カードを継承 | 初期pilot支持を消さず、現在成立しない固有claimを再提案しない。B-M/R3の方法差不足は性能実験で敗れた結果ではない |
| M1-S/FR | Gate修正・finite-signal・gamma/π3/Stage S、scalar補正・非一様FR・technical note | 同保存signalの再解析を新cohortにしない。visibility/strict bound witnessと実用shot/native利益は別 |

## 再利用候補と適合条件

以下は静的に関数・クラス・runner/testの対応を確認した入口であり、今回実行していない。
固定SOURCEには、台帳の元commitを使用する。同名main版を無条件に採用しない。

| 再利用資産 | 適する用途 | 次に必要になる適合確認／不要な再実装 |
|---|---|---|
| `src/trottertracks/pr_dictionary_validation/circuits.py` の `BasisProvider`、`build_event_sequence`、`append_central`、`actual_adjoint` | frame/Pauliの順序保持共有、control、full-wrapper意味論 | scalar/identity、little-endian、basis key、support、角度、compilerを合わせる。basis共有・diagonal-only controlを新規機構として再実装しない |
| 同packageの `gates.py`、`scripts/run_pr_dictionary_f1_saved.py`、`scripts/run_pr_dictionary_f2_resources_saved.py` | 保存inputでのcorrectness/位相保持compile、元F1/F2比較の参照 | F2修正版SOURCEを選ぶ。旧metadata/phase不具合の版へ戻さない。runnerは既存固定契約を持ち、今回起動不可 |
| `tests/management/test_pr_dictionary_compile_metadata.py`、`test_pr_dictionary_compile_phase.py`、元F1/F2 tests | 既知技術不具合への回帰・mock設計 | test定義・過去PASS・新科学validationを分離。fixture生成を伴うtestを単なる文書検査として起動しない |
| `src/trottertracks/algorithm_codesign/return_aggregation.py` の `ReturnKernel`、`rational_event_weight` | return局所query・形式恒等式・有限bit補正の既存基盤 | involution、正有理p、奇数degree、short-step domain、zero-fill全trial、support/weight/rangeを一致させる。Green/Euler/ISを独自新原理として再実装しない |
| `g7_generator.py`、`g8_pipeline.py`、`g9_native.py`、G7〜G10 runner/key inventory | canonical/Full/P5、on-demand、native angle/actual inverse会計 | bit精度・strict global/control phase・finite target・provider・準備費条件が必要。G9 cacheをG10の独立取得と数えない |
| `parent_allocation.py` とD1〜D3/RTE reallocationの既存accounting | allocation、proposal、synthesis位置、保証×価格の対照 | 訂正後KKT・同finite target・literal inverse取消を全方式に与える。固定toyの継続探索は旧closureで停止中 |
| C1のR1 packet・R2 reference修正・P5R prefix selector/fresh provider | 分子tailへの技術接続と失敗の回避 | R1 exact modelと元DF exponentialを区別。旧zero符号bit、partition、whole-path scopeを維持。native/Nの欠落を勝手に補わない。C1凍結を解除しない |
| `src/trotterlib/df_rte_qiskit.py` と `df_partial_s2_repeated.py` | DF event、diagonal-only control、repeated step境界融合 | generation/weight順、LD、identity、一体保持、実数precisionを合わせる。実装済み融合を新アルゴリズムの差に数えない |
| Track A/PauliのSOURCE・cohort・既存G01/G02 | 強いbaseline、nominee/探索/確認・copyの照合 | DF12≠DF10、DF LD≠Pauli prefix、B3一体保持≠Pauli純RTE。既存実験・価格を再生成する必要は、この棚卸しにはない |
| 研究Cのoptimizer/native/calibration/fixed-word path | PR-aware表現の関連証拠、同J/U強D、有限correctness | candidate005/064、CDF8、sector36、own-DF ground、CA条件を保持。旧UWCの欠落raw回収と混同しない |
| RPEの既存decoder/accountingとV1〜R1/enclosure | fixed-split/round schedule、normalization/半径、branch、条件付きenergy接続 | 保存state、signal kind、外部alias、ideal IID、数学的decoder、直接compile domainを保持。全RPEを最初から実装し直さない |

この調査で新しい科学ライブラリ・実験runner・承認機構を実装する必要はなかった。
既存CSV/JSONとGit metadata処理の方式を利用し、専用出力schemaへの一時的な標準library変換だけを使用した。

## 3環境の扱い

取得した[SERVER 1差分](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0a2b54650527094306f619e3d08dbb2bbc94ba14/docs/research/pr_server1_evidence_delta_20261105/REPORT.md)と[SERVER 2差分](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5b0971a2d9fc6ef474008d26d14af85e324b4b24/docs/research/pr_server2_evidence_delta_20261105/REPORT.md)は主に資源評価の追補で、今回の改善領域専用棚卸し報告ではない。
SERVER 2のS2-D10も、C1は既存統合済み・新しい科学監査なしと明記する。
この2報告の取得をもって改善領域の3環境統合完了とはしない。

SERVER 1の境界R比較は旧確認対照を再利用し、SERVER 2のH8 Stage Cは新PR確認cohortと保存PF価格を組み合わせる。
後続[3環境資源統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md)で既に反映された部分もあり、今回新しい独立実験へ加算しない。
旧H8主表とMechanismの同cohort alias、B3主要表とprivate raw全量の可用性差もそのまま保持する。
現在のSERVER 1/2と過去のSERVER_A/B、保存copyと実行hostを名前から対応付けない。

改善専用報告の未取得、private原本未監査、SOURCE/cohort不明は残るUNKNOWNである。
同じ報告が複数環境に保存されているだけでは、その環境による独立確認にならない。
