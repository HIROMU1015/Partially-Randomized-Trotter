# O2 Sampling・Return Aggregation：PR＋RPE総費用の技術調査

2026-11-06 JST。専用branch：`codex/o2-sampling-return-technical-survey-20261106`。
固定基点：[文献調査 e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8][LITROOT]。統合証拠：[3571477093441ac4852166fe829f8ccc5db5ec4f][INTROOT]。

**残るのは、既知のvariance-aware保証を利用できる条件、非加法whole-path価格を実行可能な費用で取得する構造、非自明returnを安く集約してfull-wrapperまで利益を残す条件である。全RPE総費用の改善は今回実証していない。** 既知lawの再発見、native単価だけの改善、人工API確認からの分子性能の推定を研究成果に数えない。

## 1. 成果物と許可範囲

| 成果物 | 内容 |
|---|---|
| [O2_A_SAMPLING_STATISTICS.md](O2_A_SAMPLING_STATISTICS.md) | 元PR/IS/Parent/相関law、平均・weight・finite-bit、Hoeffding/Bernstein、RPEのN、生成アクセス、反証と再利用SOURCE |
| [O2_B_RETURN_AGGREGATION.md](O2_B_RETURN_AGGREGATION.md) | G6〜G10とC1の到達層、return/native/full-wrapper会計、強対照、負材料、最小検証候補 |
| [O2_CANDIDATE_COMPARISON.md](O2_CANDIDATE_COMPARISON.md) | 独立方向S1/S2/R1、組合せの問いJ1、弱い方向、各候補の9項目と継続/終了条件 |
| 本REPORT | 確認範囲、主要所見、限界、公開・保全・STOPと次の担当 |

SamplingとReturn Aggregationは別担当で並列に静的照合し、別ファイルに記述した。統合比較・報告は主担当が作成した。新しいsampler、回路、統計定理、科学実行契約を設計・実装していない。

元の物理Hの厳密基底状態を同じ条件で利用し、状態準備を除外する。H_D/近似Hのgroundへ置き換えず、E₀/gap/状態vector/真のpath信号を無料入力にしない。目的は各指標別の `Σ_(m,b) N_(m,b) E[C_full,no-prep^(a)]`。新しい精度・成功確率・分子・compiler・主指標は固定していない。

## 2. 実際に確認した資料とSOURCE

| 資料群 | 今回の確認範囲・帰属 |
|---|---|
| 運用・入口 | 適用AGENTS、PROJECT_MAP、研究概要・現状、研究目的の状態/半径/総費用条件、VALIDATION_STATUS/manifestの関連状態。利用可能な共通指示2026-10-25の§8.1/8.2/8.4/10.1/10.3を確認 |
| 固定統合357147… | 指定5成果物。台帳のRQ-04/05、G6〜G10、C1、ALLOCATION、SAMPLING-FREEDOM、RPE関連行と再利用profile・固定参照を追跡。全129行の原科学資料を全面再監査したという意味ではない |
| 固定文献調査e8b19… | 指定6成果物、特にO2、強対照とcost-map、L04/L05/PRHS/qSHIFT/RPEの確認深度と留保を再利用。両固定REPORTの冒頭はGitHub connectorでも取得確認 |
| Sampling SOURCE | `rte.py`の有限law/normalization/event、Parent `digital_proposal/reweight_events/event_statistics`、reallocation `confidence_budget`、sampling freedomと相関監査、関連test定義。詳細な実読箇所・固定版はO2-Aの根拠表 |
| Return SOURCE | G6 `ReturnKernel`と有限bit/serializer/native、G7/G8 provider/cache、G9/G10 generator/native/accounting、C1 `WholePath/AllTrials`とR1/P5R/凍結原文、関連test定義。SOURCE・結果carrier・既存testをO2-Bの表で区別 |
| 主RPE SOURCE | [rpe_resource_accounting.py][RPE]の `_hoeffding_shots/evaluate_rpe_round_candidate/build_rpe_resource_summary` を静的読解。raw半径、座標許容幅、軸別N、同指標full価格、全round/union条件を照合 |
| 後続RPE | [V1][V1]、[V2][V2]、[R1][R1]と[後続統合 §5][ENERGY]の条件付き成功/energy接続、価格外挿と未確定性を確認。e8b19…treeにない後続統合本文は指定固定3380f6…のGit objectから読解 |
| 外部一次資料 | resource-optimal ISの§II/II.1/III、cost-aware qDRIFT、既知variance-aware保証の関連箇所を確認。直接確認/既存調査継承/未取得をO2-Aで区別。全論文の全節精読・独立再証明はしていない |

科学moduleのimport、既存runner/testの再実行、保存科学値の新規集計は行わない。再利用したのは固定資料・式・接口・実装の読み方で、過去の実行許可やgrantではない。元runのSOURCEと結果を今回の実行証拠へ置き換えない。

## 3. O2-Aで判明した事項

1. **平均保存とshot保存は別**。同path familyでの正しいlikelihood補正は平均を保つが、weighted outcomeのrange/M2とNは変わる。X/Y両軸と各roundを含める。finite-bitでsupportを失う、または相関marginalだけを合わせる構成は、この説明で保証できない。
2. **`p/√c`は既知法**。resource-optimal ISのcost×second-moment目的、cost-aware qDRIFTの別目的、whole-path IS、Parent/S4を区別する。有限confidence・整数ceil・range項・生成費まで同じlawが最適だとはいえない。[原著 §II, Thm.1][IS]
3. **variance-aware会計は一部既実装**。Track Bの [accounting][BERN] `confidence_budget` はsignal不要のBernstein十分shotを使う。主RPE経路はIID±1 Hoeffdingであり、「repositoryに分散を使う保証がない」とは言わない。既存保証を主RPEの半径・bias・energy条件へ接続することが適用課題として残る。
4. **前回range-only障害は限定結論として保持**。今回指示§4.2Cが示す前回レビューの結論を、同path集合・同target・同座標許容幅、最大weightによるrange-only十分shot評価という条件で照合した。元レビュー原本の一意の固定pathは特定できていない。新しい不可能性定理、必要shot下界、variance-aware一般の否定にはしていない。
5. **whole-path価格の取得性が主要な空白**。価格をpath生成後に取得できることと、その価格に従って効率的にpathを引けることは別。normalizer、support列挙、online生成、rejection、cold/warm cache、非加法full価格の取得を無料にしない。

厳密元H groundであっても、主RPE SOURCEのPF有効固有状態・単位基準半径が自動成立するわけではない。保存済みの条件付き18round成功/energy接続は保持し、新lawへ自動移送しない。[研究目的 §2.2–2.3][PURPOSE]、[後続整理 §5][ENERGY]

## 4. O2-Bで判明した事項

1. **G6構成、人工API、native、分子logical、全RPEは別の到達層**。人工512件は固定API correctnessであり、分子全supportの性能保証ではない。G9/G10は技術修正版で完了した範囲を保持し、旧停止を未解消へ戻さない。
2. **G9 Closed P5とGeneral Fullは同じ理想full-return ensemble**。相違は取得/生成/normalizer・zero-fillや保守的N会計であり、同じensembleを別量子改善機構として二重計上しない。G7/G8 cacheやG10の既存key/結果再利用は追加独立標本としない。
3. **Full一律優越は支持されない**。G10の各有限P_m内の比較には低次数Closed等に対するT負材料があり、CTSはCXの強い対照になる。異degreeを同じexponential精度の比較に読み替えない。通常logical/RZモデルとstrict native Clifford+Tを混同しない。
4. **C1は完了したlogical経路を保持して凍結**。linear H4 1Å/STO-3G/legacy DF12、generation-prefix L_D3、T4/5、q1、r4、m5、h1/5、TAIL_ONLY_PLUS_SCALARの限定結果である。P5R後に3方式各1024 logical pathが保存されたが、各pathのchild回転が残り、zero回避を観測せず、差は小さいという既存凍結理由がある。[C1凍結原文][C1STOP]
5. **資源主張を変え得る不足に絞る**。physical DF対応、strict native全support、共通有限confidenceのN、D halvesとfull wrapper、生成・return・normalizer取得費が主要な境界。旧worker原因など全UNKNOWNを埋めることは候補判別の必須条件ではない。

zero-fillの全planned trialと量子測定shotを分け、無条件価格とaccepted-only価格を混在させない。return集約でBが変わると統計も変わる。一つのnative回路が安いことだけでは全RPEの節約を示さない。

## 5. 残る具体的な改善条件と既知法との差

| 整理名 | 具体的な条件付きの問い | 現段階の扱い |
|---|---|---|
| S1 | 既知のsignal不要moment/variance上界をRPEのround別有限confidenceへ適用できるか | 既知保証の適用。baselineにも同じ改善機会が必要 |
| S2 | 非加法full-path価格を効率的に取得・samplingへ利用し、range項を払ってもN×単価が下がるか | 既知whole-path ISの適用/改良候補。lawの式だけに新規性なし |
| R1 | 非自明returnが多く取得が安い構造で、child・D halves・合成・wrapperまで戻して価格/統計の利益が残るか | G6〜G10の接続候補。現C1を再開する判断ではない |
| J1 | 集約後のsupport/weight/priceを同時に安く取得する追加性質があり、S2/R1の独立適用を越えるか | 組合せ効果は未確認。独立改善率の加算・乗算は禁止 |

旧SAMPLING-FREEDOMの限定 `NO_DISTINCT_FREEDOM`、Parent/S4の既知最適化、G9同ensemble、G10/C1の負材料を維持する。残る問いの主眼は有利な適用条件・取得可能性であり、新方式の正式仕様や新規性を確定していない。

## 6. 次の最小検証候補と別途承認

ユーザー／GPTが必要な問いを選ぶ場合、次の候補がある。いずれも提案で、今回未実施・未承認。

- **S1/S2**：一つの固定有限ensemble/taskについて、同target・同情報の既知lawと既知有限confidenceを対応させ、range/moment、bias、full単価、取得費を比較する。保存値の再会計でも科学実行許可が必要。
- **R1**：非自明returnを安く利用できる構造を先に説明できる入力一件について、physical bindingと必要なstrict native/full-wrapper・共通Nを確認する。C1を用いるなら、ユーザーによる凍結変更の判断と新作業範囲の許可が別途必要。
- **J1**：追加の取得可能性を特定できた場合にだけ、R1のみ/S2のみ/組合せ/同情報既知法を同taskで比較する。単独へ還元される、生成費が重い、rangeやD部分で利益が消えるならその具体候補の終了を検討する。

計算・native synthesis・実装を開始する許可と、重要な科学レビューの開始許可は独立である。今回はどちらも要求しておらず、既知条件を照合した資料作成として完了する。次の対象/精度/成功確率/主資源と正式採否はユーザー／GPTが選び、Codexは後日の明示指示の範囲だけ担当する。

## 7. 文書検査、公開、保全、mandatory STOP

公開変更はこの新directoryの4 Markdownだけ。固定調査成果物、元科学branch/main、SOURCE/input/raw/result/grant/ledger/marker/STOP、元checkoutのdirty/untracked編集は変更対象にしない。新しい科学結果・正式採択・evidence statusの更新ではないため、研究概要・manifest・既存索引の改訂は行わない。本REPORTを入口とする。

既存Phase 2A `verify_documents.py` の標準ライブラリだけの `git/blob_oid` を再利用し、この4文書のUTF-8・改行・相対link・参照定義・固定commit/path・公開path検査を実施した。87 linkと57の重複を除いた固定Git pathの存在を確認し、検査はPASSした。旧固定namespace用validator全体や科学testは起動せず、恒久validator・新台帳・承認管理物は作らない。公開対象に非公開raw、private絶対path、認証情報の混入を認めなかった。これは文書技術検査であり、科学的妥当性や全外部URLの再取得を認証するものではない。

専用branchへcommit/pushした後、remote SHAと4成果物の取得内容を照合する。最終公開SHA・一致確認は最終チャットに記す。自己commit SHAを埋め込むための再commitはしない。

**未実施**：新しい科学計算・数値再解析・集計/順位/予測値生成、sampling/RNG replay、H/state/reference生成、回路build/compile/native synthesis、新sampler/アルゴリズム実装、科学test/simulation、非自明な新定理の証明、新規重要科学レビュー、C1その他停止実験の再開、GO/新規性/論文化の最終判定。

**mandatory STOP：O2-A/O2-Bの技術照合、候補比較、4成果物の公開・取得一致確認で停止する。** 次の研究フェーズ、C1再実行、新規実装・科学実行へ自動的に進まない。既存の停止・凍結はすべて維持する。

[LITROOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[INTROOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[RPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rpe_resource_accounting.py
[BERN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/algorithm_codesign/rte_reallocation/accounting.py
[PURPOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/研究目的・研究課題.md
[V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md
[ENERGY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md
[IS]: https://arxiv.org/html/2603.13495v1#S2
[C1STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_PAUSE_AND_RESTART_CONDITIONS.md
