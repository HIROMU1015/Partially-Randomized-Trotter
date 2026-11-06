# O2-S2 複数event最小検証設計：報告

2026-11-06 JST。基点：[79253e0aa761c06ca9439b795d5619c7865f860c][BASE]。専用branch：`codex/o2-s2-multi-event-design-20261106`。

**既存資産で、同じtargetを保つ明示pair-table検証は設計できる。推奨はD2 Closed P5の `r=2, τ=5/7, x=5/14, degree=5`。** G9/G10の `x=5/7` を二回つなぐ対象とは異なる。個別path価格・新proposal・性能は未評価であり、実行条件の採択と科学実行許可が残る。

成果物は [S2_MULTI_EVENT_TARGET.md](S2_MULTI_EVENT_TARGET.md)、[S2_MINIMAL_EXECUTION_PLAN.md](S2_MINIMAL_EXECUTION_PLAN.md)、本REPORTの3件。科学GO、正式研究採択、新規性、総費用改善を判定していない。

## 1. 確認できた事項

固定Rは3-qubit synthetic involutionの `Σp_iQ_i`、`p=(1/5,3/10,1/2)`、σ=+1、state=`000`。分子geometry/basis/DF rank/`L_D`/δ窓は非該当。D1保存row 10は正しいx=5/14のevent・係数・実proposal・weight・native IR/価格を含み、D2はそれをleft→rightに接続する。有限targetは `[P5(-i(5/14)R)]²`、行列順はright @ left。

D2にはliteral inverse cancellationとnative価格取得が実装され、旧結果も完了している。ただし保存値は元lawでの集約で、個別pair価格表ではない。D2登録metadataのClosed P5は63 step events・3,969 ordered pairs。今回の新規集計ではない。D1当該eventsをGitHubから取得し、固定blobとの対応を確認した。

G9/G10 x=5/7のIR/価格はそのままr2入力へ使えない。D3には同入力の保証/価格caseがあるが、今回は旧D2保証・旧cacheの再利用を提案し、保存された優勝case/rの再選定を行わない。

## 2. 再利用する処理と必要最小限の不足

| 対象 | 再利用方針／不足 |
|---|---|
| path本体 | D1 row 10と同cache。新H/state/reference/angle/IR生成を必要としない案 |
| 境界価格 | D2のinverse/depth/native-priceとsuffix cache手法を利用。薄いpair-table接続は必要だが今回未実装。集約値から個別価格を作らない |
| law/normalizer | Parentのexplicit-table ISとRAのroot/dyadic規則。全pair価格取得・normalizer和・table費用が必要。価格依存の非列挙DPは未接続 |
| weight/統計 | 実joint qに対する `Aij/qij`、Parent event統計とBernstein scalar式。元weightを保持し、r1のParent ledger全体や独立moment積は転用しない |
| full価格 | controlled本体にwrapper/ancilla準備/readoutを接続する仕様。未知h・online費は0にせず、主指標・価格境界を採択する必要 |

適用AGENTS/PROJECT_MAP、利用可能な正式共通指示2026-10-25の§8.1/8.2/8.4/10.1/10.3、直前preflight、[O2技術調査][O2]、[三環境統合][INTEGRATION]、[文献調査][LIT]を継承した。今回はD1 row 10、D2契約/kernel/runner/tests/結果、D3条件、Parent/RA/統計とwrapperの関係に絞った静的追跡であり、既存全台帳の再作成ではない。

## 3. 次の科学検証案

Closed P5を一armに固定し、元product law P0、加法full価格で作る既知IS qA、境界full価格で作る既知IS qBを、それぞれ加法回路A・境界回路Bで比較する六cell案とした。

固定lawのA/Bで境界効果、同じB上のqB対P0およびqAでsamplingの追加効果を分ける。既知ISにも同じ価格情報・cache・相殺・保証を与える。baselineにも同じ有限table M2/LとBernstein式を使い、新候補だけtightにしない。旧Nは歴史anchorであり、新lawのNを据え置かない。

全positive pairの価格、normalizer、finite-bit q/weight、M2/L/N、期待full費用とsetup/未測定online費を取得する限定forecastを提案する。sampling/RNG・信号matrix・build/compile・新synthesisを伴わない設計にできる。既知 `a/√C` の再適用だけで新アルゴリズムとしない。全RA/CTSの最良性や非列挙効率はこの最小案では検証しない。

## 4. 実行可能性と次の担当

| 項目 | 状態・次の担当 |
|---|---|
| 同targetの複数event入力 | `TARGET_IDENTIFIED`。上記D2 r2案が成立 |
| 最小実行仕様 | `DESIGNED_WITH_PENDING_CHOICES`。完成した実行契約ではない |
| 主資源・精度/α・wrapper・confidence範囲 | **ユーザー／GPT**が、既存D2条件/T主指標/六cellの再利用案を採択するか選択する。今回正式固定していない |
| zero-price・古典費 | **ユーザー／GPT**が、zero-priceならSTOPする案と、online実測を含めるかを選ぶ。regularizerやbenchmark数/seedを無断で新設しない |
| 実装・科学再評価 | **ユーザー**の明示許可後に、**Codex**が必要最小限の接続・人工検査と承認範囲の価格/law/統計/費用評価を行う。現時点では未許可 |
| 非列挙S2・分子全PR＋RPE | `NOT_ESTABLISHED`。normalizer/online生成、物理H/厳密groundの利用条件、DF/PF部分、round/branch/energy保証、full費用を別途接続する必要 |

同価格しか得られない、range/ceil/古典費で利益が消える、強いqA/B baselineに追加利益がない、価格/target/phase対応が成立しない場合は、拡大を支持する材料にならない。改善率や科学GOの閾値は今回定めていない。重要科学レビューの開始許可は科学実行許可と独立である。

## 5. 文書確認・公開・保全・STOP

変更は新directoryの3 Markdownのみ。科学結果・正式採択・evidence statusを変更していないため、研究概要・manifest・既存索引は更新しない。本REPORTを入口に固定SOURCE・成果物へ辿れる。恒久validator、承認台帳、管理scriptは作成しない。

文書確認は標準ライブラリだけで、既存Phase 2A checkerの `git/blob_oid` を再利用した。3 MarkdownのUTF-8/LF・余分な空白、7相対link、28参照link使用、19固定Git path、非公開path/認証情報の混入確認はPASS。科学SOURCEをimport/実行せず、価格・確率・moment・Nを計算していない。専用branchへcommit/pushし、remote SHAと公開3文書の取得内容を照合する。最終チャットに公開結果を記す。

新sampling/RNG replay、保存科学値の新集計/再ランキング/最適化、path価格数値評価、build/compile/native synthesis、H/state/reference生成、sampler/回路生成実装、新科学test/シミュレーション、重要科学レビューは未実施。Pythonによる保存JSON field/schema確認と文書処理を、科学評価の実行と混同しない。

元dirty checkout・main・旧研究branch・固定SOURCE・raw/results・契約/grant/ledger/marker/STOP、前段O2/preflightとLOCAL準備案は変更しない。旧D1/D2/D3/Parent/G系列/C1等の終了・凍結条件を維持する。

**本設計の公開・取得照合でmandatory STOP。** 次の科学計算・実装・重要レビュー・候補追加を自動開始せず、旧one-shotのretry/reuseや新namespaceでのSTOP回避を行わない。

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/79253e0aa761c06ca9439b795d5619c7865f860c
[O2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/docs/research/o2_sampling_return_technical_survey_20261106
[INTEGRATION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[LIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
