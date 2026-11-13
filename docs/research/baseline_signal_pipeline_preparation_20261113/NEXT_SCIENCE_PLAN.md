# 必要最小限の有限科学実行案

**未承認・未実行。** 今回の技術実装許可から科学実行へ進まない。共通研究目的は元full DF-Hの同一energy taskに対する全round `Σ N E[C]` の最小化（P=0）。Track A最良・旧11案・本pilotのrecipeはRPE最適候補ではない。

## 1. metadata固定と重複の控除

**C/A**は公開済み成果だけから、B configuration、physical ID、全round schedule、H/state/snapshot、full-H preparation、reference descriptors、数値SOURCEとfresh出力を固定する。実行中の他担当worktree/checkpointには触れない。**B**の共通energy loss/prior/decoder/精度契約とh/Mを同じにする。

旧20セルrunnerを再起動しない。後続公開結果で既に完了しているmean/referenceは再計算候補から除き、旧source/backendと今回domain・数値gateの互換性を記録する。単純copyで足りない旧spectral meanのdirect-expm bridge/再資格は、新解析として対象セル数と処理を明記し承認scopeへ含める。現在の残15件という過去の記録を最新coverageと断定しない。

## 2. 最初のwhole-RPE signal接続pilot

まず最大3 schedule（各baseline familyから1、h=.1/M12）×13round、**最大39 complex mean**を上限とする。保存一致分はcopyし、新処理数から控除する。X/Yは同complex meanから渡し、2倍に数えない。B0/B0I両方を追加するなら別scheduleとして上限に含める。

具体的なmetadata固定の初期案は Full PF S2/Δ=.1/全10、Truncated PF B0I S2/Δ=.1/prefix7、Full-H pure RTE K2/各round q=2^m・R448。最後のlayoutは旧研究の接続用seedであり、必要最小event数や最良配置ではない。既存screenの公開coverageや並行Aの候補が別recipeの有用性を示す場合は、**実行前manifestで置換して理由を記録**する。複数recipeをその場で自動探索しない。

このpilotは3familyの全round入力を得るための最初の有限案。S2だけに最適化範囲を固定しない。標準Y4/M5、他Δ、B0/他prefix、他K/q/R/Mの比較は [前段探索計画](../rpe_baseline_optimization_design_20261113/BASELINE_OPTIMIZATION_PLAN.md) に沿い、必要neighborを次の有限batchで選ぶ。

実行前に全13時刻の同full-H reference pairを確認する。欠ければそのroundはmissingでSTOP又は未評価のままとし、新reference計算を自動追加しない。新referenceが不可欠な場合は、担当Aが最大欠測時刻数、既存reference方法、二重確認・保存条件を定め、meanのscopeと一緒にユーザーへ提示する。今の計画は新reference actionを承認しない。

**許可を求める処理**は、固定manifestの実snapshot読取/既存行列構成、必要prefix scalar（保存値がない場合の元抽出）、PF signed直接expm＋matrix_power、Full-H finite-K polynomial/normalizerの独立raw/corrected、保存bridgeの指定分のみ。新sampling/compile/N/RMSE/G/最良選定は含めない。

fresh出力へPLAN、各round record、B mean packet、数値失敗値、SOURCE/入力来歴を保存する。1contextで共通行列・指数・spectrum等を共有し、必要候補だけ評価。実行順に数値gate又はSOURCE不一致が出たら以降を未開始として停止する。閾値緩和、投影、別q/K、元結果上書き、自動retryなし。全指定mean完了又は失敗保存後mandatory STOP。

## 3. 総コスト最適化へ接続する追加条件

- **B/統計担当**：同じ原target/prior/decoder/全事象energy riskに対するNを既存solverで評価。数値gateは保証ではなく、producerのstate/数値/経験的allowance・law契約が不足すればUNKNOWN。MC/Nの方式・回数・scopeは別の有限科学実行として承認する。
- **D/価格担当**：実行可能性と有用性が確認された全scheduleの同physical/law/profile/full wrapperにCを供給。Full PF/Truncated PFの直接価格とaffine検証、Full-H固有law平均・rare/cohort/残差を担当する。イベント数/構造価格をcompiled下界としない。価格SOURCE `24b0a6eb` の探索設計を再利用し、新compile/fitを本pilotに混ぜない。
- **A/PR担当**：同taskのPR候補/meanを既存pipelineで供給。baselineをPRのprefix/formulaへ拘束せず、説明用の同prefix比較とfamily内の最小G探索を分ける。
- **B**：Nと全round Cの資格・不確かさが揃った後、実G集計/有限menu選定を別許可で実施する。欠測を0で埋めず、未探索family・価格区間・条件付き結論を明示する。

最初の新scienceに必要なのは、**ユーザーによる上記の確定したSOURCE/最大39 mean・指定保存bridge・reference/scalar処理の明示許可**。承認証明書や新grant発行は不要。未固定の入力又はreference条件が残れば、その不足を埋めて具体的なmanifestを提示する。本技術納品はここでmandatory STOP。
