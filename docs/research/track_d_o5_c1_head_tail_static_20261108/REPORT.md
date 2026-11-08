# Track D（O5）C1：head-tail接口融合の静的技術調査

2026-11-08。**最終提案：C1継続。異frameのhead出口→tail入口に、既存phase-corrected relative Gaussianを接続する一種類の最小人工判別を検討する価値がある。** raw側の未接続箇所は確認できたが、**P0後の追加削減はUNKNOWN**。実装の採択・科学的GO・新規性は認定していない。

## 到達した判断

1. `head_tail_cross_fusion=False`に加え、C3が閉じたheadとtailをcomposeし、横断`transition`を呼ばないことを確認した。head末尾G_H→tail入口G_T†は具体的な接口であり、単なる未実装フラグだけの判断ではない。
2. 同じsystem/wire上でbasisが非制御、actual frame/gaugeが適合し非可換演算が挟まらない場合、G_T†G_Hへの局所置換は元の全branch演算を維持できる候補。既存StrongBuilderを再利用でき、新Gaussian合成器・SE-QPE gadgetの再実装は不要。
3. head内relative、tail同basis sharing、stepのdirectional半＋uncontrolled半→ordinary full、scalar/identity集約は既に実装済み。これらを新しい利益へ重複計上しない。異modeの中心合算、近似frame一致、global cut越しの別lane融合は無条件に許せない。
4. P0は単qubit列の簡約や指定inverse pair相殺を行い、固定C7-2にも局所相殺の証拠がある。一方、PR frameを使うrelative再合成の横断入口はSOURCE上未接続。P0が実際のhead-tail界面の利益を全回収するかは静的に決められない。`POTENTIAL_GAIN`とP0増分UNKNOWNを併記した。
5. 同じ局所規則をStrongとReferenceへ与え、S₀→S₁/R₀→R₁と、前後のR−Sを区別する。Referenceの相対優位が縮小しても正しい結果として扱う。局所節約の和ではなくfull-wrapperだけで判別する案。

## 成果物

| 資料 | 内容 |
|---|---|
| [HEAD_TAIL_INTERFACE_MAP](HEAD_TAIL_INTERFACE_MAP.md) | head→tail、tail→head、step接続、A†/B lane、global cutのframe/mode/phaseと呼出し |
| [FUSION_ELIGIBILITY](FUSION_ELIGIBILITY.md) | 既存relative恒等式、control branch条件、gauge・scalar・vacuumと拒否条件 |
| [P0_OVERLAP_ANALYSIS](P0_OVERLAP_ANALYSIS.md) | phase-preserving P0の実装、公式level1 pass、局所重複と全接口UNKNOWN |
| [CANDIDATE_COMPARISON](CANDIDATE_COMPARISON.md) | 全5分類、Strong/Reference公平性、A/B/C1の現状 |
| [NEXT_EXECUTION_PROPOSAL](NEXT_EXECUTION_PROPOSAL.md) | 一種類の接口、既存2-mode q2人工例、negative/phase witness、最大8 wrapperという未実行案 |
| [SOURCE_INDEX](SOURCE_INDEX.md) | 固定資料・SOURCE・行番号・出所と取得状態 |
| [STOP](STOP.json) | 静的調査完了、次の科学段階は未許可 |

## 範囲・未取得の証拠

基点は`0d596e7a51a79013bd93cad133b6b14c5b1e2573`、tree `33d188a44c8988f8107d4dd45a6c9108fc239f93`。専用branch `codex/track-d-o5-c1-head-tail-static-20261108`、既存科学コードを無変更のまま読解した。AGENTS/PROJECT_MAP、研究概要・正式研究目的、固定H4/C3/C5契約、正式共通指示2026-10-25版を取得し、§8.4・§10を適用した。詳細科学レビューではなく通常の静的技術調査である。

実H4系列の背景条件はLinear H4/隣接1.00Å/STO-3G/Nα=Nβ=2/元H/number-CDF8/元frame順LD2/ordered partial S2/s1/r32/K0/h=δ=.1、元H厳密ground仮定・準備費除外。今回これを新しく評価していない。候補Bの既存q4 calibration16本は全て中央cutという完了報告を引用し、再検証・再集計していない。

O2の二lane真空Referenceは既知SE-QPEと同じ機構という固定整理を維持する。O2のlegacy DF12/LD3/.02/r1/K2/opt2準備系列はTrack D H4/P0と別で、そこでの融合価格や人工PASSをC1の追加効果へ移送しない。

今回の新最適化コード・fixture生成・人工correctness・P0 compile・保存H4再compile・QPYロード・保存価格数値再解析・sampling/raw simulation・qモデルfit・N/13-round更新・詳細科学レビューは**すべて0**。新しいPASSやcompiled改善率は取得していない。文書リンク・固定Git bytes・公開整合性の技術確認だけを行う。

未取得なのは、対象H4接口の実際の適格頻度、full-wrapperの追加7資源差、P0重複の大きさ、全law/長q/全RPEへの移送・新規性。Gaussianのraw個数や「未融合」という名前から正の費用利益を推定しない。

## 保全・次の担当

元main、他worktree/dirty/untracked、C6a/C7/C8-1/O2のSOURCE/raw/QPY/NPZ/result/grant/ledger/marker/STOPは無変更。q64のcontroller/worker/CPU affinity/memory/SOURCE/input/output/STOPを操作せず、未完了結果を調査・選別に使わず、今回進捗も照会していない。共有OS/scheduler/venv/cache/他ユーザーの設定を変えない。科学計算を行わない静的調査として必要な小規模I/Oだけを使い、資源上限や環境を新設しない。

**次の担当はユーザー／GPT。** 一種類の接口の最小実装・人工全入力correctness・共通P0比較を許可するか判断する。科学計算/compileの許可は今回には含まれず、重要な科学レビューが必要ならその開始許可も別である。C1がP0と重複する可能性を受け入れ、人工増分0なら探索を区切る案も保持する。候補C2やO1/O2/O3共同最適化は開始しない。

文書を専用branchへcommit/pushし、remote SHA・tree・公開bytesを照合した後、**mandatory STOP**。このSTOPは今回のC1静的調査に適用し、別承認で進むq64ジョブを停止する指示ではない。
