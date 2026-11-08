# 重複分類と次に必要な最小scope

2026-11-08。固定証拠の到達点とCodexの提案を分ける。
[REPORT](REPORT.md)／[条件・費用・SOURCE表](COST_IMPROVEMENT_MATRIX.md)。

## 1. 同じ機構の重複分類

分類はsnapshot時点の成果物に対するもの。別担当のworkerが現在走っているとの認定ではない。
「他Trackで進行中」は既存担当・設計・実行範囲があるという意味で、今回その計算やSOURCEを操作しない。
複数状態がある場合は、完了部分と未資格部分を明記する。

| 調査対象 | 分類 | 既存作業／担当 | 追加で異なる問いがあるとすれば |
|---|---|---|---|
| 形式return、Closed P3/P5、G6〜G10 | 実施済み＋部分的に実施済み | Track B/O2。G9/G10 toy native・十分N、H4 C1 sampling完走／凍結、CのP3接続 | 新しい取得構造や同targetの未扱い操作。P3→P5の読み替えや同じtoy再実行は不可。[B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a8299889de93c201bf21e727415d629f5e4b9f48/docs/research/track_b_existing_results_survey_20261108/REPORT.md) |
| Closed P3×共同isometry×対角fusion | 実施済み | Track C、2e38af3b。weighted平均／range／N／全18roundnative | 登録条件を変えず同じ接続を作り直す必要なし |
| Closed後のh／resolution／α共同最適化 | 実施済み | Track C、c5a3c102、extended正本。両P3とFullへ同機会 | 旧REPORTの「次課題」記述から未着手と判断しない。無制限menu細分は今回推薦しない |
| D内部BCH、PF family、round別構成 | 実施済み＋負結果・凍結中 | Track C D内部、混合公式、C2-A。正時間順序・Y4負時間等の診断 | 新機構／情報モデル差がない中央PF探索の再実行はしない |
| singleton／joint isometry、同basis sharing | 実施済み | Track C、旧WP06も再利用済み | jointのdepth tradeoffは残るが、RZ減とdepth減を同一成果にしない |
| controlled-RZZ、parity合算、diagonal fusion | 実施済み | Track Cの強term/merge/network、Fullにも適用 | 既知control削減をPR固有利益として再計上しない |
| 新A／Track D／O2 vacuum-reference gadget | 実施済み＋部分的に実施済み | 既知SE型の同機構。新Aはcompiler A、DはP0、O2は人工／別legacy準備 | SOURCE／phase／law／wrapperの異条件接続だけが差分。新gadget実装ではない |
| Referenceのnative全round費用 | 条件付き完了／未資格部分あり | 新A18round上下界、Track D13round proxy会計 | 同じ結果の再集計は不要。D長q価格／actual資格不足と新A総T不足は別問題 |
| D q64 cohort | 他Trackで担当・限定範囲が決定済み | Track D C8-1、a708a130のsample00修正結果 | 最終総合gate・残7等は同担当で範囲判断。現在の未公開進行状態はUNKNOWN／資料未取得。[q64](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a708a13077b0616a6047bbca2ba36f21ae58755b/docs/research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_RESULT.md) |
| D selector／cut／head-tail C1 | 他Trackで担当・設計済み、部分的に実施済み | Track D O5。cut人工/H4 q4／cohort metadataと、C1静的設計を区別 | C1のP0後の増分だけが未評価。selector事後min・price oracle無料化は不可。[設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2131917656446bd22f294946d7c1064a47332e4c/docs/research/track_d_o5_algorithm_design_20261108/REPORT.md)・[C1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62c7b578b8c0bb3d0079434daf637c587f53c24c/docs/research/track_d_o5_c1_head_tail_static_20261108/REPORT.md) |
| 新A m7 finite CT | 実施済み、全support／全roundは未資格 | CT-4B固定2trajectory／ε1e-11 | m17価格への単純移送、1-shot率×全round会計は不可 |
| m17 guard／43-word／inverse／D | 負結果・保留中 | 新A D1/D2まで、7819602b | 総費用目的に必要との判断前にD3や新guardへ進めない |
| 合成precision配分 | 部分的に実施済み・保留 | C2-B強baseline、同保存poolで最適性gap | .514% feasible差を強対照最適性証明にしない。新Aへの無条件移送不可 |
| DF／Pauli／low-rank／partition | 実施済み部分＋未資格／負結果 | Track A M1/M2/B2、研究C DF fit/Z再配置 | 元H ground・誤差・公平なPF対照・N×whole価格の共通接続。単なるrank／tail norm再探索は既存と重なる |
| Truncated PF／元Hへの誤差接続 | 実施済み | Track C H2/H4/H6、P1/P2相殺分離・有限近傍も完了 | H4はFullだけでなくTruncatedを強対照へ残す。元HをHkeepへ変更する方式と混同しない |
| sampling IS／Parent／SCV／LCU／grouping | 負結果／既知法への還元／未確立 | O2、S2、Parent D+E終了、C1凍結、A2、最新非同値性選別 | 計測アクセス・取得summaryが具体的に異なる場合だけ次仮説。[O2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md)・[選別](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bf88b251fd09caa0a227972ee0f1b56c5888e6a/docs/research/o2_distinct_mechanism_screening_20261108/REPORT.md) |
| C-S同位置residual | 局所完了・負結果、full未資格 | SERVER1のC-S V0、7b576545。旧Track C C-Sとは別scope | 同じresidual機構による局所T増を無視してfull優位を提案しない |
| RPE以外の推定器 | 未検証候補／資料不足の境界 | O6固定文献調査。現PRの有限非unitary/weighted meanへの全費用接続は未確認 | ideal unitary oracleを無料にしない。代替推定器へ移るかはユーザー判断 |
| 別チャット未公開成果・他SERVER rawの最新状態 | UNKNOWN／資料未取得 | 直接閲覧権限・引継ぎ資料なし | 不存在・未実施とは判定せず、既存担当へ所在確認する場合もユーザーの指示による |

研究C（CDF/ENTRY）とTrack C（PF/native資源）は別系列。C2-A/B/Cも新Aの継続ラベルとして再開しない。現時点の他担当の新承認・実行中状態はGitHubの固定報告だけでは確認できない。

## 2. 改善の二重計上を防ぐ具体的な比較単位

1. **basis：** inverse cancellation、run sharing、singleton QR、joint isometryは同じbasis変換列へ作用し得る。順番に得た削減率を加算せず、最終実回路の一つのpolicyを比較する。always-partialで共有を壊した既存負結果も保持する。
2. **diagonal／control：** 単一Pauli control、controlled-RZZ分解、angle merge、parity topology、D境界融合は同じゲートを減らす。強term-controlからのPR固有差を、弱いcontrolからの全差と分ける。signed D境界補正は負の物理誤差ではない。
3. **shot：** normalization／range低下、PF位相余裕、α配分、整数ceilは同じNへ結び付く。Closed P3のlaw利益と共通h/r/α選択利益を別ablationで読む。RPE保証のNと価格校正MC標本数を二重に掛けない。
4. **Reference：** 全evolutionの制御解除と、baseline側directional controlやD/R境界共有の節約は重なる。Referenceにだけ新fusionを与えない。B′／strong／Full／Truncatedは本来の比較taskとscopeで扱う。
5. **合成：** strict wordの再利用は取得計算を節約するが、同じkeyの物理出現T数は消えない。guard改善は費用改善とは限らず、626／630 shots差、emitter/global W/actual adjoint/shellも保持する。
6. **誤差：** 共通joint global scalar、controlled相対位相、channel一致、coherent first momentは別の概念。compiler diskと合成diskを同じ単位へそろえ一度だけ足し、位相／半径へ戻す。未資格roleの0補完や有限例→全supportの外挿はしない。

## 3. 残る候補と担当：Codexの提案

優先順位は「追加効果を判別できる具体性」「既存証拠」「最小作業量」に基づく。未取得Tや予想改善率によるランキングではない。nativeで有効でも全18round総Tの改善とは認定しない。

| 候補 | 目的資源と因果経路／残る根拠 | 既存との差・再利用・重複リスク | 最小検証・STOP／担当・承認 |
|---|---|---|---|
| 第一候補：D C1のhead-tail接口 | phase-correct relative Gaussianで異frameの閉じたhead出口／tail入口を直接つなぎ、追加basisを減らせるか。raw SOURCEには具体的な未接続接口があるが、P0増分はUNKNOWN | 既存transition/sequence、whole-path builder、P0、人工checkerを再利用。head内／tail内／step融合を新利益へ数えない。新Aでの二重実装リスクが高い | 既存案の一種類・最大8人工wrapper。両方式の前後比較で増分0／逆位相なら終了。**既存Track D担当**を優先、ユーザーが既存提案を採択し実装・correctness・compile scopeを許可する場合のみ |
| 条件付き接続候補：Reference×最新Track C回路 | 同じ最新law・basis／diagonal最適化後にもcontrol削減余地が残るか。新A/Dのpathwise数学・native実績が動機で、組合せの追加効果は未取得 | Track Cのc5a3 final policy、既存Reference部品・strong controlled比較を再利用。D C1や既知SEの新実装ではない。joint basisのgauge、D負時間、weight、vacuum位相を新たに結ぶ必要 | まず§5の静的binding。条件が閉じる場合だけ固定短fixtureのcontrol/Reference×PR/決定論×X/Y比較。**Track C回路担当とD担当の調整**をユーザーが指定。合成・長round・N移送は別許可 |
| 表現・partitionによるRTE負担低減 | tail／basis出現／normalizationの同時改善でN×単価を減らす。Track A/研究Cの保存資産はあるが、own-DF oracleの差が障害 | J/U/keep、元H/state binding、既存split、強PFとTruncated資格を利用。既存ENTRYやZ移動、C0/M1/M2を再提案しない | 元H ground課題との適合を静的確定してから、既存候補一組で局所signal／native／N比較の有限案を作る。**Track A／表現担当**。新fit/gridは先に許可しない。目標不一致・単価減とN増が相殺ならSTOP |
| joint isometryのdepth目的選択 | jointはRZを下げたがserial depthを増やす既存tradeoff。もし主目的がdepthなら同じcatalogueの選択目的を調べる余地 | 625 catalogue、4layout、同run/full fallback、価格表を再利用。RZ最適回路の各列を都合よく混ぜない | 先にdepthの定義／baselineを判断。保存値の新Pareto再解析も科学承認が必要。**Track C担当**。全DAG／全T-depthへ自動拡張しない。総T目的の第一候補ではない |

候補の追加価値が静的に不明なまま計算を提案しない。特に「samplingを変えればよい」「precisionを上げればよい」は実行可能な機構の指定ではない。O2の安価summary／新測定アクセスが具体化されない限り、同じIS/CV/LCUを再実行する候補にはしない。

## 4. 推奨する次の最小scope

**新しいコスト削減機構の最小判別としては、Track Dの既存C1人工案を優先する。** [固定NEXT_EXECUTION_PROPOSAL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62c7b578b8c0bb3d0079434daf637c587f53c24c/docs/research/track_d_o5_c1_head_tail_static_20261108/NEXT_EXECUTION_PROPOSAL.md)を入口とし、新A側へSOURCEや担当を移さない。これは既存担当の研究を新規成果として取り直す提案ではなく、ユーザーがその既存計画を選ぶ場合の推奨である。

- 固定入力：既存2-mode q2人工fixture、指定した一種類のhead出口→tail入口。新H4入力・samplingなし。既存gauge／scalar／control branch／lane／cut／actual inverse規約を保持する。
- 比較：S0/S1およびR0/R1のX/Y、最大8 wrapper。StrongとReferenceへ同じ合法relative融合を与える。回路全体の共通phaseとbranch相対phaseを混同しない。
- SOURCE：既存StrongBuilderのtransition/sequence、C3 whole-path、Reference、P0、既存2-mode exterior/CAR checker。新Gaussian合成器、Reference gadget、科学管理基盤を作らない。
- 必要な検証：全入力operator／X/Y／vacuum・χ・adjointのcorrectnessと固定P0後の資源。RZ/CX/depth/size/qubitは別々に保存。SOURCE上の未融合だけで費用利益を認定しない。
- STOP：不適格接口、phase/target不一致、P0後の増分0または負、登録人工scope完了。人工利益があってもH4／長q／13-round／18round／Tへ自動進行しない。
- 承認：この文書は実行許可を与えない。既存担当の現在の許可状況はユーザーが判断し、必要なら当該固定scopeの実装・build/compile・correctness実行を明示指定する。今回Codexは実行しない。

q64の未完了cohort資格はこのC1とは別課題である。**費用proxyの信頼性を先に上げたい場合は、ユーザーがTrack D担当の既存q64残範囲を優先できる。** それは新改善機構の実証ではなく価格資格の補完であり、q64一箇所からq4096までの外挿保証にはならない。

## 5. 組合せ候補を進める前に固定すべき差分

Reference×Track C最新pipelineの追加効果を確認した資料は、本調査の固定資料集合では見つからなかった。不存在証明や独立新規性の認定ではない。まず既存担当が後続成果の有無を確認し、下記bindingを一組だけ具体化するのが実行前の最小作業である。

| 固定するもの | 解消すべき不一致／使用資産 |
|---|---|
| 比較目的・baseline | 同じ元H ground／εE／failure／prep除外で、PR内の改善とFull/Truncatedへの優越を分離。H4ならda8b2c3f／92495a04の強対照を残す |
| 入力・law・round | c5a3 extended/POLICY／ROUNDSを原本とし、まず既存保存短trajectory一つを事前指定。canonicalとClosedのweight/range/normalizerを無断変更しない。価格によるtrajectory選別なし |
| evolutionの意味 | 制御済みnativeからcontrolを機械的に剥がさない。uncontrolled factorization、D負時間、signed event、scalar、basis completionの実数値、vacuum χ、actual inverseを旧builderで一致させる |
| compiler／context | Track C compiler AとD P0は同じではない。片方の価格・局所保証を移すのでなく、採択compilerの同じfull-wrapper scopeで両側を扱う。新角度・contextは再資格が必要 |
| 最小科学call案 | 条件が閉じた後だけ、固定一PR trajectoryと一決定論構成に対してcontrolled／Reference×X/Y、最大8 wrapperのbuild/compile/correctness。これは候補案であり、物理入力・選択規則・source・閾値が未固定なので現在実行可能scopeではない |
| shot／総費用 | 局所価格だけでNを流用しない。joint disk、coherent mean、finite-bit law、半径／位相／α／整数Nを資格化できる場合のみ後続へ。これらの科学再解析・18round再会計には別の固定範囲と明示許可が必要 |
| 公平性／STOP | 同じcontrol・basis・diagonal機会、両armの追加register/reset/CSWAPを含む。適用条件不足／強対照が同じ利益を得る／増分0ならSTOP。全supportTを目的に不足wordを自動取得しない |

この準備を先に行う場合の担当はユーザーが指定するTrack C／D担当であり、今回は新A担当による新実装・新計算を開始していない。

## 6. 今回保留し、自動再開しない作業

- 新A D3 inverse defect厳密評価、basis全体actual-adjoint化、composite wordの新strict guard、新word合成、m17 strict誤差会計追加改善、43-key runtime全面資格。
- V0/V1中央precisionの再探索、S1/S2再実行、D1/D2の追加pool／precision拡張。domain・central・emitter・shell・terminalは未資格のまま。
- C2-A/B/C、旧C-N/C-S/C-P、P-D、BF/BM、PR-3、Track B C1凍結、Parent終了等。今回の候補名でSTOPを解除しない。
- 他担当のq64実行・cohort・compile・N更新への介入、共有設定変更、他worktreeのmerge／上書き。
- 新しい科学契約／承認証明／grant管理基盤、未取得価格0補完、異条件の率の加算、代替推定器研究・重要科学レビューの自動開始。

**今回の静的調査・比較表・資産整理・次案は完了。公開commitのremote SHA／tree／3文書本文照合後mandatory STOP。**

**次の担当：GPT／ユーザー。** 主資源、強baseline、第一候補C1か価格資格補完かを判断し、必要なら既存担当に固定scopeを指定する。Codexの追加科学計算は未着手であり、今回の包括指示をその実行許可にしない。
