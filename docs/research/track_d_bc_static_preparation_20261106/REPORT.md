# Track D：Stage B・Cの並列技術準備

2026-11-06 JST。基点はStage A結果`3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0`、先行静的調査`523efe89fde52a14a8e0a4eba4ea6b70e3f512e1`。専用branch：`codex/track-d-bc-static-preparation-20261106`。

**Bは元H入力での信号・Nの資格、Cはwhole-round構成・期待回路価格の資格が不足している。両者の準備は独立して進められるが、Stage Aの32本のCX／depth削減だけでは同じenergy精度・成功確率の全RPE費用優位を結論できない。**

- [Stage B：再利用可能な統計・保証と不足事項](STAGE_B.md)
- [Stage C：全round Reference構成・費用評価の成立条件と不足事項](STAGE_C.md)

本書と上記2文書は**静的調査・技術準備**である。既存SOURCE／Markdown／JSONの読取りとGit object metadata確認だけを行った。新しい科学計算、保存数値の再解析・集計、sampling、ground solve、DF再fit、NPZ配列展開、QPY decode、回路生成、compile、科学module import、test／runner実行、アルゴリズム実装、詳細科学レビューは0。掲載数値は既存欄の引用であり、新しいphase・radius・N・成功率・価格比・全費用は算定していない。

## 1. 固定した参照と保全

共通AGENTS.md、repository AGENTS.mdとPROJECT_MAP、研究概要・現状、研究目的・方法、VALIDATION_STATUS／manifest、正式共通指示`research_project_instructions_unified_2026-10-25.md`を確認した。正式共通指示はlocal原本を読み、架空のrepository内リンクは付けていない。主エージェント自身の調査であり、AI sub-agentや別担当へ委任していない。「並列」はBとCの独立した調査軸・将来の計算依存関係を意味する。

対象候補はlinear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered S2/s1/r32/K0、h=δ=.1。同じ元H厳密groundを仮定し、状態準備費を除外する。既存13-roundは再利用候補であり、主資源指標、共通εE、成功確率、誤差配分、新しいcutや追加sampling数は今回採択していない。

| 役割 | 参照identity／読む範囲 |
|---|---|
| [Stage A結果][A]／[SOURCE][ASOURCE] | 科学実行`2dd012dc77dcc877d5918c7d212819eb79b97557`、結果`3dc9a61e…`。保存32 path、固定cut16、1 h-step、128 wrapperと測定演算子の数値照合 |
| [サーバー静的調査][SURVEY] | `523efe89…`。CDF8／legacy／別endpointの区別、保存先・law・N・費用境界の入口 |
| [CDF8 r32下流][C32] | 科学SOURCE`4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、実装`703938aba8cc38080e29e75acde8e0edc8dce868`。PRの13-roundと同cohort保存資料 |
| Stage Aが再利用した元C SOURCE | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。StrongBuilder・signed tail・full-frame provider・位相保持compiler。C r32の科学実行SOURCEとは役割を分ける |
| 保存結果carrier | `3649b75af191916daf44abb91e14d34c43b83e8f`。同cohort、PF／corrected kernel、physical／candidate ground保存物の所在。科学SOURCEの代わりにしない |
| 元H packet SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| PR endpoint J/U SHA256 | `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| Cohort／SOURCE manifest | `near_practical_PR_r32_fixed32_v1`／`103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739` |

既知の本人のproject資産と固定Git objectだけを扱い、未知hostへのSSH、projects／temp／履歴の広域探索、他ユーザーのファイル・process・共有設定への操作はしていない。分離worktreeで文書だけを追加した。旧SOURCE・入力・raw・結果・grant・ledger・marker・STOPは変更・再消費していない。Stage Aのmandatory STOPは解除しておらず、今回の許可はB/Cの**技術準備**だけである。

## 2. B・Cそれぞれの到達点

| 軸 | 再利用できる入口 | 追加確認が必要な点 |
|---|---|---|
| B：信号・統計 | 同CDF8のlaw、PF/RTE operator誤差、A、既存radius→phase→Hoeffding N、energy／failure会計、dyadic decoder | 保存信号とNはown-DF ground条件。元H入力の正半径・phase・alias・全誤差の資格、全roundのReference分布対応、implemented-law／合成誤差は未結合 |
| C：全round価格 | Stage A adapter／actual inverse／χ／correctness／P0、強いsymmetric、basis共有／relative fusion、repeated request・境界融合、serial会計 | whole-pathの分割・full ordered law・compiled observable、強い対照の全q構成、全law価格母平均・long-q domain、routing／合成精度等の費用境界 |

Bでは元Hの保存数値groundも見つかったが、physical JSONのstateラベルはown-DFの共通表記である。入力Hとvector／sector順序を確認せず使わない。Cでは一回のReference gadgetとq回の短gadgetの信号保存条件が異なり、旧serial会計へStage A wrapperを直接代入できない。この2点を次の実装・検証の入口条件とする。

## 3. B/C間で受け渡す情報

新しいschemaや管理frameworkは実装していない。次の表は、後続scopeで取り違えを防ぐために必要な情報の一覧である。

| 受渡し | 必要な内容 | 依存しない作業 |
|---|---|---|
| 共通identity | 元H、J/U、sector／encoding、LD／順序、δ/h/s、round m/q/r/K、scalar／identity、signed atoms、law・normalization、SOURCE版 | Bは回路価格なしで信号条件を調べられる。CはNなしで同じordered pathの構造correctnessを調べられる |
| B→C | 候補round domain、exact original-H ground仮定、raw X/Y・符号／読出し、fresh-shot／occurrence・weight規約、許容する実装誤差の単位・資格 | Cの最初の固定path検査に新shot数は不要。13-roundは当面の参照域であり、別εEならhorizonは再確認 |
| C→B | whole-path構成・cut・register・actual inverse・χ、全qの測定演算子対応、compile／合成／routing誤差のcoverageと未確認欄 | Bの元H向けdirect信号条件の評価はCの費用推定を待たずに行える |
| C→会計 | 同じscopeでの資源g別期待価格・paired差、不確かさ、valid q domain、scheduled depthか一体compileか、peak幅 | BのNを使う前でも価格自体は報告可能。費用によるα再配分を行うなら、その選択は別途明示 |
| B→会計 | 同じεE・成功確率を満たす方式別`{N_m,b}`、alias・正半径・phase・law誤差のstatus | Referenceとの同分布が資格化されれば、方式間で同じN列を使う。保存own-DFのNの単純転用とは区別 |

将来同じ意味で加算可能な資源gについて、既存の目的関数は

\[
G_g=\sum_m\sum_{b\in\{X,Y\}}N_{m,b}\,
\mathbb E_{\omega\sim p_m}[C^{\mathrm{no\text{-}prep}}_{g,m,b}(\omega)]
\]

である。今回はこの和を計算していない。必要peak qubitsはこの加算式へ入れず、幅として別記する。depth総和・実行時間・qubit-timeも定義なしに同一視しない。B/Cのどちらかが未資格なら、Gは未確認のままであり欠落項を0にしない。

## 4. 次に必要な最小限の検証案

**以下は提案のみ、未採択・未実装・未実行。** 将来の明示許可で固定された範囲内は関連作業をまとめて進め、各小工程ごとに承認を取り直す手順にはしない。既存の個別契約とSTOPは維持する。

| 順序・並行軸 | 最小の作業候補 | 再利用SOURCE・入力 | 得られるもの／まだ得られないもの |
|---|---|---|---|
| 共通の入力固定 | 元H／CDF／sector／law／SOURCEの組と、実行するround・fixture／cut・費用scopeを固定する | Stage A SOURCE.json、C r32契約、同carrierの既存packet・J/U・reference | 新sampling・fit・ground solveなしで準備可能。科学値の再評価は別許可に含める |
| **B1（Cと独立）** | 元H targetへのoperator誤差接続を優先。保存PF/RTE/AとDF差の適用条件を確認し、必要な元H-relative誤差・radius／phase余裕だけを評価する | `c1_stage3_reference::{reference_kernels,reference_round}`、保存sector kernels／physical_reference、accuracy／feasibility | state-vector不要の十分条件を先に扱える候補。保存own-DF Nの資格を自動継承しない |
| **C1（Bと独立）** | 最初のmulti-occurrence域（q=2が候補）で、保存pathを結果非依存の固定規則で連結した少数fixtureを使い、whole-path Referenceと強いsymmetricのraw correctnessを比較 | Stage A events／CAR、`stored_events`、`StrongBuilder`、`directional_head`、`reference_gadget`、既存check harness | 新samplingなしの構造検証が候補。AB／actual A†／χ／vacuum／X/Yを確認。fixtureの価格を母平均とは呼ばない。新cutは実行前に別途固定 |
| **B2（必要な場合だけ）** | B1が不足なら保存physical vectorによる同roundの数値signal pilot。正当なradius／phaseから必要なNだけを評価し、既存decoderへ接続 | carrierの`shared/ground_physical.npz`、既存matrix-power／raw signal／Hoeffding／decoder | binding・精度資格を明示。新ground solve・大量仮想samplingを初期案へ含めない。解析的十分条件で足りれば仮想測定を省略 |
| **C2（C1 PASS時のみ）** | 成立した構成だけ同P0・同whole-wrapper scopeでpaired compile／postcheck。既存q1を再計算せず参照し、必要な短q価格と未使用holdout、必要long-q域のcoverageを段階的に確認 | 旧compile_one／verify_one、位相保持compiler、state-action、既存cost provider／holdout手法 | q1のCX減を長qへ外挿しない。compile数・追加標本・long-q移送精度は許可scopeで固定。最大q直接compileを自動必須としない |
| **共同接続** | Cの全round測定対応／実装誤差とBの元H向け条件を照合。同じNの可否を判定し、資格のある価格とNだけで資源別Gを会計 | `c1_stage3_accuracy`、`rpe_resource_accounting`、scopeを確認したserial/provider資産 | 全lawの価格coverageや誤差が不足すれば条件付き／未確認として終了。元Hへの最小接続差分は必要になった場合だけ実装 |

Stage Aのq1・32本・128 wrapperを再実行する計画ではない。全round統計に量子shotの明示simulationが不可欠だとも決めていない。既存operator referenceと解析十分条件を優先し、足りない検証だけ追加する。価格母平均のために新samplingが必要な場合は、whole-round law・sample単位・推定精度・selection規則を明示して別scopeへ含める。別endpointの640本、legacy DF12／18-round、固定word02を混合しない。

## 5. 次の実行に必要な判断・許可

| 担当 | 次に決める／行うこと |
|---|---|
| **ユーザー／GPT** | 共通εE・成功確率・claim水準、主資源指標または資源を別々に報告する方針、13-round条件を継続するか、whole-path構成／費用境界を選ぶ。旧CA／failure／配分は履歴であり自動採用しない |
| **ユーザー** | 次の技術実装・人工検証と、科学評価を行うならSOURCE・入力・round／path・cut・誤差評価／compile／sampling・出力・STOPの範囲を明示許可する。今回の文書公開はその実行許可を含まない |
| **Codex（指示後）** | 既存closure・関数・test・pipelineを再利用し、元H state/誤差の意味とwhole-path接続に必要な最小差分だけを実装・検証する。古いguardを外して偽のstate名で通す方法や、新grant／承認証明基盤を作らない |

詳細科学レビューを開始する場合の許可は科学計算許可と別である。今回はレビューを開催・実施しておらず、新しい理論の採択や方式の最終選定もしていない。技術準備の完了を科学的GOとは呼ばない。

将来の重い処理は実空きCPU／RAM等を確認し、独立なB/C評価・path・axisや既存cache／state-actionを適切に並列化する。今回CPU/RAM/workerの人工capは設定していない。過去runの16 worker等を恒常上限へ転用せず、共有設定・共有venv・他job・他利用者のファイルやprocessを変更しない。sub-agent禁止は計算worker並列と区別して維持する。

## 6. STOP・未確認の扱い

| 条件 | 後続scopeでの扱い |
|---|---|
| SOURCE／元H／J/U／sector／law／path順序・位相の不整合 | dependentな新科学callを止め、partialと不一致を保全する。別endpoint・word・cutへ置換して継続しない |
| AB、actual adjoint、vacuum／χ、raw／compiled X/Yのcorrectness不一致 | compile前の失敗ならcompileへ進まない。失敗後の閾値緩和、phase fitting、無断retryでPASSへ変えない |
| 強いsymmetric・既存共有／融合を保つ公平なbaselineを構築できない | 弱い対照へ変更せず不足を報告する |
| 元H向けalias／正半径／phase／target horizonが未資格、または十分条件不通過 | N／成功保証／総費用優位を出さない。不通過と物理的不可能性を区別し、目標・配分変更を自動採用しない |
| 全law誤差・価格support・long-q coverage／scopeが不足 | UNQUALIFIED／未確認を保持。不明を0、有限標本SDを母平均CI、経験的guardを厳密certificateへ変換しない |
| 追加sampling・別cut・精度／条件変更が必要 | 許可範囲内か確認し、範囲外ならユーザーへ戻す。旧one-shot／grant／STOPを新namespaceで迂回しない |
| 指定scope完了またはSTOP | 証拠と不足を保全し、許可された公開・照合後に終了する。次の段階を自動起動しない |

## 7. 今回の公開と終了

追加するのは本3文書と入口への案内だけ。科学SOURCE、input、raw／result、grant／ledger／marker／STOP、VALIDATION_STATUS／科学manifestは変更していない。既存科学結果・研究判断を更新する作業ではないため、過去の研究概要・結論・履歴も書き換えていない。公開文書はrepository相対path・固定commit・科学identityに限定し、内部host/IP・絶対環境path・資格情報・非公開rawを含めない。

UTF-8、相対リンク／固定参照の存在、変更pathと`git diff --check`を確認する。専用branchへcommit/pushし、remote ref SHAと公開Git tree/blob・文書本文の一致を確認して、実測結果を最終チャットで報告する。文書検査を科学検証PASSとは呼ばない。

**公開・照合後mandatory STOP。次担当はユーザー／GPT**：本書の不足と条件を踏まえ、B/Cの採択範囲と次の明示許可を判断する。Codexは新規科学計算・数値再解析・sampling・回路生成・compile・アルゴリズム実装・詳細科学レビューへ進まない。

[A]: ../track_d_h4_paired32_reference_20261106/RESULT.md
[ASOURCE]: ../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/SOURCE.json
[SURVEY]: ../track_d_pr_rpe_static_survey_20261106/REPORT.md
[C32]: ../research_c_near_practical_r32_downstream_execution_20261103.md
