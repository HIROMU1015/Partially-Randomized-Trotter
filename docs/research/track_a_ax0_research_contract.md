# Track A AX-0：拡張研究契約

作成日：2026-10-09。位置付け：独立研究レビュー用の設計文書。**AX-1以降は未認可**。GOは科学的な移行条件であり、実行認可ではない。

## 1. 文書境界と来歴

- 対象：`HIROMU1015/Partially-Randomized-Trotter`、branch `pr2-v4-s2-parallelization-20260928`。作成開始HEAD：`2a80f1d5d5e5734e51d970b2b6822cd2543fd596`。
- 作業worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928`。主checkoutは別branchであり、本研究の作業対象ではない。
- 既存成果の参照点：原稿・図v0.1 `4c23453c541700c6a41ba71fc5ec9323b53858d6`、PM-1 `194cc604b90c56a0e7e949b91b064a4bcfc846da`、PM-2 `5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc`。
- 入力提案：主checkoutの `track_a_extension_detailed_plan_v1_20261008.md`、GPTとの研究方針議論、前回静的レビュー、今回のAX-0作成指示。提案を既存研究の確定仕様や実行許可に読み替えない。
- 読取順：`PROJECT_MAP.md`、[研究概要・現状](研究概要・現状.md)、現行研究文書、`VALIDATION_STATUS.md`、validation manifest、個別契約・source・保存結果、決定履歴。dirty文書とcommitted evidenceを区別する。
- 今回追加するのは本書と[モデル対応](track_a_ax0_model_correspondence.md)、[証拠目録](track_a_ax0_evidence_inventory.md)、[実験仕様](track_a_ax0_benchmark_protocol.md)、[計算予算](track_a_ax0_compute_budget.md)の5件のみ。既存の結果・契約・status・manifest・研究ノート・索引・v0.1・図・Track B・未コミット変更を更新しない。実装したvalidation pathの追加ではない。

本書の「固定」は新しいAX研究内の比較規則を意味する。既存M1〜PM-2の契約を変更しない。AX-0ではJSONのfield/schema/来歴、CSV header、source、一次文献を静的に確認した。科学計算、保存値の新規分析、fit、テストを実施していない。

## 2. 背景と研究課題

既存Track AはH4線形鎖、1.00/1.30 Å、STO-3G、DF rank 12、二次PF、canonical finite-RTE、主にT=0.8における測定込みwrapper資源の局所研究である。M1-A/B1、M2、PM-0/1/2によってdiscard近傍、固定構成移送、要求信号精度依存まで扱った。しかし、サイズ移送、厳しい信号精度、解析モデルの予測時点の情報、強い決定論baselineは十分に検証していない。

**RQ-R**：同じ有限DF Hamiltonian、状態、finite-time complex-signal task、compiler契約、誤差・失敗確率配分において、B0/B1/B2/B3の探索集合内の測定込み資源は精度・サイズ・構造によってどう変わるか。PRが競争的になる範囲と、ならない範囲を同じ手順で特定する。

**RQ-P**：原論文会計、現行実装対応解析モデル、H4校正モデルは、一回のwrapper費用、shot、accuracy eligibility、構成選択をどこまで予測できるか。情報条件を揃え、未使用条件で予測誤差と選択損失を測る。上界の保守性、会計scopeの差、経験予測の誤差を分ける。

二つのRQは独立である。PR優位、モデル不一致、順位逆転は成功条件にしない。モデルが正確なら、安価な予測で選択できる範囲・必要な入力費用・外挿限界が成果となる。

## 3. 先行研究と新規性の境界

一次資料は2026-10-09に確認。原論文の式番号は[arXiv:2503.05647v2](https://arxiv.org/pdf/2503.05647v2)に固定し、[刊行版](https://doi.org/10.1103/ynxb-p2xq)と混同しない。詳細な対応はモデル対応文書を参照する。

| 一次資料 | 既知として扱う内容 | 本研究に残る問い |
|---|---|---|
| Günther et al., [Phase estimation with partially randomized time evolution](https://arxiv.org/abs/2503.05647v2) | partial randomization、DF適用、normalizationと測定費用、化学系・水素鎖のsingle-ancilla energy-estimation資源 | 同じ実装・task・単位へ対応付けた予測検証、discardも含む具体的比較、サイズ移送 |
| Hagan–Wiebe, [Composite Quantum Simulations](https://quantum-journal.org/papers/q-2023-11-14-1181/) | deterministic/random hybridと誤差解析 | channel誤差の保証をcoherent-signal/RTEの保証として移用しない |
| [ROIS, arXiv:2603.13495v1](https://arxiv.org/html/2603.13495v1) | 回路費用と推定量varianceの共同最適化 | cost×shotsという組合せ自体を新規性にしない。固定canonical samplingに限定して検査する |
| [Kanasugi et al., arXiv:2603.22778v2](https://arxiv.org/html/2603.22778v2) | partial randomizationを含む量子化学QPEの具体的資源評価 | qDRIFT等の方式・FT accountingと、本研究のfinite-RTE/native wrapperとの条件差 |
| [SPRINT/GRADE, arXiv:2606.30741v1](https://arxiv.org/html/2606.30741v1) | factorized chemistry Hamiltonianでの構成選択、誤差見積もりとcompiled step費用の組合せ | 単にDF回路をcompileしたという主張は弱い。予測情報分離と凍結移送が必要 |
| Simon–Love, [Halving the Cost of Controlled Time Evolution](https://arxiv.org/html/2511.13855v1) | 対称PFのcontrolled synthesisの改善 | naive controlだけを決定論baselineにすると優位を過大評価しうる |
| [Regularized compressed double factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/) | factorizationと費用・表現誤差の関係 | DF rank変更だけを新アルゴリズムと呼ばない |

独立論文候補は「予測誤差の原因と移送可能な範囲を、精度適格性・選択損失まで含めて同一taskで検証したこと」である。これは予定する寄与であり、新規性が確定したという判定ではない。関連研究との重複はAX-5直前にも再確認する。

## 4. 証拠と主張の階層

| 区分 | 現在言えること／追加条件 |
|---|---|
| 理論上の既知事項 | original RTE/PRの構成・normalization・energy accounting。finite cutoff、別DF representation、native gate setへの変更後も同じ保証が自動成立するわけではない |
| 保存済みTrack A結果 | H4 development 210候補、追加B0 8候補、held-out固定5構成。PM-2は同じ保存bias・回路費用を精度方向に再使用した解析であり、独立した追加実験ではない |
| 狭い資源結論 | 指定候補集合・状態・T・二次PF・wrapper scope内の費用とpoint comparison。family全体の一般的最良性、formal winner、held-outでの再最適化を主張しない |
| 既存移送結果 | M2の`TRANSFER_SUPPORTED`は固定5構成の支持。held-out参照shotを使う条件付きcost移送であり、truth-freeな総費用予測ではない |
| 実装のみの能力 | sector、matrix-free、symbolic tail、GPU helper、repeated PF、sampling/compileが存在。新しいTrack A H6/H8 runnerとしての整合性は未検証 |
| 新しい科学的主張 | H6の初回凍結評価と、改良後のH8独立評価が必要。新しいモデルfitもまだ行っていない |

旧selectorの`SELECTION_LIMITED`、過去の主研究候補のSTOP、M2/PM-1/PM-2のmandatory stopは維持する。selectorのcoverage失敗や未決定な優位を、PR方式そのものの不成立と解釈しない。既存のSTOP理由は証拠目録に記録する。

新研究が検査する事前仮説は、(H1) basis遷移・boundary cancellation・controlによる一回費用の非加法性、(H2) normalizationとbias marginによるshot感度が費用モデルの小さい誤差を増幅すること、(H3) H4校正が別サイズで有効な範囲と失敗する範囲、(H4) 高精度側で強い決定論baselineとdiscardの適格性が選択を変えること。方向・優位は仮定しない。AX-1後、H6の参照結果を読む前に検査可能な形へ凍結する。

## 5. 成果の水準と終了条件

| 水準 | 必要な成果 | 言える主張 |
|---|---|---|
| 最小：technical/resource study | AX-1でscopeを揃えたsaved-only model照合を完了。誤差・選択損失・欠測・情報取得費用を追跡し、既存W_actionとcompiled費用の意味を区別 | 新しい予測監査。ただしH4への事後照合であり、移送性能の実証や独立論文の十分条件ではない |
| 目標 | correctnessと数値予算を通過し、H6でH4凍結モデルを評価。必要な改良を記録し、H8で改良後の凍結予測を独立評価。精度anchor、強いbaseline、探索境界を含む | 登録した水素鎖・task・候補集合における資源の適用範囲と予測信頼性 |
| 上位 | 誤差を構造情報で説明する少数parameterモデル、別geometryまたは別構造での事前予測、予測取得費用を含む選択効用。energy taskは別契約で必要なら接続 | 再利用できる予測・選択原則。水素鎖3サイズだけで化学全般や漸近scalingを主張しない |

独立論文を狙うには、単なる図の追加を超える再現可能な問い、情報漏洩のない未使用評価、誤差原因の検証、妥当なbaseline、境界・欠測・否定的結果の扱いが必要。RQ-Rだけなら「二次PFに対する優位」と「強い決定論法に対する優位」を分ける。RQ-Pは予測が一致しても、不一致でも、連続量の誤差と利用条件を報告して成立する。

AX-1がPM-0/PM-2の既存統計の再掲に留まる場合、それを拡張研究の新成果と数えない。最小の新規成果には、対応を明示した未校正モデルの照合、校正情報の限定、欠測を含む予測coverage、選択への影響の新しい監査が必要である。それも既存監査と重なる場合は対応・再現資料として閉じ、H6/H8を増やすだけで独立論文へ格上げしない。

PRが優位でない場合は、scope内の非優位・discard/高次PFが選ばれる領域を報告する。モデルが一致する場合は、校正の不要な範囲と予測の実用性を報告する。計算予算でH8へ届かない場合はH6までの検証範囲で閉じ、H6改良後の独立外挿確認は未実施とする。新しい情報を得られない候補追加をしない。

## 6. 段階、成果物、GO/STOP

すべての段階は別の実行判断を要する。ここで成果物名を指定してもrunnerの実装・実行を許可しない。

| 段階 | 目的・入力 | 成果物・検証方法 | GO条件／STOP・閉じ方 |
|---|---|---|---|
| AX-0 | 提案、一次文献、source、保存schemaの監査 | 本5文書。リンク・来歴・field・保護差分の静的確認 | GPT独立レビューと未解決事項の担当段階が明確ならAX-1の判断へ。対応不能はN/A。認可の自動継承なし |
| AX-1 | H4保存結果のみ。M0/M1/M2の利用情報を分離 | saved-only coverage表、unfit照合、H4校正と診断、shot/cost分離、欠測、H6前の仮説・予測契約。新しい科学計算なし | 読取allowlist、入力hash、unit/scope、fit規則の事前確認。運用shotモデルがない場合はoracle-cost研究に限定。欠測を再生成しない |
| AX-2 | AX-1結論、実装の静的再利用計画、割当予算 | 認可後にのみ最小backend接続・H4照合・sector/phase/数値誤差検証・小規模profile。H6/H8候補上限と予算を登録 | corrected/raw、identity、state-action、full wrapperの一致と予算成立。不成立ならbackend修正または対象縮小。pilotで見た科学条件はdevelopment扱い |
| AX-3 | 凍結H4モデル、事前登録H6集合・予測、AX-2通過 | H6主検証、誤適格選択・regret・原因診断。初回モデル評価を保存した後、必要なら改良 | 改良の有無にかかわらず結果を保存。H8が解く未解決の問いと予算があればAX-4判断へ。PRの勝敗をGOにしない |
| AX-4 | H6までで凍結したモデル、H8事前予測 | H8独立評価。比較契約、モデル、候補選択、予測fileを参照結果の前に固定 | correctness・情報分離・coverageが保たれる場合のみconfirmatory。H8を見て改良した結果はexploratory。予算不足は限定結論で閉じる |
| AX-5 | 独立評価までの結果、全欠測・failure記録 | claim/evidence map、原稿、再現手順、原論文との差分、計算/量子資源の別表 | 事前RQへの回答と主張範囲が一致すれば終了。最小/目標水準を区別。既存v0.1は別versionとして保存 |

研究上必要なのは、同じtaskでのモデル対応、費用とshotの要因分離、accuracy誤選択、探索境界、強いbaseline、少なくとも一つの本当に未使用な評価である。全geometry×時間×精度×全prefixの直積、全サイズ同じ候補数、H12への拡大、QPEへの即時接続は完成条件にしない。T=3.2と1.30 Åは主張を識別する限定診断としてのみ追加する。

## 7. AX-0完了監査と次の一段階

| 完了条件 | 対応文書 |
|---|---|
| RQ、着地点、否定的結果、終了条件 | 本書2〜6節 |
| 公平に比較できる量、N/A、上界/予測/会計 | モデル対応2〜5節 |
| oracle/operationalの分離 | モデル対応4〜6節、実験仕様6節 |
| AX-1のsaved-only範囲と欠測 | 証拠目録3〜6節 |
| H6/H8情報分離、公平な候補、強いbaseline | 実験仕様2〜6節 |
| 計算制約、未知量、確定方法・blocking範囲 | 計算予算2〜7節、実験仕様7節 |

次に確認するのは本書の独立レビューである。AX-1の実行を別途判断する際、(1) evidence allowlistとhash、(2) M0の利用可能な原論文入力、(3) cost/shot predictorのscopeとoracle表示、(4) H4 fitとgroup診断規則、(5) AX-1上限と出力先を固定する。保存値の再分析・モデルfitは本作業に含めない。

文献対応の未解決事項U01、basis/event入力U02、DF規則U03、探索上限U04、sectorと数値誤差U05/U06、割当資源U07、強いsynthesis U08、operational shot U09は実験仕様・計算予算に確定手順とblocking範囲を記す。未解決事項が残ることを隠して実行可能な研究契約と呼ばない。本5文書はAX-0のレビュー可能な成果であり、実行前の各freezeは後段の成果物である。
