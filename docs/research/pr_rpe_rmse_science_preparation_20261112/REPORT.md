# PR–RPE 実RMSE評価の技術準備（A、2026-11-12）

実科学評価の前に必要な証拠を列挙するmetadata-only preflightを追加した。固定B configurationと固定A scheduleを照合し、保存signal/referenceの所在・欠測・不一致、将来の追加call範囲、未評価N packetを返す。**新しい実signal/N/RMSE、保存値のRMSE再解析、sampling、C/G、κ/M最適化は0。科学的適格性は変更していない。**

A branch `codex/pr-r-dependent-n-model-20261111` を再利用。A固定成果510fbacd01a2d855202de4c045d8ba1972e5ff3dとP1の原本・旧pipelineは不変。B/Cのworktreeを変更していない。作者による技術確認であり、独立科学レビューではない。

## 評価条件と必要な証拠

原典は[PR論文v2](https://arxiv.org/pdf/2503.05647v2) B.2/E.1/E.3、A38/E21等。理論の再証明は行っていない。各軸の経験的配分は `ceil(B²[11+4(M−m)])`、X/Y合計はその2倍。論文の経験的D=4やCmax（PF .08、PR .10）を現在のfinite-Kに対する厳密十分Nと同一視しない。旧αtotal=.05/β=.4,.2のHoeffding N・条件付きPASSをRMSE用Nへ変換しない。

| 項目 | 確認済みSOURCE・再利用 | 実RMSE評価に不足する証拠 |
|---|---|---|
| 対象H/state | 保存Linear H4/H6、隣接1Å、STO-3G、tol-only 1e−8 DF10/DF19、保存数値state | 同sector保存DF-H groundを対象とするbindingと、state/overlap・有効発展モデルの適用契約。小さい保存residualは厳密ground認定ではない |
| 外側設計 | A scheduleのM/h/prior/decoder、Bの全configuration ID | RMSE target、QPE/発展近似budget、合成則、M/κの正式選択。旧13round/h=.1を自動採用しない |
| decoder | `reconstruct_rpe_phase`、raw位相−ET、E=−φ/h | priorのalias-free性・branch tie/外れ値・全round誤推定を含むenergy RMSE。E_stateでbranchを選ばない |
| QPE統計誤差 | 論文経験式とA nominal評価器 | 固定N配分・decoder・実meanに対する統計誤差の適用証拠。horizon診断だけでは保証にならない |
| 発展誤差 | PF/finite-K mean、exact-tail内部対照、bias/phase/radius/closureの保存値 | stateがPF有効固有状態とは限らず、finite-K mapも一般に単一の厳密unitaryではない。局所diskから全energy RMSEへの移送が未確立 |
| 誤差合成 | Aのpaper_quadrature/minkowskiを別identityで保持 | 論文の二乗和を使う仮定（有効energyに対する統計推定等）または別の正当化。予算の算術整合性だけでは適用資格にならない |
| normalization | finite K2の既存分布・B/logB、独立raw/corrected、scalar一回会計 | finite B plug-inの経験的配分への校正、数値allowanceとrigorous boundの区別、actual sampling/implementation誤差 |
| κ/event | A/Bのpaper回転数κ、整数outerへのdraft mapper、元allocation | finite-K eventと論文回転数の対応資格。eventをcompiled RZとみなさず、κ_Bへ置換しない |
| H6 signed構成 | m5_bestの負時間を含む11 piece、ABS_TIME_PER_OUTER、qPF×Rの全event数 | piece別整数丸め・signed normが論文の単純exp(1/κ) envelopeを満たすとは未確認。係数やκを黙って補正しない |
| 全round | 全candidate identity、H/state/snapshot、reference/mean/誤差診断の所在 | 全0..Mの統一policy証拠、state/target/prior/decoder/finite-Kを含むRMSE bound。局所pilotや旧13round PASSで代用しない |

K2の一eventは有限次数の積を含み、次数2を一律3個の独立なpaper回転と数えない。DF fragmentのbasis-conjugated involutionと単純Pauli回転の価格差はC側の仕事。Aの回数は `r*=κ λ_R²T² → ceil(r*) → max(J,ceil(ceil(r*)/qPF))` という構造案であり、finite-K/signed-pieceの理論保証ではない。

## 保存成果の再利用範囲

[SAVED_INVENTORY.json](SAVED_INVENTORY.json)は8個の固定carrierについて、候補・binding・SOURCE・field有無だけを記録する（70 mean descriptor、26 reference descriptor、重複出所を含む）。signal/N/phase/logB等の数値はコピー・再集計しない。原signal carrierはexact candidateを既存binding付き結果へjoinし、旧N/marginは選択しない。全carrierのcommit/path/内容hashと基準taskのSOURCEを保持する。

| 保存SOURCE | そのまま所在を特定できるもの | 今回行わない追加処理 |
|---|---|---|
| A P1 `71ba49bb…`、科学SOURCE `a1a79213…` | PR 12候補：H4 m0 R2/4/8・m6 R4/128/256、H6 m4 R11/16/32・m8 R16/128/256。mean・normalization・診断、H6 T1.6の独立reference pair | 同一候補でもRMSE policyでの再解析は新科学実行。P1の旧Nは利用しない |
| H4 `89c91c69…` | 固定B0/p7とPR p1/R4/K2、全13roundの所在。m3は薄い結果表、m4は保存signal利用 | m3は元carrierへjoin。古いexact-tail norm診断・reference pairの欠測をゼロで埋めない |
| H6 `f4c0bdab…` / D2 `95d01b6a…` / D1 `45607a50…` | 固定Full19/PR、m0..12。m0..3のPR p5/R32/q1/δ=T、m≥4のp7/R16/δ1.6 | m3/4の元carrierはsignalを持つが、薄い診断だけで全数値gateを再認定しない |
| 原carrier `155529e2…/signal_records.json.gz` | H4 m3、H6 D1等のreference/raw/corrected/exact-tail/logBのfield。元結果とcandidate完全一致を確認 | raw=corrected/Bの再構成、allowanceの新計算、旧alpha_axis=.025の再利用はしない |

独立2経路referenceの**field coverage**はH4でT=.1,.2,.4,3.2,…,409.6、H6でT=.1,.2,1.6,3.2,…,409.6。H4 T=.8/1.6とH6 T=.4/.8は今回選択した保存carrierにpair fieldが揃っていない。これは「referenceが存在しない」という主張ではなく、scalar/referenceの所在と独立照合証拠の不足を区別したもの。保存差分の数値gateは再評価していない。

H4 S2/δ=.1/PR p1/K2、H6 m5_best/δ1.6/PR p7/K2、H/state/snapshot・signed order/scalar/lawを保持する。新κのcandidateが保存IDと一致しなければ、そのmeanを流用しない。同一H/state/Tならreferenceの再利用候補はより広いが、独立pair gateの再確認も承認後の科学範囲に含める。

## 実装とB最終版への接続

新module `pr_rpe_rmse_preflight.py` とCLI `preflight_pr_rpe_rmse.py` を追加した。Bの固定82bc774a92b65cac06b88ebcb2c6979c2726253bから**変更しないmetadata関数ASTだけ**を読み、generate/_validate/exact scalarを再利用する。Bのassess/cost/select・controller・価格providerを複製・変更しない。

preflightは供給されたB configとA scheduleだけを検証し、実候補やκ/Mを新たに選ばない。A evaluateはprovider=Noneによるhash/recipe検証だけで、nominal_N計算も起動しない。全round、候補、SOURCE/H/state/profile、時間のexact表現、κ/λ SOURCE、law SOURCE、11 piece allocationを照合。異なるpinでもlaw sourceの同じGit blobを確認できる場合は同一SOURCEとして扱い、無関係なworktree HEAD/statusを条件にしない。

B finalとA旧草案packetの差は次の通り。

- B IDは`RMSE:`付きの全configuration依存ID。A physical IDと同一ではない。preflightは両方を残し、Bのidentity/law_bindingを変更せずpacketへcopyする。
- B finalはconfiguration_id/policy/source/status/evidence_kind/全roundと、全scheduleの`rmse_evidence`を要求する。今回のpacketは`NOT_EVALUATED`、`PREFLIGHT_ONLY`、N_X/Y/nominal_N/boundはnull。Bの科学N受理条件を意図的に満たさず、costへ投入する結果ではない。
- BのPF baselineはfull-H。H4の旧B0/p7をPF Full10へrelabelしない。不一致はBLOCKED。PF比較条件の採択は別途必要。
- B finalは全round固定L_D。H6 m0..3のp5/R32/q1/δ=Tという保存例外を、通常p7のconfigへ表現できない。外部event mapperだけではprefixを変えられない。Aは元earlyを保持して不一致を返す。Bが表現方法を整理するまで、whole-H6 RMSE packetは適格化できない。
- Bのceil(T/δpf)とAの固定δ整数gridは一般には異なる。丸めを無視せず、T/δをexact照合する。prior/decoder/QPE予算とB error_contractの科学的joinも未資格のまま明記する。

各roundは保存mean/referenceの再利用候補と不足fieldを返す。保存FAIL/STOP/数値未解決・SOURCE不一致は自動retryせず、call案もnull。不足時はまず保存診断回収を優先し、回収不能なら最大1 mean＋PR内部対照1、pair不足なら最大2 reference actionという**将来承認用の上限案**を返す。これは実行命令でも、必要最小call数の科学的確定でもない。

## 再利用する関数と科学leafの境界

| 既存module/関数 | 今回・将来の利用範囲 |
|---|---|
| `pf_pr_rpe_pipeline.candidate_id/structure/identity`、`h46_m5_plan.weights/allocation` | 今回は純metadataとして再利用。DF12の数値係数や旧Nは移植しない |
| `pr_rpe_rmse.schedule/evaluate` | 今回は供給scheduleをevaluate(provider=None)で検証。新しい実schedule/nominal Nは生成しない |
| `h6_df19_rpe_d2.evolve/half_step/reference_pair` | 将来の有限mean・独立raw/corrected・exact-tail・2経路reference leaf。保存H/state context前提、新SCF/DF/eigensolveなし。今回call0 |
| `h4_df10_rpe_batch_provider.evolve_h4` | H4 B0 scalar規則の将来route。H6処理へ無条件に付け替えない。今回call0 |
| `rte.finite_rte_distribution/finite_taylor_operator`、`pr_r_dependent_n.check_normalizers` | 旧normalization/finite-K実装を維持。小τでnormalizerが丸めで1以下になる等の未解決をB=1として通さない。今回実値評価0 |
| `pr_r_detailed_n_p1.checked_reference` | 将来の同一H/state/T、独立pair数値gate。今回field有無だけで、数値gate call0 |
| `c1_stage3_accuracy._assess_raw_signal`、H4/H6 `analyze/assess_supplied_round` | 既存phase/radius geometryの所在を確認。ただし旧Hoeffding Nまで計算するwrapperを新RMSE solverとして丸ごと呼ばない。将来は既存式の局所診断部分とRMSE policyを分離する必要がある |
| `rpe_four_round_phase_validation.reconstruct_rpe_phase` / A `decode_supplied_coordinates` | decoderを再利用可能。今回は既存人工回帰のみ。実全round統計・simulationは未実行 |

## テストと最小の次pilot案

[TESTS.txt](TESTS.txt)：追加14人工/mock/保存schemaテスト、既存RMSE17、旧高確率型R依存12の**計43 PASS**。欠測/null component、未資格N、全identity改変、κ/λ/law SOURCE、保存FAIL/数値未解決、H4 B0/H6 early不一致、負piece、旧N混入拒否、B固定fixtureの受動検証を確認した。numpy/scipy/qiskitをimportせず保存inventoryを作れること、preflightから数値leafを呼ばないことも検査。[ARTIFICIAL_IO.json](ARTIFICIAL_IO.json)はMOCK条件の全I/O。既存人工matrix/decoder回帰では既存docstring警告のみ発生し、共有環境変更はない。

次のpilotは自動採用しない。**ユーザー**が1系、1組(M,κ)、外側H/state・split・δ・law、RMSE target/予算/normalization ruleを固定し、保存再解析と欠測計算の範囲を承認する必要がある。

| 参考pilot | 再利用の利点 | 採択前の条件 |
|---|---|---|
| H4 PR m0/m6 | P1 mean/誤差診断と独立reference pairがあり、新signalなしの局所RMSE診断にできる可能性 | 共通κから出る両候補IDがP1に一致することを確認。H4 PF baseline問題とは分離し、PRだけのpilotを全PF/PR比較と呼ばない |
| H6 PR m4/m8 | P1 mean/診断、追加済T1.6 reference pairを利用可能 | 同一κ候補一致。局所regular roundは検証できても、B whole scheduleのearly例外問題は解決しない |

承認案の最小範囲は**2roundの保存mean/reference数値gate再確認＋finite B/bias/phase/radiusと経験的RMSE型nominal Nの診断（X/Y）**。欠測候補だけ新mean/内部対照各最大2、必要なpairだけ2経路を追加する。既存独立pairが一致すれば新reference0。R/κ探索、C/G、P2/P3、decoder sampling、全RMSE boundの認定は含めない。このpilotが判定できるのは候補一致・局所finite-K/normalization・経験式の数値的利用可否まで。

全RMSEを判断する次段階では、固定0..M全mean、state/target/alias/decoder・finite-K移送の証拠と、採用した統計評価法が必要。decoderの統計校正を行うなら、law・試行数・seed・誤差/信頼区間・実行上限を別途固定して承認を得る。二時点pilotから全energy RMSEや総費用順位を認定しない。

**Codex B**へは[HANDOFF.md](HANDOFF.md)を渡し、最終controllerの統合をBに委ねる。**Codex A**は明示承認後だけ上記限定科学scopeを実行する。今回はcommit/push・remote SHA/主要成果物照合後、**mandatory STOP**する。
