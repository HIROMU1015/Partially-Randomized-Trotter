# C-α／α0-P：macrostep・gap保証改善と到達範囲

2026-11-10 JST。担当：Codex。

**結論：指定レビューのC_multi／一macrostep defect上界ではm17の0.14 rad条件は成立しない（分類B）。追加の状態依存評価を精密化すると、保存binary64 operandによる再構成モデルでは二点の条件が包含付きで成立した。一方、固定sampling lawそのものへの保証は未確定（分類D）。全18round・shot・総費用へは進まず、成果保全後mandatory STOP。**

これは「固定α0-Pの実信号に問題がある」という負結果ではない。旧信号は再simulationしていない。新たに取得したのは誤差保証の入力・評価と、その資格である。再構成モデルで得たm17の位相上界候補は **0.1294225186 rad**、半径下界候補は **0.1102381355**。この数値を、そのまま固定primitive／native実装の保証へ昇格させない。

## 1. 今回の許可・固定条件・SOURCE

今回の `.agent/docs/comment.md` §2の明示的な実行許可により、追加承認を挟まず必要な実装・人工確認・科学評価を進めた。旧run01／run02をretry／resumeしたものではなく、それらのSOURCE・raw・log・REPORT・STOPは不変更。

| 項目 | 固定条件 |
|---|---|
| 系 | Linear H4、1.00 Å、STO-3G |
| target | 元H_phys。同じ元H保存state、厳密基底状態の研究promise、prep費用除外 |
| DF・split | 保存DF12、同じgeneric frame、generation-prefix LD3 |
| sector | Nα=Nβ=2、sector36 |
| law・時間 | 正時間partial-S2、α0-P／paired K0、δ=.02、r4、θ=.005 |
| 初期評価点 | m0：q1／T=.02、m17：q131072／T=2621.44 Ha^-1 |
| 新C-α予算 | systematic .14 rad、stat .24、reserve .02、全体 .4。旧P1の個別枠は不変更 |
| branch・基点 | `codex/track-a-calpha-h4-step1-20261110`、`0d5ca357ccfd8fcc850bef800c5bcadfde2a153e` |
| 作業場所 | 既存 `.worktrees/track-a-calpha-h4-step1-20261110` を継続利用 |

直接の科学入力は[最新GPT限定レビュー](reviews/track_a_calpha_macrostep_gap_limited_scientific_review_20261110.md)。原本SHA256 `175cf13d23e4d6290fe0ba6a10b76f05598b48f698434fcf923e0ee4abf0bf29`。全463行を確認し、§3–6・§8–9を実装へ接続した。[元理論レビュー](reviews/track_a_calpha_theoretical_review_20261110.md)と[sector追補](reviews/track_a_calpha_sector_condition_review_20261110.md)も原内容を保存した。元付録の符号訂正は最新レビュー§3.1に従い、原本を上書きしない。外部文献の再監査・新規性の再レビューは行っていない。

`research_project_instructions_unified_2026-11-09.md` は確認したrepository・worktree・添付では未取得。GPTによる読了記載をCodex自身の読了と扱わず、提示されたAGENTS.md・comment.md・取得したレビューに従った。

## 2. 再利用・追加実装・保全

既存の `b2_p1_signal_validation.load_private_inputs`、`compose_step`、`b2_p1_remaining_rounds.checked_prior`、generic diagonalizer、CAR作用、vectorized exterior-minor liftを再利用した。既存のPF／交換子関連コードを検索したが、対象が別law・別研究条件へ結合されており、そのrunnerを流用しなかった。新規の処理は今回の保証入力に限定した。

- [macrostep／正確なLDL helper](../../../src/trotterlib/calpha_macrostep_bounds.py)：C_multi、η_macroの数値診断、実行列の正確な有理数慣性。
- [区間helper](../../../src/trotterlib/calpha_interval_bounds.py)：mpmathの基本区間演算、打切り残差を明示した指数・tail log、状態defectの包含。
- [初期上界・gap runner](../../../scripts/run_calpha_macrostep_gap.py)、[状態defect runner](../../../scripts/run_calpha_state_defect.py)、[参照包含runner](../../../scripts/run_calpha_interval_state_bound.py)、[再構成接続runner](../../../scripts/run_calpha_reconstruction_bridge.py)。旧runnerは変更していない。

run01の4ファイルは従来の[保全archive](../../../artifacts/calpha_h4_step1/2026-11-10/preserved_run01.tar)とbyte一致。旧run02の[raw](../../../artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected/result.json)、[REPORT](RUN02_RESULT.md)、SOURCEも不変更。旧run02 raw SHA256は `8ff3e10d6fb663714b68aa0e9b2affa0816bc04d92c509eede7dca8791fbfffb` のまま。旧46件のatom診断を再実行・改変せず、個別atomのsector保存を再び要求していない。

元H・state・DF・sector identityは既存loaderで照合し、旧run02と同じものを使用した。private state、積分、HDF5、再利用用の数値operand NPZは公開対象外。既存stateの再solve、入力stateの再正規化・phase変更、frame／law／係数／確率／λ／identityの設計変更はない。解析中に使う正規化snapshotは誤差を記帳する数学的参照であり、原入力・raw信号を変更していない。

## 3. Stage orderと旧台帳からの置換

既存append順は `one_body → fragment0 → fragment1 → fragment2 → tail → fragment2 → fragment1 → fragment0 → one_body`。行列の左右の表示順を区別し、中央から外側の順は **fragment2、fragment1、fragment0、one_body** とした。各headは正時間δ/2、tailは `(M_P(δ/r))^r`。scalarは一回だけ付ける。

`B=−iRθ−Γθ`、`G_0=B`、中央から外側へ `G_j=G_{j−1}+A_j` としてC_multiを取得した。これはhead内部とhead/tail splitを含む。旧 `q(η_forward+η_reverse)` と旧 `Tδ²C_split` は**両方置換**し、再加算していない。model、arctan、full-column接続、raw表現差はそれぞれ一回。

人工テストで新factorizationと既存 `compose_step` の一致を非可換例で確認した。個別Gaussian gateや `J†Q_lJ` の指数化は利用していない。

## 4. 指定レビューの上界：大きな改善だがm17は不成立

[run03 raw JSON](../../../artifacts/calpha_h4_step1/2026-11-10/run03_macrostep_gap/result.json)。以下の交換子・defect normはbinary64の数値診断であり、認証値ではない。

| 共通量 | 数値診断 |
|---|---:|
| C_multi | **0.09495117730737149** |
| η_macro | **4.992045884504641e-7** |
| δ³ C_multi | 7.596094184589721e-7 |
| C_multi内のfragment2項 | 0.00172424097434 |
| fragment1項 | 0.00176596291082 |
| fragment0項 | 0.00647821012109 |
| one_body項 | **0.0849827633011** |

| 新しい台帳の数値診断 | m0 | m17 |
|---|---:|---:|
| Tδ² C_multi | 7.59609418e-7 | **0.09956352570** |
| q η_macro | 4.99204588e-7 | **0.06543174382** |
| 選択したPF項（上記min、診断のみ） | 4.99204588e-7 | 0.06543174382 |
| 旧run02のarctan項 | 2.23572240e-8 | 0.002930406067 |
| gap非依存のvariance項 | 5.14640862e-12 | 0.08841462679 |
| 資格付き行列gapを使うvariance診断 | 5.14640862e-12 | 0.0005799333245 |
| gap使用のd（他項は旧診断値） | 5.21566967e-7 | 0.06894208426 |
| gap使用のa−d | 0.999983709145 | 0.05763187825 |
| gap使用のarcsin位相値 (rad) | 5.21575192e-7 | **0.5760054111** |
| 0.14 rad条件 | 数値値は成立、未認証 | **不成立** |

旧head項10.74909696は除去できたが、新C_multiが旧split定数より小さいわけではなかった。one_bodyを外側へ包む項が主に支配し、η_macro経路でも位相予算に届かない。新上界の採用だけでGOとしない。

## 5. Gapの資格・resolvent診断

元Hの保存energyと保存gapは、固有値区間の**探索seed**としてのみ使用した。半幅2^-36 Haの区間に対し、同じ36次元 `H_ref=(H_phys+H_phys†)/2` の保存binary64成分を**正確なdyadic数**として解釈し、`H_ref−shift I` のLDL慣性をFraction演算で取得した。

4境界で負固有値数は **0、1、1、2**。したがって、低い二固有値の漏れのない包含として、

`Δ_- = 1047673349999319 / 4503599627370496 Ha`

を得た。下向きに丸めた表示は **0.2326302150910828 Ha**。保存診断値を改名したものではなく、指定された明示的な有限行列に対する新しい数値資格である。原始積分の物理的不確かさや、別のH／sectorのgapへ拡張しない。新しい基底stateのsolveは行っていない。

resolventも同じstate／H_refで診断した。`||g||=0.000160412103910`、`||w||=0.000112964590452`、m17のvariance診断は **6.75761799e-5**。ただし、数値SVD／solve残差だけを可逆性・逆作用素normの認証にしなかった。指定レビューのPF項が支配している段階では、この改善だけでも0.14 radに届かない。後続の包含には、完成済みのgap経路を使用し、resolventの形式的な認証完了を目的化していない。

## 6. 状態defectの精密化と参照モデルの包含

最新レビュー式(3.6)の状態defect和を取得した。[run04 raw](../../../artifacts/calpha_h4_step1/2026-11-10/run04_state_defect/result.json)。m17の和は **0.0178249927940**、eigenmode三角上界の数値診断は0.0178292388810。arctan補正を加える前からdisk予算にほぼ達し、無重みの(3.6)でも不足した。ここで評価した中間vectorはdefect normの入力であり、新しいRPE round信号ではない。

さらに、同じtelescoping恒等式に含まれる左側の収縮を残した。

`F^q−E^q = Σ_(k=0)^(q−1) F^(q−1−k) (F−E) E^k`

`||F^q−E^q|0⟩|| ≤ Σ_k exp(−μδ(q−1−k)) ||(F−E)E^k|0⟩||`

ここで `Γθ ≥ μI > 0` を同じtailのスペクトル包含で確認し、各完成headがunitaryな**CAR参照モデル**では `||F^j||≤exp(−μδj)` を使用できる。これはPF familyやlawの変更ではなく、既存telescoping評価で捨てていた既知の収縮を利用する精密化である。最新レビューが明記したC_multi／η_macroの数値を、それだけで保証へ変更したものではない。

[run06 raw](../../../artifacts/calpha_h4_step1/2026-11-10/run06_reference_interval/result.json)では、以下を実施した。

- 40桁の基本区間演算。binary64値を正確なdyadic inputとして導入。
- Rθはarctan級数、Γθはlog級数で包含。級数の打切り残差を行列normで加算。
- 行列指数は32次Taylorと `exp(||A||)||A||^33/33!` の明示的残差。mpmathの点評価用 `expm` の停止判定を認証根拠にしない。
- 近似eigenpairを正確なeigenpairと見なさず、generatorの固有残差と係数復元残差をDuhamel評価へ入れる。geometric sumを囲み、131072個の区間時間発展を展開しない。
- 保存snapshotのnorm、H_ref残差、認証済み低固有値区間から、真の基底rayとの差を囲む。入力stateは不変更。ray距離上界は約1.20649e-14。

収縮率下界は **μ≥0.00019538994057824506**。γ上界の表示は0.0007884706336364658、vΓ上界は2.573204308077935e-8。すべての正確な包含端点はrawの有理数文字列に保持する。

## 7. Generic再構成への接続を含む包含結果と、その限界

[run07 raw](../../../artifacts/calpha_h4_step1/2026-11-10/run07_reconstruction_bridge/result.json)。同じgeneric eigensolver／exterior lift／保存係数projectionで得た**binary64 basis・diagonal operandをliteralな数として固定**し、その行列関数・積を囲んだ。CAR参照へのfull-column差、model差、raw表現差を包含した。

full-column処理は、全Fock空間のtotal-N=4 blockを疎な表現として利用した。元sector36の埋め込みから外側への34成分も保持し、それ以外のtotal-N blockとの非結合を構造・arrayで確認した。これはsector70のstate・gap・信号を評価する研究への変更ではない。個別trajectoryをsectorへ射影していない。

| 再構成モデルの包含量 | 上端表示 |
|---|---:|
| tail辞書和とCAR参照の差 | 1.28795538e-15 |
| full macrostep norm | **0.9999960922088783 < 1** |
| full-column接続差 e_conn | **2.28144225e-14** |
| model差 | 2.26524612e-14 |
| raw／Hermitian参照差 | 1.51051289e-15 |

| 包含付き評価（再構成モデルに限定） | m0 | m17 |
|---|---:|---:|
| 重み付き状態PF項 | 3.21896690e-7 | **0.01282547307** |
| state表現差 | 2.42448327e-14 | 2.42448327e-14 |
| arctan項 | 2.23573119e-8 | **0.002930417584** |
| gap variance項 | 5.14640862e-12 | **0.0005799333245** |
| model項 | 4.53049224e-16 | 5.93820678e-11 |
| full-column接続項 | 2.28144225e-14 | 2.99033198e-9 |
| raw表現項 | 3.02102579e-17 | 3.95971892e-12 |
| **d上端** | **3.44259196e-7** | **0.01633582704** |
| 参照半径a下端 | 0.999984230712 | 0.126573962505 |
| a sin(.14)下端 | 0.139540914149 | 0.017662524961 |
| **半径下界候補** | **0.999983886452** | **0.110238135469** |
| **位相上界候補 (rad)** | **3.44264625e-7** | **0.129422518552** |
| 再構成モデル内の正半径・.14条件 | 成立 | 成立 |
| 固定sampling lawそのものへの保証 | **未確定** | **未確定** |
| finite-bit／native込み保証 | **UNKNOWN** | **UNKNOWN** |

**資格の境界**：この包含は、丸められたbasis・diagonal operandを固定し、そこから数学的な行列関数を評価する再構成モデルに関するもの。丸められたbasis liftを、そのまま正確なunitaryや各atomの正確なinvolutionと認定していない。`U diag U†` と実際の固定Gaussian primitive列の意味の対応まで、この計算では認証していない。full macrostepが収縮であることを囲めても、固定primitive lawとの完全な一致がそれだけで証明されるわけではない。

したがって、再構成モデルの成功を**固定α0-Pの包括的なA判定へ格上げしない**。この差を0と置いたり、旧engineering allowanceを無断で新保証の証明に転用したりしない。個別atomのsector保存が必要という旧条件へ戻すものでもない。

重み付き精密化の正式な科学的採択、および上のモデルと固定lawの対応の扱いは、Codexが代行しない研究判断として提示する。既に採用されたレビューのC_multi／η_macro経路はB、新たな包含候補の固定lawへの適用はDと区別する。

## 8. RPE会計・費用との接続状況

**全18round・shot・価格・総費用の新規評価は0。** comment.md §8の前提である「m0／m17の必要な資格を満たした固定law保証」を、上記の候補だけで成立済みと扱えないため、後続実行gateは閉じたままとした。未知項を0、未取得価格を0や単純Q倍で補っていない。

既存 `b2_p1_shot_accounting.py`、`account_radius_round`、RPE配分・branch入口を静的に確認した。旧driverはK2、旧P1の成分枠、旧参照半径・補正diskを固定しており、新C-αのraw半径保証をそのまま渡すとnormalizationを二重適用する危険がある。再利用するなら、既存Hoeffdingとbranch幾何を用い、raw半径とsystematic合算枠を直接つなぐ限定adapterが必要。今回はgate前なのでshot callを行っていない。

| 比較対象 | 今回の到達範囲 |
|---|---|
| 既存K2 finite-RTE PR | 保存m0/m17と既存P1会計を参照。再simulation・再会計なし |
| 同じpaired K0の既知PR | 同じlaw／設定なら信号・物理回路価格はα0-Pと同じ。新しいアルゴリズム差とは扱わない |
| C-αの保証を適用したα0-P | 今回は保証構成と資格の改善候補。最終shot／費用改善は未評価 |
| 強Full PF | 同じ元H/state/sector36の既存記録がある。旧数値STOPと後続高精度記録を混同せず、再実行なし |
| Truncated PF | 元targetとの差、同じ精度・成功条件・compilerを揃えた価格の採用資格を今回取得していない |

保存full no-prep価格記録 `origin/codex/track-a-b2-p1-full-no-prep-cost-20261107:docs/research/track_a_b2_p1_compiled_cost_20261107/REPORT.md` も確認した。これは固定K2・代表m0 trajectoryのX/Y compileであり、期待価格・全18round価格ではない。既存の長回路資源問題・compiler境界の取消を、短回路価格のQ倍で回避しない。今回は不足価格を取得する段階へ進んでいない。

既知K0に対するC-α固有の寄与は、保証・会計・それを使う設計選択として評価すべきもの。同じ設定の既知K0と物理的な価格が変わったという主張や、総費用勝利はない。

## 9. 技術確認・資源・実行履歴

人工確認は[macrostep／LDL 8件](../../../tests/test_calpha_macrostep_bounds.py)と[区間処理5件](../../../tests/test_calpha_interval_bounds.py)がPASS。stage order、局所3次、可換例、正確な固有値count、不適格行列の拒否、explicit Taylor remainder、scalar tail log、複素共役、重み付きscalar geometric sumを確認した。旧科学runや旧科学test全体を再実行していない。

区間処理の初回run05はmpmath 1.3.0の `iv.mpc.conjugate` で例外となった。[失敗記録](../../../artifacts/calpha_h4_step1/2026-11-10/run05_reference_interval/result.json)と当時のSOURCEを同directoryの `executed_source/` へ保全し、real／imagの基本区間演算による共役へ修正した。関連人工確認後のrun06は成功。これは今回scope内の技術修正で、旧run01／run02のretryではない。共有mpmath packageは変更していない。

| 出力 | 内容 | 記録時間 |
|---|---|---:|
| run03_macrostep_gap | C_multi・η_macro・正確なgap・resolvent診断 | 1.522秒 |
| run04_state_defect | 式(3.6)の状態defect和、再利用用private operand | 1.289秒 |
| run05_reference_interval | 共役処理の技術例外。失敗SOURCEも保存 | raw参照 |
| run06_reference_interval | CAR参照の状態上界・moment・state表現の包含 | 48.979秒 |
| run07_reconstruction_bridge | full-column接続・model／raw差の包含 | 65.013秒 |

実行前はCPU affinity32、RAM約62 GiB／available約53 GiB。小さいdense行列の点評価はBLAS／OpenMP各1thread、独立な13 frameの重い区間演算は**13 processで並列化**した。プロセス数は独立frame数と利用可能affinityから選び、BLAS過剰並列を避けた。GPUは使用せず、共有設定・venv・他ユーザーjobは不変更。速度倍率やGPUの利用可能性は主張していない。

各結果に実行SOURCE SHA256・不変確認・入力identityを保持。新raw信号・減衰信号・ground state solve・別分子・r/LD探索・sampling・回路build/compile・価格MCは行っていない。旧run01／run02のbyte保全、全実行SOURCE版、JSON／リンク／diffを最終確認済み。成果の公開対象は研究SOURCE・REPORT・非機密JSON・log・人工testのみ。

実行履歴の共通prefixは `env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/calpha-macrostep-mpl PYTHONPATH=src '/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python' -u`。専用worktreeで、次のprogram／引数を使用した。

```text
scripts/run_calpha_macrostep_gap.py --private-dir artifacts/calpha_h4_step1/2026-11-10/input_private --review '/home/abe/Project/Partially Randomized Trotter/track_a_calpha_macrostep_gap_limited_scientific_review_20261110.md' --output artifacts/calpha_h4_step1/2026-11-10/run03_macrostep_gap
scripts/run_calpha_state_defect.py --private-dir artifacts/calpha_h4_step1/2026-11-10/input_private --operands artifacts/calpha_h4_step1/2026-11-10/input_private/state_defect_operands.npz --output artifacts/calpha_h4_step1/2026-11-10/run04_state_defect
scripts/run_calpha_interval_state_bound.py --operands artifacts/calpha_h4_step1/2026-11-10/input_private/state_defect_operands.npz --output artifacts/calpha_h4_step1/2026-11-10/run05_reference_interval
scripts/run_calpha_interval_state_bound.py --operands artifacts/calpha_h4_step1/2026-11-10/input_private/state_defect_operands.npz --output artifacts/calpha_h4_step1/2026-11-10/run06_reference_interval
scripts/run_calpha_reconstruction_bridge.py --private-dir artifacts/calpha_h4_step1/2026-11-10/input_private --output artifacts/calpha_h4_step1/2026-11-10/run07_reconstruction_bridge
```

run05の実行版は保存した `executed_source/` に対応し、現runnerは修正後のrun06版。既存結果directoryは各runnerが上書きを拒否する。

## 10. 次の判断とSTOP

今回得た研究上の結論は、**新macrostep norm上界とgapの改善だけでは固定m17は閉じないが、収縮を利用した状態defect評価では、再構成モデルの十分条件を数値包含付きで閉じる具体的な経路がある**というもの。固定lawとモデルの対応・精密化の採択を未確定のまま、費用会計へ進む根拠にはしない。

**ユーザー／GPT**：§6–7の重み付き評価とその証拠水準を確認し、再構成モデルから固定ideal lawへ必要な対応条件をどのように研究保証へ組み込むか判断する。これが重要な数理・対象の採択判断であり、単なる実装一区切りによる追加承認の要求ではない。

**Codex**：判断後に、同じlawとの対応の資格を確定できる場合だけ、既存のraw半径会計へ接続する。今回の広い許可を段階ごとに求め直す意図はないが、未確定の対象対応や新しい保証の正式採択を代行しない。全18round／shot／総費用は未実施の次候補として残す。

今回のSOURCE・結果・失敗履歴・旧STOPを保存し、comment.md §11に従い**mandatory STOP**。
