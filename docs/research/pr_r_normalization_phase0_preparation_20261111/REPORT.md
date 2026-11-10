# 代表24候補のnormalization評価準備（2026-11-11 JST）

固定Phase 0の24候補について、保存tail norm・入力binding・SOURCEのpreflight、承認後のnormalization専用入口、出力形式とB/C HANDOFFを完成した。**今回は技術準備と人工/mockだけで、実H4/H6の新log B・attenuation・shot係数は計算していない。** 新signal、phase/radius、十分N、C/G、R選択・候補追加も0。公開・remote照合後mandatory STOP。

既存A branch/worktree `codex/pr-r-dependent-n-model-20261111` を再利用した。B/Cのworktree・process・環境を変更していない。実装SOURCEは `6ed8c6873311085bb3b39799b268105b0bc97211`。新module/runner/test以外の科学codeは変更していない。

## 固定対象と入力対応

対象は`f98982118324bcc8402ed513b5c720b37bb03e87`のEXECUTION_PLAN Phase 0と同一。候補IDとlaw/requestは同commitのmetadataからコピーし、Bのcanonical ID・allocation・request_contextで照合した。[実行PLAN](PLAN.json)に24候補の全identity/law_bindingを固定した。

| 系・round | T | qPF / qRPE | R_outer |
|---|---:|---|---|
| H4 m0 | .1 | 1 / 1 | 2,4,8 |
| H4 m6 | 6.4 | 64 / 64 | 4,128,256,512 |
| H4 m12 | 409.6 | 4096 / 4096 | 4,8,8192,16384,32768 |
| H6 m4 | 1.6 | 1 / 16 | 11,16,32 |
| H6 m8 | 25.6 | 16 / 256 | 16,128,256,512 |
| H6 m12 | 409.6 | 256 / 4096 | 16,32,2048,4096,8192 |

H4はS2/δ=.1/prefix1/K2、H6は4th(m5_best)/δ=1.6/prefix7/K2。H6は11 signed pieceの順・値・負時間、ABS_TIME_PER_OUTER配分、Σr_j=R、r_j≥1を維持する。event数はqPF×Rであり、qRPEとは別field。12候補ずつ、計24候補・piece slotは12+12×11=144。

保存DF10/DF19、Linear H4/H6・1Å/STO-3G、weighted completion、fragment順、sector/state/H/snapshot binding、RPE精度・α/β・prior・decoder、compiler/control/full measurement wrapper/P=0は凍結taskのまま。normalization計算はstateに依存しないが、N/Cと同じidentityを保持する。bindingに残る歴史的T=.8は保存入力の履歴で、今回のTはcandidate.Tである。

保存SOURCE索引は `7a8bc7f132e166390c943e6b32c92cc6abe1baf4:artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json`。既知の入力だけを読み、H4 receipt/snapshot/contextの3ファイル、H6 DF receipt/snapshot receipt/snapshot/contextの4ファイルをhash照合した。Hamiltonian/state hash・rank・sector・context bindingも保存receiptと照合した。NPZ配列はロードせず、tail normの再計算・DF再分解は行っていない。

| 系 | 保存prefix | コピーしたlambda_R | context SHA256 |
|---|---:|---:|---|
| H4 | 1 | 2.8629570945298686 | `8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f` |
| H6 | 7 | 0.1649333839054104 | `61d87e2fe7e03892f7db6a075a891d9b83af834832b083ec3de3b75e8a7e9d38` |

lambda_Rは既存decompositionの保存値で、R_outerや新しいlog Bと混同しない。保存contextの`prefixes.1.lambda_R`/`prefixes.7.lambda_R`をそのまま使う。この照合は保存SOURCEへのbindingの確認であり、Hamiltonianからnormを数学的に再証明したものではない。H6の過去snapshot status `H6_NOT_AUTHORIZED`とground資格false等は履歴値として変更せず、今回の実行許可の代替に使わない。

## 再利用と実行入口

- 既存 `pr_r_screening.screen_normalization`：K2 finite normalizer、log B・2log B、小τ主項、1 ULP累積感度、精度未資格処理を不変で再利用。
- 既存 `finite_rte_distribution`、`h46_m5_plan` / frozen allocation、Bのmetadata関数を不変で使用する。N solver/費用evaluatorは呼ばない。
- 新 `pr_r_normalization_phase0.preflight/public_plan/execute` と `run_pr_r_normalization_phase0.py`：固定24候補の検査と保存・表示だけを追加。候補・R・budgetを変えるCLI引数はない。

既定コマンドはpreflightだけ：

```sh
python -B scripts/run_pr_r_normalization_phase0.py --output-dir <fresh-preflight-dir>
```

今回このモードを実行し、[PREFLIGHT_LOG](PREFLIGHT_LOG.txt)の`normalization_evaluations=0`を確認した。Planの24候補、7入力hash、9 SOURCEの対応が揃った。

**ユーザーの明示実行許可後だけ**、既存project venvで次を一回実行する：

```sh
python -B scripts/run_pr_r_normalization_phase0.py --execute --output-dir <fresh-phase0-results-dir>
```

実行時は保存PLANとの一致と、使用SOURCEが記録commitと一致することを確認する。出力はfresh directoryのみ。候補追加/drop・別lawへの置換を拒否し、入力/SOURCE変更、例外・数値精度未資格を保存する。新Rや別精度での自動retry/resume、Phase 1やN/C/Gへの自動移行はない。

## 固定した科学call範囲・数値処理

許可後の範囲は**2 system×12候補の既存normalization評価だけ**。piece slot144に対して既存finite式を使い、qPFとpiece event数を解析的に会計する。qR回のtrajectoryや行列作用は行わない。小さなscalar処理であり、worker/CPU/RAMの恣意的な上限は追加しない。

log Bは既存K2式、attenuationはexp(−log B)、normalization由来のshot係数はexp(2log B)。後者はゼロ誤差を仮定した既存geometryの係数で、actual N・最小shot数・厳密な資源下界ではない。Rの科学的PASS/FAILや推奨Rは出力しない。

減衰予算は常にnull、閾値なし記録モードに固定する。1 ULP累積感度は診断で、rigorous error boundではない。bが1へ丸まる等は既存のNUMERICAL_NORMALIZATION_UNRESOLVEDを保つ。exp表示のoverflowはvalue=null、underflowはvalue=0と専用status、logは保持する。浮動小数点の表現限界を科学的不成立やR採択へ読み替えない。標準JSONへInfinity/NaNを書かない。

## 出力・B/Cへの受渡し

[HANDOFF](HANDOFF.md)にfield/単位・消費条件を記した。実行時にはresult.json、HANDOFF.json、NORMALIZATION.csvとprogress.jsonを保存する。各recordにcandidate ID、T/qRPE/qPF/δ、prefix/K/R、signed weights・piece allocation/event数、全request.identity/law_binding、normalization方法/SOURCE、log B、attenuation/shot係数と各表示status、丸め感度を含める。

`N=C=G=null`、phase/radius/N未評価、recommended_Rと科学的PASS/FAILはnull。B/Cは同じcandidate/law/回路条件で照合し、normalizationの比較材料としてだけ使う。全24候補が数値記録済みでSOURCE/input保全が成立した場合だけ、HANDOFF.normalization_evidence_valid=trueとなる。未資格/STOPや不足を0で埋めない。

## 検証・未実行範囲

[TEST_LOG](TEST_LOG.txt)：**新10テストPASS**。実分子の新normalizationを計算しないpreflight、固定24/144slot、tail値のそのまま転記、H/state/source不一致・候補変更拒否、人工tailで24候補・mock保存、law/件数置換拒否、overflow/underflow・precision未資格・入力消失、出力上書き/retryなしを確認した。元screenの人工12件はSOURCE不変の保存記録を再利用した。独立科学レビューではない。

現在、実24候補のlog B/attenuation/shot係数は全て未評価。新signal/phase/radius/十分N・C/G、R選択、SCF/DF/state生成/固有値問題、trajectory、回路build/compileは開始していない。保存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/過去STOPを保全した。

次の担当：ユーザーが本固定24候補のnormalization-only実行を判断する。許可後にCodex Aが同じ入口で一括実行し、B/Cへ同じrequestの記録を渡す。今回の準備公開・remote SHA/主要成果物照合後はmandatory STOP。
