# 研究C：Fresh IID-grid校正基盤の技術実装

2026-11-04 JST。**標本生成・再現記録・失敗保全・既存統計処理への接続を人工入力で実装した。科学校正は未実行、NOT_READYを維持する。** 新実H4 sampling、native build/compile、誤差評価、期待bias校正、Stage C、actual CA判定、重要科学レビューは0。以下のテスト値・人工のtotality/enclosure前提は科学的証拠ではない。

Branch：`codex/research-c-fresh-iid-calibration-infrastructure-20261104`。独立worktree：`/tmp/prt-research-c-fresh-iid-calibration-infrastructure-20261104`。起点はStage P完了`c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e`。固定P0技術準備`60dcaa94fd44675bc3184e409aa4ebdef7a07dbf`をmergeし、既存モジュールと固定証拠をそのまま利用する。競合は4索引の追記のみで、双方の記載を保持した。元のdirty checkoutと並行作業のbranch/worktreeは編集していない。

正式共通指示`research_project_instructions_unified_2026-10-25.md`原本の§8.1・§8.4・§10を確認した。今回の明示指示が技術実装・人工検証・commit/pushの許可であり、新しい承認証明・grant・署名・承認台帳は作らない。

## 固定条件と成果物

将来の対象はLinear H4、隣接1.00 Å、STO-3G、number CDF8、元順LD2、partial ordered S2/s1/r32/K0、h=δ=0.1、m0..12。Nα=Nβ=2、sector36/system256/controlled512、own-DF ground oracle、状態準備費用除外。PR endpoint SHA256：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、既存Gaussian/relative fusion、MPFR160 phase、RPE/CA政策を維持する。

| 成果物 | 役割 |
|---|---|
| [adapter](../../src/trottertracks/c_fresh_iid_calibration.py) | 明示整数列/PCG64 replay、記録、失敗保全、統計payload |
| [passive入口](../../scripts/prepare_research_c_fresh_iid_calibration.py) | SOURCE/input bytesと非実行契約の照合のみ |
| [人工テスト](../../tests/management/test_research_c_fresh_iid_calibration.py) | 境界、順序、失敗、統計拒否、SOURCE/STOP |
| [未承認契約](research_c_fresh_iid_calibration_contract_unapproved_20261104.json) | 未採択項目をnullで保持、executable=false |
| [技術SOURCE](../../artifacts/research_c_fresh_iid_calibration/2026-11-04/technical/SOURCE.json) | 新adapterと直接再利用コードの再現性用identity |
| [passive結果](../../artifacts/research_c_fresh_iid_calibration/2026-11-04/technical/passive_preflight.json) | 実行不可・科学call0 |
| [検証記録](../../artifacts/research_c_fresh_iid_calibration/2026-11-04/technical/tests.json) | 検証範囲、既存失敗、使用環境 |
| [STOP](../../artifacts/research_c_fresh_iid_calibration/2026-11-04/technical/STOP.json) | 技術準備終了、科学実行へ進まない |

## 標本と乱数の記録

`make_context`は供給されたStage A形式のbinary64 CDF、53-bit counts、順序付きatom ID、保存event角bits、candidate、科学的identity、新SOURCEを結び付ける。CDF/count一致と正q supportを検査する。元資料の数値lawを再推定しない。実H4のcontext/law IDは今回生成・採択しておらず、将来契約に固定する必要がある。

`sample_from_integers`は各occurrenceに明示された64個の53-bit整数を受け取る。各eventの**order選択→atom選択**を省略せず、既存`grid_index`の`searchsorted_right`を適用する。K0のorder選択は常に0でも、その乱数入力とoffsetを残す。新しい一般RTE samplerは作らず、event実体化は`rte_events`から既存`_make_event`を呼ぶ。符号・位相・角度・順序の実装を重複させない。

記録はdataset/sample index、Sample ID、SOURCE/candidate/law/context identity、m、axis、coherent occurrence、event ID/順序、両53-bit入力/offset、選択atom、保存角bits、word ID、round word IDを含む。Kernel単位は1本のlength32 word、round単位は2**m本のoccurrenceを保持する。Roundの`paired_XY`は同じword列を使う古典的なX/Y評価であり、量子測定shotの共有ではない。Actual inverseが必要なら同じ実現回路を反転する既存政策を保ち、逆回路用に再抽出しない。

`sample_from_pcg64`はNumPy1.26.4/PCG64を実装候補として用意する。新SOURCE/law/dataset/masterで領域を分け、既存`trajectory_seed`にcandidate/m/axis/sample/occurrenceを渡す。各doubleに対応するuint64上位53-bit、seed、PCG64前後state、消費64個を保持する。人工テストで既存`c_pr_aware_df_stage_b.independent_occurrence`とevent列・最終stateの一致を確認した。実H4では呼んでいない。

数学モデルはevent/occurrence/shotで独立な一様53-bit整数を仮定する。一方、PCG64は有限の決定的streamであり、seed領域の分離・異なるSample ID・再現一致は数学的IID性の証明ではない。全record/resultで`PCG64_joint_stream_qualified=false`を保ち、統計interfaceも明示的なIIDモデル仮定に条件付ける。実PRNG/backend/master/Nは未採択である。

## 失敗・未完了wordの保全

状態は`GENERATED`、`RAW_BUILD_FAILED`、`P0_COMPILE_FAILED`、`NATIVE_INELIGIBLE`、`PER_WORD_CERTIFICATE_FAILED`、`NUMERICAL_UNEVALUABLE`、`INCOMPLETE`、`SUCCESS`。Outcomeは元の標本全体、失敗段階、例外type/message、部分証拠を保持する。

`open_batch`はfresh directoryだけを許し、`save_sample`でbuild前に抽出結果を保存する。Sample/outcomeは排他的な新規ファイル作成で書き、重複IDの上書き・再抽出を拒否する。**別Sample IDに同じwordが出現することはIIDで合法**であり、word重複は削除しない。Outcomeがない標本は読取り時にINCOMPLETEとなる。書込み途中の破損は消去・再生成せず読取りエラーとして保全する。`finish_batch`は状態件数とmandatory STOPを保存し、以後の追加を拒否する。読取りはresumeではない。

即STOPと固定N全件記録のどちらも今回採択しない。保存関数はpolicyを実行するrunnerではない。既存`Calls`は最初の例外でstopをsetするため、固定N用にそのstopをresetして継続する実装はしない。将来は採択されたpolicyに適合するrunner/call定義を別に固定する。失敗aware estimator、fallback、replacement、retry/resumeは実装していない。

## 統計処理への接続状態

`calibration_payload`は事前固定Sample ID集合と全outcomeを照合する。不足・非SUCCESS・不明enclosure・未採択confidence/policy・未資格population/compiler premiseは`NOT_READY`。成功した標本だけを渡しても、元の固定ID集合を満たさず拒否される。

Per-word enclosureは既存`evaluate_word`形式に、新しいsample/SOURCE/candidate/law/occurrence/wordのbindingを付けたものを要求する。旧固定4-word/native32の未結合certificateは受け付けない。既存word/target/angle、full512の二leg、phase-fit禁止、rounding enclosure、provenance、population rangeの検査を再利用する。SOURCE/bindingは再現性のための照合であり、数学的包囲の前提自体を証明する仕組みではない。

| 接続先 | 接続内容と未資格事項 |
|---|---|
| Threshold tail | `c_per_word_population_range.threshold_tail`へ供給。既存zero-bad/保守的indicator処理を再利用 |
| Variance-sensitive | 同`variance_sensitive`へ供給。N≥2と数値区間が必要 |
| Direct round | `c_iid_grid_expected_bias.paired_round_upper`へ供給。全occurrenceのper-word certificateに加え、shell込みX/Y差区間、raw complex compiled/literal信号、m別population axis rangeが必要 |
| Confidence/qualification | 既存`failure_accounting`と校正event IDを保持。全13roundのjoint配分・完全reference transportは将来の`qualify`に必要。Stage C/CAは呼ばない |

Stage Rの保存population norm/rangeは取得済みだが、全word compiler成功の前提は今回も未資格。`calibration_inputs`が要求するall-q structural totality premiseを本adapterは生成しない。失敗率案・別algorithmは未採択であり、現実の校正はNOT_READYのまま。人工テストだけが明示的な架空premiseを用いる。全word一様小誤差、実PCG64資格、actual CAへ昇格しない。

## 再利用・保全

Stage P `c01bd2742`、P0 `60dcaa94f`、期待bias `bd560a454`、Stage R `1c0e1d1b4`、Stage A `ae9a4e13d`を固定参照とした。新しいsampling/record adapterが必要だったのは、既存samplerが乱数入力・標本IDを保存せず、旧Callsが即STOPだけを実装しているためである。

再利用した主な処理は`independent_occurrence`（一致reference）、`trajectory_seed`、`rte._make_event`、C2 `choice_grid`、expected-bias `grid_index/sampling_gate/paired_round_upper/failure_accounting`、population `grid_support/calibration_inputs/threshold_tail/variance_sensitive`、P0 `check_word`。科学実行の接続候補として既存`SharedQiskitCalls`、`compile_once`、`Calls`、per-word certificateも確認したが、今回のadapterから実H4 runnerを起動しない。共有moduleの実装は変更していない。

保全はGitの固定blobと差分で確認する。旧SOURCE/raw/契約/grant/ledger/marker/STOP、固定Stage Pの部分STOP・完了結果、Stage R原本は無変更。P0技術準備の22 pathは固定commitからの統合であり、新しい科学結果ではない。相手のcache/lock/process・共有venv/OS設定は変更していない。

## 人工・回帰検証

新規72件は、53-bit/CDF境界・zero bin、order消費、既存PCG64 samplerとの一致、occurrence/round区別、ID/SOURCE再現、重複検出、旧成果物拒否、全失敗状態、INCOMPLETE、無断再抽出禁止、IID/PRNG区別、enclosure欠落、wrong SOURCE/candidate/law、retry/resume/STOP、科学library importを禁止したpassiveを検査した。

固定環境Python3.11.1/NumPy1.26.4/Qiskit1.3.0/python-flint0.8.0/FLINT3.3.1で、7ファイルの最終対象範囲は**276 PASS（新規72＋既存204）・既存2件を明示deselect**。除外した2件の失敗も全試行ログで確認・保存した。全テストは人工・mockまたはbyte/schema確認であり、外部CI・科学的GOではない。

既存2件は、(1) 実行済みStage Rの出力が起点commitにも存在するため旧passiveのfresh判定が拒否するもの、(2) 旧SOURCEの`c_pr_integrated_performance.py`固定digestが起点commitの同じfile bytesと不一致になるもの。起点blob/現file/旧STOPを照合して継承状態を確認した。旧契約・STOPを修正せず、失敗ログと理由を残す。

最初の回帰試行は別の既存venv（flint0.9.0・threadpoolctlなし）を選んだためruntime/dependency検査に失敗した。環境を変更せず、既存の固定環境へ切り替えて再検証した。その初回ログも技術記録として保持する。新adapterの初回71件は通過し、最後にsector-only enclosure拒否を1件追加した。

今回のworkloadは小さい整数列・人工回路・scalarで、重い科学計算はない。将来のsubstantial runでは実CPU affinity/quota/free RAM/GPUと既存timingを確認し、独立標本/occurrence/列を並列化、immutable input/cache共有とprocess-local Arb/Qiskit、BLAS/Rayonの多重並列を調整する。人工的CPU/RAM/worker/wall capは置かず、速度向上率は未測定として主張しない。

## 未承認契約・次の担当

契約は`executable=false`。N/N_by_round、δ_cal、必要ならδ_compile、方法、failure policy、数値資格、実PRNG/master、将来科学SOURCE/context/law ID、call総上限はnull。Kernel/roundのcall式と不足条件を記録し、旧Stage P/R/固定4-wordの予算を流用しない。Fresh outputを別namespaceに指定し、retry/resume0・partial保持・mandatory STOPを維持する。承認証明やgrantを要求しない。

研究上の結果・科学資格は変更していないため、研究概要・科学manifest・科学ノートは更新せず、関係する4索引と本技術報告だけを更新した。Commit/push後はremote SHA/treeと本報告・contract/SOURCE/code/testの本文を照合する。

次の担当は**ユーザー／研究C**：校正方法、N、confidence、compiler failure policy、数値資格を選ぶ。**Codex**：選択後にSOURCE/context/law、call上限、資源計画、fresh出力を固定した新科学契約を完成させる。科学校正はその固定契約に対するユーザーの別途明示許可後の作業であり、今回の許可には含まれない。

**MANDATORY STOP。技術準備・公開照合で終了し、新実H4 sampling/build/compile、誤差評価、期待bias校正、Stage C、actual CA、重要科学レビューへ進まない。**
