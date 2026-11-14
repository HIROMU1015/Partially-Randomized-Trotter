# PR-Pauli B1 caller：技術準備

**技術実装完了・B1科学計画は未採択・mandatory STOP。** 新しいH4 preparation、sampling、回路build/compile、statevector、価格fit・E[C]・N/Gは行っていない。ただし回帰suiteの選択を誤り、禁止範囲にあった**保存H4 P0演算子のbit順数値再検査を1回実行した**。[SCOPE_DEVIATION](SCOPE_DEVIATION.md) に入力・処理・ログと非採用の扱いを記録した。「保存科学値の数値再解析0」とは報告しない。

実装SOURCEは `bd1b1b2862da6d174fa521e7612051810a636e7f`。branchは `codex/d-pauli-b1-caller-prep-20261114`、基点はB0結果 `00f74f86d16cc6878e1d4c5977eb9abd9ce9c076`。Full-H P-2、決定論PF、他担当worktreeから独立させた。merge、main変更、他worktree変更はない。

## 実装とA設計の対応

既存B0 callerの `canonical_request`/`p0_arrays` にprefixを、`pilot_core` にindex・step_seeds・event metadata引数を追加した。LD1/index0/従来の結果形式という既定動作はtoyで旧SOURCEと比較した。新 `scripts/run_pauli_b1_prices.py` は選択した有限cellだけを既存pipelineへ渡す。

| 対応段階（未採択） | 内容 | compile能力（X/Y合計） |
|---|---|---:|
| B1-W | LD2 m0、normal n1、元canonical physical ID | 新規2 |
| B1-V（任意）LD1 | m0、事前宣言時のみB0を元cohort/path0として利用しpath1追加 | 新規2＋保存2 |
| B1-V（任意）LD2 | B1-Wの元cohort/path0を保存SOURCEから利用しpath1追加 | 新規2＋保存2（W未取得なので実行不可） |
| B1-CAL LD1 | 1 step R=171/1361/10885 各n2、独立q8/R171 n2、paired q2/R171 n1、m5 n2/m6 n1 | 24 |
| B1-CAL LD2 | 1 step R=66/526/4205 各n2、独立q8/R66 n2、paired q2/R66 n1、m5 n2/m6 n1 | 24 |

LD1 B1-Vには全てfreshなn2（4 compile）の能力もあるが、B0と同じcellなら事前宣言した再利用を優先する。LD2 B1-VはWの続きであり、Wを新たな独立標本として数えない。現在LD2 Wの保存証拠は固定SOURCEに存在せず、templateの参照は未設定。B1-CAL LD2も公開Wのphysical/gate metadataなしでは開始しない。

A `7a0c9468b2499fac9fa1cd279ae2980c9dab048e` の `finite_price_design.py` はbyte一致。CALは `check_plan` によるrank3・domain・role/pair検査を通す。W/Vのrank1は配線/ばらつき専用として明示し、モデル較正recordを出力しない。planの欠落、q/R/s/T、law、lowering、prefix不一致や未採択実行は取得前に拒否する。全request、辞書binding、proposed cohort・step seedは [PROPOSALS.json](PROPOSALS.json)。

## 科学的・技術的契約の保全

保存P0辞書・係数/順序/hash、C plannerのphysical ID、保存H/state、normalization、DF headのblock順、`s_R`とnative tail identity、`c_H+s_R`一回、full-space oracle、low/high control・leakage・X/Y符号・compiled-vector gateは既存B0の処理を使う。LD1/LD2ごとに対応辞書とP0演算子を選び、非configuration cellには辞書/context/q/R/Tを結んだ別physical IDを付ける。

`PR_PAULI_TAIL`、`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、Qiskit 1.3.0/ rz,sx,x,cx / opt1 / seed17、測定付きfull controlled wrapper、fingerprint/cacheを保つ。DF用ZZ loweringや旧IDでは通さない。native/wrapper/D2 emitterは固定blobとbyte一致、full-space oracle・scalar関数等はAST一致を確認した（[TECHNICAL_CHECKS.json](TECHNICAL_CHECKS.json)）。実H4でのLD2 gate成功を今回証明したわけではない。

step seedの明示指定は元normal samplerのseed入力だけを置き換える。イベントの順序、selected label、order、D2 product token/step phase、seed、fingerprintを保存できる。paired q2は1 step path0/1を順に連結し、親receiptと位相込みで照合する。scalarは連結後のtotal Tで一回、親とpairedの統計相関を保持する。

価格は `DIRECT_COMPILED_FINITE_SAMPLE` / `FINITE_COHORT_ONLY`、実n_paths、law expectation false。B0のX/Y各1380 RZは元単一pathの保存証拠であり、再取得・新law平均への昇格をしていない。非configuration cellはC較正record、canonical m0/m5/m6は既存価格sidecar＋固定C guardへ接続する。構造S・未検証モデルをcompiled価格の厳密区間にしない。

## 検証結果と範囲逸脱

| 許可範囲内のsuite | 結果 | 対象 |
|---|---:|---|
| [B1 metadata](d2-b1-prep-tests.txt) | 11 PASS、3.422 s | rank/domain、prefix/辞書/physical、seed/pair、B0/W再利用、人工C guard、凍結/W metadata |
| [B1 toy回路](d2-b1-circuit-tests.txt) | 3 PASS、3.118 s | toy 4-mode、旧B0既定値同値、D2作用・label/product/位相・scalar一回、paired、prefix拒否 |
| [B0安全回帰subset](d2-b0-safe-regression-tests.txt) | 親8 PASS、11.126 s | 内部toy5件、metadata/record/guard/import-only。保存H4数値caseを除外 |

toyはfake compilerで、新H4 sampling/build/compile/statevectorを含まない。作者自身のlocal検証であり、独立レビューやCIではない。テスト時のcodeを上記SOURCEで固定した。

別途、当初のunfiltered B0親suite（9件、12.042 s）は保存P0演算子検査を1件含んでいた。これは今回の許可範囲外である。即時にユーザーへ報告し、元ログを保全した。検査のPASSを新しい科学証拠には採用しない。以降のdefault/`--metadata-only`/通常discoveryは当該caseを除外し、将来の別途許可がある場合に限る明示flagを残した。詳細は [SCOPE_DEVIATION.md](SCOPE_DEVIATION.md)。

## 重複照合・残るUNKNOWN・ロードマップ

正式共通指示2026-11-14 §10.5に従い、既にAが監査したB0/DF-tail/P3/旧Pauli-JWの範囲と新planのmetadata差分を確認した。B0再利用は元SOURCE・canonical LD1 m0・辞書・cohort/path0/seedを保つ場合だけ。LD2、新CAL、独立holdoutへ付け替えない。旧P3/DF-tail/旧Pauli-JWはlaw/head/K/辞書等が異なり再利用不可。今回live running job等を含む科学実行直前照合が完了したとは認定しない。

ロードマップ上はB1取得callerの技術対応が完了した。B1-W/V/CALの採択、科学SOURCE/取得scope、実LD2配線、価格分散・CI、精度/tolerance、sampler law差、rare寄与、m7〜m12外挿、正式E[C]/Gへの資格はUNKNOWN。Wの2 compileやCALの48 compileを今回採択したわけではない。巨大m12の直接compileを前提にしない。

決定論PF Stage0 `acea1118a4df4e727e13398ce0f8ae11511c7396`・32件計画、旧B0 run01/run02・raw・STOP、旧P-1等への変更はない。[HANDOFF](HANDOFF.md)、[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md) に続行条件を記す。**新しいB1科学計算には進まずSTOP。**
