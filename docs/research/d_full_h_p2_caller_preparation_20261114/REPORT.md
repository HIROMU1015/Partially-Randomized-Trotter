# Full-H κ2 P-2 caller：技術準備

**技術実装完了・科学計画は未採択・mandatory STOP。** 新しいH4 preparation、sampling、回路build/compile、statevector、保存科学値の数値再解析、fit、E[C]、N/Gは実行していない。

実装SOURCEは `6069605d50270111d1510478b08e11b8e5a18d86`。branchは `codex/d-full-h-p2-caller-prep-20261114`、基点はP-1結果 `a04470c0ad489eea1798856e648f863f9092cfb1`。既存branchを確認し、P-1の系列を保ちつつPauli・決定論PFの並行作業から分離した。merge、main変更、他worktree変更は行っていない。

## 実装とA設計の対応

既存P-1 callerにcohort/indexとround引数を追加し、既定値を保全した。新しい `scripts/run_full_h_kappa2_p2.py` はcell選択・seed表・paired receipt照合・Cへの出力を担当する。sampler、compose、native emitter、wrapper、fingerprint/cache、compilerは既存経路を使う。

| 対応案（未採択） | 較正cell | holdout | 最大compile（X/Y合計） |
|---|---|---|---:|
| D6 | G566、G4528、G36223：q1各n2。Q8R566：q8/R566、n2 | m5 n2、m6 n1 | 22 |
| D7 | D6＋P2R566：q2/R566、paired n1 | 同上 | 24 |
| D7_H2 | D7と同じ | m5 n2、m6 n2 | 26 |

q1 groupはT=0.1、その他は元binary64規則 `T=0.1*q`、`s=q*R`。holdoutは元canonical scheduleのm5/m6であり、新しいconfigurationを作らない。全cellのphysical row、law、cohort、group seedは [PROPOSALS.json](PROPOSALS.json) に保存した。cohort名とseedは技術例であり、科学実行に採択していない。

A `7a0c9468b2499fac9fa1cd279ae2980c9dab048e` の `finite_price_design.py` をbyte一致で再利用する。取得入口の最初に `check_plan` を呼び、旧q=1のみの18 compile案（rank 2）を拒否する。D6/D7/H2はrank 3。これは識別可能性の検査であり、モデル妥当性や精度の証明ではない。

paired q2はG566のpath0/group0、path1/group0のseedをその順で再利用する。親receiptのphysical row・cohort・index・seedをcontext前に確認し、生成された順序付きevent（位相を含む）と親の連結を比較する。保存JSONでtupleがlistになる表現差は、値を丸めない順序付きJSON比較で吸収する。toyの親receiptをJSON往復させて確認した。親は独立標本、paired cellは親と相関する。独立較正点として誤ったSEを計算してはいけない。

## correctnessと価格quantity

P-1の修正済み `SCALAR_EVENT_PHASE_ANCILLA_SIGNATURE_V1`、event order-2位相、ancilla命令順、scalar一回、new controlled ZZ、旧経路との命令parity、X/Y同一制御進化、index0作用gateを再利用する。P-1のphase helper等はAST一致を確認し、変更関数は `pilot` と `price_bindings` に限定した（[TECHNICAL_CHECKS.json](TECHNICAL_CHECKS.json)）。

対象は測定付きfull controlled Hadamard wrapperのcompiled RZ数。Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17、`CONTROLLED_ZZ_CX_CRZ_CX_V1`を維持する。X/Yを個別にcompileする入口を保ち、片軸で代用しない。状態準備のgateは追加せず、保存状態は既存correctness検査に使う。

非configuration cellはCの `PRICE_CALIBRATION_RECORD_V1`、canonical m5/m6は既存B価格record＋C固定guardへの入口を使う。`DIRECT_COMPILED_FINITE_SAMPLE`、`FINITE_COHORT_ONLY`、実際のn_paths、law expectation falseを保つ。構造Sやevent数をcompiled Cへ昇格させない。今回生成したのはmetadataと人工recordだけで、実価格recordは作っていない。

## 人工・回帰検証

| suite | 結果 | 対象 |
|---|---:|---|
| [P-2 metadata](d1-p2-prep-tests.txt) | 10 PASS、4.705 s | rank、q/R/s/T、domain、未採択拒否、stage/cohort分離、seed、人工C recordと固定guard、凍結metadata |
| [P-2 toy回路](d1-p2-circuit-tests.txt) | 2 PASS、4.258 s | toy 4-mode、order-2、scalar/event位相、X/Y、paired seedのevent一致、compiler patch復元 |
| [元P-1回帰](d1-p1-regression-tests.txt) | 親4 PASS、18.218 s | 内部の既存toy 11件、metadata/子import-only、元C guard |

toyの価格はfake compiler。実H4の回路・compile・statevectorを含まない。テストは作者自身のlocal検証であり、独立レビューやCI再現ではない。実施時の作業treeと同じcodeを上記SOURCEで固定した。実H4の性能・資源上限・長時間価格は未測定。

## 重複照合と残る条件

正式共通指示 `research_project_instructions_unified_2026-11-14.md` §10.5を適用した。Aの既存coverage調査を再利用し、今回は固定記録とのmetadata差分だけを確認した。P-1 run02はm3配線用の異なるR/cohortであり、P-2較正へ混入しない。旧Full-H価格は旧loweringであり、新ZZ価格へ再ラベルしない。旧P-1のsampling/compile、run02再解析は行っていない。

本実行直前にはprice table/catalog、元trial/receipt、aliasを含むphysical/law/profile、関連running job、SOURCE/cohort/seedを再確認する必要がある。今回はlive科学実行preflight完了とは認定していない。決定論PF Stage0 `acea1118a4df4e727e13398ce0f8ae11511c7396`、32件計画、過去raw・STOP等への変更はない。

Aの資源順序案はG566/P2/Q8/m5を先、G4528/G36223/m6を後としている。一方m5はHOLDOUTなので、callerは全較正の凍結モデルSOURCEなしでは実行しない。**順序の整合を採択時に解決する必要がある。** 今回m5を較正に変えたり、部分較正モデルを採用したりしていない。

ロードマップでは「P-2 cellとpaired callerの技術対応」が完了した。D6/D7採択、価格精度、許容誤差、finite-cohort不確かさ、rare/sampler差、m7〜m12外挿、正式Gへの資格は未決定のまま。[HANDOFF](HANDOFF.md) と [NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md) を参照。**新科学実行へ進まずSTOP。**
