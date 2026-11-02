# H2 PR/PF保存費用の同一runtime静的結合（B-2）

**PR134軸と共通Full4軸の保存費用objectを固定証拠へ結合した。同じPython3.12 binary・主要compiler/runtime sealと元RZ計測定義を支持する証拠はある。一方、現在の共通nativeへのFull全parameter stream結合、provider差の完全な出力非干渉、完全な過去actual-load、Aの独立diagonal証拠は未解決。政策2aの限定的な採択検討資料まで完成し、無条件の再利用認定・正式採択・COMPARE GOは行わない。**

2026-10-23、GPUA100-gleap02。専用branch `codex/h2-saved-cost-static-join-b2-20261023`。基準はGPU01回収 `bb9e0bd167daa8e618ec203f8ea3b3cd26440ebf`（前回B `ca29dd9ddc8a09cace84310f0dfdf7124652e07a` を含む）。技術SOURCE `51af878110f704a0140553aec57c984cd7fef084`。担当Codex自身の技術検査であり、別agent・独立科学レビュー・外部再現ではない。

## 保存証拠の結合範囲

| 系列 | 固定対象 | 保存単価と確認範囲 |
|---|---|---|
| PR S2 | `H2O32_c7ec590a32a859bc`、33 trajectory/66軸 | 前回Bの保存整数を参照。各元cost receiptをGPU02で再hashし、compiled raw SHAと前段operation証拠を結合 |
| PR Yoshida4 | `H2O32_8a2ecdd0294ffb74`、34 trajectory/68軸 | 同上。EXPLORATIONの元cohortを維持 |
| 共通Full S2/q4 | `H2_FULL_3_4_2nd`、cosine/sine | 保存1655 RZ/軸。GPU01原本検証を継承し、原plan/result/QPY archive/guard/費用receiptを固定Git資料で結合 |
| 共通Full Yoshida4/q2 | `H2_FULL_3_2_4th`、cosine/sine | 保存2463 RZ/軸。同じ範囲の結合 |

[138軸の差分identity索引](integrated_axis_identity_138.json)は固定commit/path/JSON pointerとobject SHAを持つ。新しい費用平均・総費用・比・順位を作らず、旧Full1367/2033 RZを混入していない。**保存cost identityの結合は138/138軸。現在の共通nativeと完全identityを新認定した、という意味ではない。**

Full原plan SHAは `6eef028aff888dd64d8c6bb1833f01112df1026e452f8b89318c866f5e1a1892`。入力5件は原planのrole/SHA/bytesへ結合。Hamiltonian `d0bc260d…`、state `080b7ad2…`、Full preparation `98c1beb120c22077b409550a4d61c31638b9a52637246fae4cb1ac55a776f1d9` を照合し、受入済みDIAGNOSTICS prefix3のpreparationとも一致した。[Full原plan・入力・QPY対応](Full_plan_input_QPY_binding.json)。

Fullの原result2件と12 archive（wrapper4/compiled4/vector4）の原本byte確認者は**GPUA100-gleap-01**。GPU02は原本を受領・再hash・QPY再parseしていない。GPU02で今回直接確認したのは固定Git資料の全文byte、元PR cost134 receipt、関連PR runtime30 file、および既知保護対象である。GPU01のfixed_sources97資料はローカル固定GitからSHA/bytesを再照合した。[固定資料索引](fixed_source_inventory.json)。新しいremote取得・private転送は0。

旧PR snapshot容器 `62c6481b…` と共通PF実使用容器 `8bb1aa35…` はbyteが異なる。原SOURCE・H/state・保存array-equivalence receiptを継承した対応であり、旧容器の原本復元、新しい配列decode/数値比較は行っていない。

## 科学条件・native stream

固定条件はlinear H2/1.0 Å/STO-3G、tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4。Fullはprefix3/S2q4・Y4q2、PRはprefix1/K4/outer32/S2q4・Y4q2。prefix/lawの違いは意図された比較自由度として維持し、不一致としない。元RTE law、signed schedule、usual signed bias、empirical allowance、旧reference条件付き再利用、shot政策は変更しない。

PF科学SOURCE `1b35f9ac…` と元PR SOURCE `4ec26921…` の `h46_new2_native`、`h46_m5_native`、H2 phase guardの全文byte一致、および現在のacquisition側file一致を確認。追加で`h46_new2_core`、`h46_m5_contract`、`h46_m5_plan.schedule`のSOURCE一致を確認した。[native/compiler対応](scientific_native_compiler_join.json)、[schedule/scalar SOURCE補完](schedule_scalar_SOURCE_supplement.json)。元関数は一度も呼ばない。

元Full経路は`cost_one → h46_m5_native.compose → signed S2 native pieces → scalarを末尾に1回 → full Hadamard wrapper`。共通productionも同じprepare/compose/native providerを参照する。symmetric-directional control、元scalar、signed schedule、Gaussian loweringのSOURCE定義は整合する。GPU01はwrapperの先頭H、末尾H/Measure、sineのSdg、ancilla4・測定bit0を直接検査している（5量子bit・1 classical bit）。compiled QPY内guard metadataと原resultの一致もGPU01の検査を継承した。

**不足はFullの全gate/wire/parameter/global phase/control/orderと、現在意図する共通native streamとの逐命令結合である。** handoffのQPY証拠は全byte消費parseのstream digest・instruction数・register数とwrapper終端検査まで。全文instruction projectionは含まれず、digestだけから各parameterの意味を復元しない。SOURCE一致とpreparation hash一致は強い補完であるが、この不足を自動PASSにしない。保存streamの静的identity、生成経路の対応、all-state/operator同等性は別の主張である。

Aは固定 `43be11d6…` の状態を参照するだけで、worktree/cache/outputを変更していない。`absolute_deterministic_diagonal_vs_common_preparation` は134/134軸で未解決のまま。既存独立receiptがない場合の3 piece時間×2 block＝6 primitive評価案は別承認対象であり、今回0件。

## compiler・runtime・provider差

[runtime結合](runtime_join.json)はFull原planの365 closure entryを**元のindex・role・SHA・bytes**で固定projectionへ照合した。SOURCE14組の重複entryは同じpath/role/SHA/bytesを保ったまま残し、365個の独立fileとは表現しない。

- Python binary1/1、loaded-runtimeと名付けられたseal8/8、compiler/array binary21/21がPR sealと一致。対象30 fileはGPU02で現存byteを再hashした。
- provider236 entryはrelative pathで元PR sealへ結合し、235一致・1差分。単なる同じSHAの別pathへの置換をしない。
- compiler/cost counter/native provider/guardの関連SOURCEは一致。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし、guard内num_processes1という**歴史的compiler設定**を維持する。これを新しい資源上限へ一般化しない。
- NumPy cp312 binary19件、Qiskit Rust binary、SymEngine cp312 binaryの21件を照合。version名だけの同一性判定ではない。

[provider差のSOURCE監査](provider_difference_scope.json)：`h48_preflight_contract_v1.py` はPF `ab0879e0…`、PR `f5db5d88…`。追加は`h8_state_plan/phase_plan/phase_resources`、変更は`verify_binding`。H4/H8 preflight・H8_STATEの計画/資源/authority/binding制御で、RZ counterやnative算法の変更ではない。H2関連compiler/native/fingerprintとpackage initializerの検査対象8 fileには当該moduleへの直接参照がない。

ただし、これだけで過去の動的・推移的import全経路の非干渉を証明したことにはならない。**provider byte差はMISMATCH、compiler出力・費用への非干渉はUNRESOLVED。実際の費用誤差が見つかったとはしない。**

GPU01は原planの宣言絶対path上のbyteを確認済みだが、runtime pathの非公開mapはGPU02へ未転送。公開projectionはruntime ordinal/basenameまでで、GPU02が完全なorigin対を復元できたとはしない。両系列の完全historical worker actual-loadとtransitive dependency manifestも未発見。支持する主張は**固定宣言closure・関連SOURCE・主要binaryの対応**までであり、完全な過去実load identityや3.12対3.11出力同等性は含まれない。この限界だけから再compileが必須とは判断しない。

## correctnessと政策2aの判定

[correctness scope](correctness_scope.json)：PRはCT01 single saved input state、Fullはcompiled_vector各軸のsingle saved input state、いずれも保存strict <1e-7・phase_alignment=false。Fullのlow-control/high-leakage/independent-occupation/saved-signal/Hadamardのstrict <1e-9は元receiptの別scopeとして継承。vectorのbyte identity、compiled cost identity、single-state正しさ、all-state/operator保証、完全historical runtimeはそれぞれ区別し、新しい誤差評価は0。

| 分類 | 判定 |
|---|---|
| 証明済み | 固定138軸の保存cost-object対応、Full原plan/input/result/archiveのGPU01検証との結合、H/state/DF/preparation、関連native/compiler SOURCE・主要runtime binary・保存correctnessのscope |
| 条件付き | 同じ宣言Python3.12系・関連compiler SOURCEの**保存費用系列**として扱う限定政策の採択検討。prefix/law・historical load・correctness限界を明示する必要がある |
| 未解決 | Full逐命令parameter/phase streamの共通nativeへの結合、provider差の完全な非干渉、完全historical load/origin、A独立diagonal、all-state保証、旧3.12対3.11出力条件 |
| 実際の差分 | provider preflight1 file、およびsnapshot容器byte。科学的出力の不一致や対象軸の費用誤差を新たに確認したものではない |

[受入条件別表](policy2a_conditions.json)と[metadata-only preflight](metadata_preflight.json)。**限定した同宣言runtimeの保存費用候補を検討する材料は揃ったが、無条件の共通native再利用候補としての証拠は未完。** Codexは政策2aも代替契約も採択しない。旧strict SOURCE `46a3b33e…`、契約digest `a65f3e90…`、旧DRAFTを維持し、COMPAREはBLOCKED。

## 追加compile・転送・次の判断

**現段階で追加compileの必要性は認めない。** 必要な元134+4費用は保存され、Full原本もGPU01で確認済み。旧4軸runtime probeや全134軸の再compileは起動・新規必須化しない。過去実loadを新compileで復元することもできない。

**このB-2静的結合の完了にprivate転送は不要だった。** 残るFull stream結合は、まずGPU01の保持者/Codexが既存QPY parserと元SOURCEを使い、対象4 wrapperのordered gate/wire/hex parameter/phase/control/scalarについて限定静的検査receiptを追加する方法が最小候補（科学call/回路生成/compile0）。その証拠をGPU02へ渡す範囲は別途公開/私的経路の判断対象で、今回送信・依頼・転送していない。

もしGPU02自身の直接stream検査が必要と判断された場合だけ、GPU01本人所有研究領域の管理者から**対象4 wrapper gzip**をアクセス制限付き私的領域へ渡す案を検討する。memberは固定[archive_identity_12](../../track_a_h2_common_pf_original_recovery/2026-10-23/archive_identity_12.json)のwrapper4行、目的は上記parameter streamの静的照合。受領時は同じ`verify_gzip → read_qpy → 既存stream検査`でstored/raw SHA・byte数・全消費を検証し、原本を上書きしない。compiled QPY4件はcompiled stream自体の独自確認を追加要求する場合だけ別scope、vectorやseed、grant類の転送は不要。新たなprivate転送を既定の前提としない。

担当ユーザー：政策2aをどのruntime/correctness/native証拠scopeで正式採択するかを判断する。Full静的streamの追加証拠を先に求めるか、限界を明示した限定政策として扱うかを区別する。Aの6 primitive科学評価、COMPAREの新SOURCE/契約/unused grantを伴う科学実行、private受渡し、成果物公開はそれぞれ独立した判断/承認である。

担当Codex（今回未開始の後続作業）：必要と判断されたFull静的receipt補完やSOURCE固定・実行契約準備を行う。科学実行の明示承認なしにprimitive/compile/COMPAREへ進めない。[次の判断・最小範囲](next_steps_DRAFT.json)。

## テスト・保全・終了監査

既存の原本/SHA検証、QPY静的parser、operation stream証拠、Bのguard/cost/seal join、GPU01 gzip最小adapter、保全observeを再利用した。[再利用記録](reuse_inventory.json)。新規実装は異なるhost/出力schemaの間のjoin・証拠確認者の区別・非実行preflightに限定し、compiler/角度算法/cost counter/correctness計算を再実装していない。元scientific runner全体は実行しない。

[人工133件PASS](test_results.json)：新43、既存B34、GPU01 gzip/schema8、代替契約48。[ログ](technical_tests.txt)。対象混入/欠測/重複/元SOURCE/cohort/axis、archive/raw SHA/bytes、設定/guard、runtime role/path/index/SHA、費用単位、scope昇格、未承認科学入口を検査。初期のcommit identity不足、closure重複entryの扱い、結合辞書の重複keywordという技術不具合はcommit履歴と[技術試行記録](technical_attempts.json)に残した。いずれも科学call0で、元結果・保全baselineは維持した。最終静的audit PASS。

[保全監査](preservation_audit.json)は既知の原PR/関連SOURCE/input/runtime/結果/authorityとDIAGNOSTICSの16,804 fileについて、当初baselineからSHA/bytes変化0。他worktreeのHEAD/statusを凍結条件にせず、Aや共有venv/OS/scheduler/他ユーザーprocessへ書込み0。空きCPU/RAM/容量を確認し、恣意的な低い資源上限を追加しない。科学grant発行/旧grant再利用0。最新運用指示§8.4/§10を適用した。

[最終照合](final_verification.json)、[成果物inventory](file_inventory.json)、[STOP](STOP.json)。科学call0、primitive0、回路生成0、compile/transpile0、simulation0、sampling0、u/N/G再解析0、ranking0、COMPARE0。private転送0、詳細科学レビュー/最終freeze0、push0。最新個別指示§12に従いローカルcommitまでで終了する。**政策2a・代替契約は未採択、COMPARE BLOCKED、mandatory STOP。**
