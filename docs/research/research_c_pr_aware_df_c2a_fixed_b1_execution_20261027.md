# 研究C：C2-A B1・保存固定回路の事後validated科学実行

2026-10-27 JST。結果：**FIXED_B1_ALL9_VALIDATED_PASS**。登録9判定の内訳はPASS 9／correctness FAIL 0／UNRESOLVED 0／NOT_RUN 0。

固定科学SOURCE `518a730510a57d0cafa1575a91f5f1979c203be3`を変更せず一度実行した。[固定契約](research_c_pr_aware_df_c2a_guarantee_contract_draft_20261027.json)の歴史的UNAPPROVED metadataは保持し、ユーザーのチャット上の実行許可に従った。新grant・承認証明書・承認原文保存は作っていない。

## 対象と実行範囲

保存直鎖H4・1.00 Å・STO-3G、通常number CDF rank8、Nα=Nβ=2/sector36、system8+ancilla1/full512。B2 run02のbaseline_00だけ、元frame順LD2、partial symmetric S2、δ=.1/s4/h1/40/r16/K0、元g補正κ・scalar、own-DF ground oracle/状態準備除外という保存回路来歴を維持。今回oracleやHamiltonian/DF/PF/RTEを再計算していない。

保存raw/compiled kernel、実際のraw/compiled adjoint、共通head/X/Y raw/compiledの10QPYを読込んだ。P0 Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=falseを変更せず、新しいbuild/compileは0。

## 全gateと数値上界

| 比較 | 状態 | 次元 | operator2 upperの表示近似 |
|---|---|---:|---:|
| kernel_compiled_raw | VALIDATED_WITHIN_FIXED_ALLOWANCE | 512 | 3.4694675824E-12 |
| kernel_raw_actual_adjoint | VALIDATED_WITHIN_FIXED_ALLOWANCE | 512 | 1.1468373022E-53 |
| kernel_compiled_actual_adjoint | VALIDATED_WITHIN_FIXED_ALLOWANCE | 512 | 5.5421187600E-15 |
| head_compiled_raw | VALIDATED_WITHIN_FIXED_ALLOWANCE | 2 | 7.4993994327E-17 |
| head_raw_exact_reference | VALIDATED_WITHIN_FIXED_ALLOWANCE | 2 | 1.2744832619E-57 |
| X_compiled_raw | VALIDATED_WITHIN_FIXED_ALLOWANCE | 2 | 7.4993994327E-17 |
| X_raw_exact_reference | VALIDATED_WITHIN_FIXED_ALLOWANCE | 2 | 1.2744832619E-57 |
| Y_compiled_raw | VALIDATED_WITHIN_FIXED_ALLOWANCE | 2 | 4.3297802812E-17 |
| Y_raw_exact_reference | VALIDATED_WITHIN_FIXED_ALLOWANCE | 2 | 1.2744832619E-57 |

規範値は[CSVのexact rational](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1_certificates9.csv)と各JSON。表示値は上向きDecimal近似。固定allowanceは`1/100000000`、Arb192／python-flint0.8.0／FLINT3.3.1、phase fitなし。Upperがallowance以下ならPASS、lowerが超えればFAIL、区間が跨げばUNRESOLVEDとして後続を止める。事後の閾値・precision・SOURCE変更、fallback、retry/resumeはない。

Kernel3比較は各full512の全512計算基底columnを評価する。各gate後にArb ballを中心とcertified vector2 radiusへ戻し、全columnの距離上界をFrobeniusで集約する。有限probe最大の外挿や局所Gaussianのsector射影ではない。Gate評価・丸めを含む既存validated-libraryの実装契約に依存し、hardware/FLINTの独立formal verificationではない。

Actual adjointは保存された実inverse QPYと、対応するrecordの数学的adjointを比較した。Shellはlast ancilla-onlyを厳密検査し、唯一のterminal ancilla measurementだけを除いたunitary prefixを2×2へ縮小した。System空間へのembedding normは不変。Raw shellの基準はHまたはH*Sdgで、自由なglobal phase合わせはしない。

## 資格と限界

得られるのは**この保存固定gate列のraw/compiled、actual inverse、shellの事後operator証拠**だけである。Raw circuitが物理Hamiltonian/CARの正しいevolutionであること、そのreferenceの数値enclosure、Gaussian vacuum phase、未検証trajectory、全law、PCG64 IID、CA達成、shots、費用support、総資源優位は認定しない。未知値はnull。

旧C1の約83%仮定付き点推定と、旧Aのblocked全law契約は変更しない。検証したbaseline_00を全coherent occurrenceへ繰返せば、元PRの独立trajectory lawが変わるので、その置換を許可・提案していない。新たなarm比較やsample選択もない。

## Call・資源・保全

実績は**19/19 logical units**、COMPLETE 19／FAIL event 0。内訳 `{"qpy_decode": 10, "fixed_pair_certificate": 9}`。新fit/optimizer/build/compile/sampling/C1 shots/総資源再計算0、retry/resume0。内部全基底評価は固定契約の各比較unitに含む。Runner wallは2055.6671408824623秒。資源詳細は[原本](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1-run01/resources.json)、[監視記録](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1_resource_monitor.jsonl)。Parent/children peak RSSはそれぞれの定義であり、children peakを合計RAMとしない。

独立worktree、task-local venv/cache、依存は既存環境のread-only参照。Workersは実CPU affinity/quota/空きRAMに従う既存process-column経路。CPU/RAM/worker/wallの恣意的上限、共有設定変更、他jobへの操作はない。

[Preflight](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1_execution_preflight.json)はSOURCE128/input44/QPY10/外部Python SOURCE22/package11を照合し、直前人工37 tests PASS。[Postflight](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1_execution_postflight.json)で固定SOURCE/input、旧tracked 1926件のbyte保全、raw hashesを確認した。Scientific SOURCEは変更していない。

SOURCE manifest SHA256：`e8180d8e0a5479df4bfda527fbe7610a7a3d812a6b7ff7c3fc077427d0d68e80`。契約SHA256：`50b0d1e7051c707284c3d4a1296036b82773a5bc07fe3c6303af26a6b06d3e6a`。

公開後処理はderived CSVのCRLFが`git diff --check`に掛かり一度停止した。9行の全cellが同一であることを照合してLFへ統一し、公開処理だけを継続した。[修正記録](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1_publication_repair.json)。科学SOURCE/rawの変更、追加call、科学retry/resumeは0。

## 次の担当とSTOP

次は**ユーザー／研究C**が、この有限scopeの結果から、別の保存kernelへの拡張、raw/reference enclosure、または確率モデル付き資格のどれに情報価値があるか判断する。**Codex**は新しい具体的指示と必要な別科学実行許可後だけ進む。今回の結果から全law・CA/shots・C1総資源・compiler政策変更・GPT詳細科学レビューへ自動進行しない。

Branch：`codex/research-c-pr-aware-df-guarantee-b1-execution-20261027`。結果をcommit/pushし、remote SHAと同じGit treeへ固定したSOURCE/input/成果物を照合後、**mandatory STOP**する。[実行原本](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1-run01/)、[要約](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1_execution_summary.json)。
