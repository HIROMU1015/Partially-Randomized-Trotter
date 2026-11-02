# 研究C：保存Proposed固定回路F1・限定科学実行

2026-10-27 JST。**FIXED_F1_ALL3_VALIDATED_PASS**。保存H4・直鎖1Å/STO-3G、通常number CDF8 proposed_00、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、system8+last ancilla/full512、own-DF oracle/状態準備除外。SOURCE `e7d22a404442234662ef6d117c55dd9868217c9e` を無変更で一度実行し7/7 calls、PASS3/correctness FAIL0/UNRESOLVED0/NOT_RUN0、retry/resume0。Arb192全512基底・位相込みの有限固定回路保証。共通shell6 certificateはidentity一致で再利用。Raw/CAR・全law・CA/保証shots/総資源は未資格、F2 executable=falseを維持。

ユーザーのチャット上の明示許可に従い、[固定F1契約](research_c_pr_aware_df_fixed_proposed_contract_draft_20261027.json)を変更せず実行した。歴史的UNAPPROVED metadataは保全した。承認証明書・新grant・承認原文保存は不要で、作成していない。

## 固定SOURCE・比較対象

公開起点 `fb5cfa01a98ecad63c004ec11d7b95d0e22f8592`、科学SOURCE `e7d22a404442234662ef6d117c55dd9868217c9e`、独立branch `codex/research-c-fixed-proposed-f1-execution-20261027`。科学SOURCE manifest132件 SHA256 `0b10324fd59835dc371a65290dbd52e8909d9d85fbb44120421907f107f77701`、契約SHA256 `07d4dde2260179e84d828fa89f82bac783df5052c7c9dfa4b512e4010e073e07`。保存B2 run02のproposed_00 raw/compiledと実際のraw/compiled adjointの4QPYだけをdecodeした。元J/U、h/g、groundやNPZ referenceを数値decodeせず、CAR/F2は実行していない。

保存科学条件は上記H4・CDF8・LD2で、元g補正κ・scalar一度・元frame順を維持。各DFのground oracle／状態準備除外、CA=.00159 Haは来歴であり、今回CAやoracleを再評価しない。P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false、MPFR160 declared phase、B2共通Gaussian/relative head融合は不変。新しいbuild/compileは0。

## 全3判定

| 比較 | 状態 | 次元 | 評価column数 | operator2 upper（上向き表示近似） |
|---|---|---:|---:|---:|
| compiled_raw | VALIDATED_WITHIN_FIXED_ALLOWANCE | 512 | 512 | 3.2092694515E-12 |
| raw_actual_adjoint | VALIDATED_WITHIN_FIXED_ALLOWANCE | 512 | 512 | 1.4620980294E-53 |
| compiled_actual_adjoint | VALIDATED_WITHIN_FIXED_ALLOWANCE | 512 | 512 | 5.5421187600E-15 |

規範値は[exact rational CSV](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F1_certificates3.csv)と[各原本](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F1-run01/)。固定allowance operator2=1e-8。Upper≤allowanceでPASS、lower>allowanceでcorrectness FAIL、跨ぐ場合はUNRESOLVEDとして依存後続を停止する。precision192・phase規約・SOURCE・閾値の事後変更、fallback、retry/resumeはない。

既存Arb/FLINT gate算術で、保存binary64 parameterとglobal phaseが定義する数学的unitaryを比較した。全512の計算基底columnについてlocal gate actionを行い、recenter後のcertified vector2 radiusを含めてFrobenius上界を集約する。有限probe最大の外挿やGaussian gateのsector射影ではない。Actual adjointは保存inverse QPYと対応recordの数学的adjointを直接比較した。Global phase fitはしない。Validated libraryの実装契約に依存し、hardwareやFLINT自体の独立formal verificationではない。

## 再利用と両armの資格

共通head/X/Y raw/compiledの6QPYと旧B1の6shell certificateは、保存path/hash・last-ancilla-only・唯一のterminal measurement除外規則が一致することをpassive確認し、そのまま再利用した。新しいshell decode/certificateは0。

[旧Baseline固定B1](research_c_pr_aware_df_c2a_fixed_b1_execution_20261027.md)のraw/compiled上界3.4694675824e-12は既存値の引用であり再計算しない。今回結果と合わせても対象は各armの保存index0だけ。Baseline／Proposedは別SOURCE/JU、別seed・別sampling lawで、index一致をpaired trajectoryや全law対応と扱わない。誤差の大小によるsample再選択や新しい費用比較はない。

## Call・資源・保全

実績 **7/7 logical units**、COMPLETE 7、FAIL event 0。内訳 `{"qpy_decode": 4, "fixed_pair_certificate": 3}`。新fit/optimizer/sampling/build/compile/NPZ数値decode/DF-PF-RTE再計算/C1再解析/shots/総資源0、retry/resume0。

Wall 2255.4224520605057秒、runner peak RSS 121772 KiB。Runner peakはparent定義であり、worker合計RAMではない。[資源監視](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F1_resource_monitor.jsonl)は当該jobだけのprocess数・aggregate CPU/RSSとpressureを記録する。Tool sessionごとのPID namespaceのため通常sandboxから別session processが見えない場合があり、ホスト側read-only監視で自分のjobだけを確認した。32 workerは実affinity/quota/空きRAMに従う既存全column経路。共有環境・他jobへの変更、CPU/RAM/worker/wallの人工上限はない。

[Preflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F1_execution_preflight.json)：SOURCE132/input29/package11、fresh output、F2非実行を照合。直前の人工回帰6件PASS。[Postflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F1_execution_postflight.json)で固定SOURCE/input・旧B1 raw25件不変と新raw hashesを照合した。旧契約・grant・ledger・marker・STOPを変更していない。

## 未資格事項・次担当・STOP

この結果は**保存固定回路のraw/compiledとactual inverseの位相込みoperator証拠**。Rawと独立理想CAR／元物理Hamiltonianの一致、reference自身の誤差、未観測trajectory、all-law synthesis、PCG64 IID、実装CA、保証shots、cost support、母平均費用、総資源優位は認定せずnullを維持する。C1約83%の仮定付き点推定も変更しない。

F2は保存dyadic Uの非unitarityを扱う理想target選択が未採択のため、元の `executable=false` を維持し、F2/F3やpolar案を自動採択・科学実行しない。**ユーザー／研究C**がCAR target方針と次の最小科学scopeを判断し、**Codex**は具体的な別指示と必要な明示許可後だけ対応する。重要な科学レビュー・追加sampling・全law/CA/shots/総資源へ進まない。

SOURCE・raw・call journal・STOP・結果をcommit/pushし、remote SHAと成果物を照合後、**mandatory STOP**する。
