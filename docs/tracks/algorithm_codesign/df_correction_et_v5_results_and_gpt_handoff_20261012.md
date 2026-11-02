# Track B E/T v5 one-shot結果とGPT引き継ぎ

2026-10-12 JST。固定sourceから一回だけ実行した。**`TECHNICAL_INCONCLUSIVE`**で終了し、
**mandatory STOP**している。旧runを再試行せず、新v5のretryも0。

直接原因はE-r6 controlled whole回路の数値誤差guard失敗である。
source／contract／上限を修正せず、T合成や追加の科学計算へ進んでいない。
保存したpartial結果からA/Bの科学的優劣、研究GO、PR総費用の勝敗を判定しない。

## 固定source・実行HEAD・承認

| identity | 値 |
| --- | --- |
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| execution branch | `track-b-df-correction-et-v5-one-shot-execution-20261012` |
| frozen source S5 | `d0a12bae817ce75c5876a06c2db00397914ca9b0` |
| execution HEAD、authorization-only A5 | `0e147e2746544657072053b7d648b41eee299ffc` |
| A5の唯一のparent | S5 |
| contract SHA256 | `586b2ca43a49e6e8adb539ac364b0b6d76cbf02fe827981b9d9ee95334ac00fa` |
| source manifest SHA256 | `a5267a3f8121f50a01447299265fa5a64a9b00d6a47867fed1e9ebbe46a7903e` |
| authorization SHA256 | `adcdb41e7296e0faea96b47b925447e72c028fb2dd03caa6f36884c7be0835e8` |
| consumed marker SHA256 | `5499d1aebb8edff3d5d38be719cb1a45e2ac9f8b120c1ba677e7738297229063` |

利用者の「次の作業に進んで大丈夫」を、直前に提示したS5の別明示one-shot承認への応答として
[receipt](df_correction_et_execution_authorization_v5.md)とauthorization JSONに原文どおり記録した。
A5で変更したのはそのJSONとreceiptのみ。A5をpush後、remote SHA=A5、clean、direct parent、
705個のmanifest対象identity、新result namespace未存在をlaunch gateで確認した。
固定numeric/synthesis runtime identityの照合はmarker消費前にrunnerが実施した。

旧source S4、旧authorization A4、旧v4結果とconsumed marker、以前の比較・診断・G10・RA-D0証拠は保持する。
今回のnamespaceは`artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v2/`。
資料公開後のHEADは結果／公開確認commitであり、実行時HEAD A5と区別する。

## 確認できた停止原因

外側processのexit statusは**2**。stdout、terminal、外側receiptは次で一致する。

```text
classification: TECHNICAL_INCONCLUSIVE
technical_reason: CapFailure: Child failed or output missing: E-r6
```

E-r6 childはexit **1**。保存ログは固定sourceの`native_evaluation.py:98`で
`ArithmeticError: Whole native numerical allowance not certified`を示す。
最後の保存raw observationはr6／arm A／controlled=trueで、
`COST_RETURNED_BEFORE_IDENTITY_GUARD`、fingerprint_attempts=1、transpile_attempts=1、
posttranspile instruction count=**50,554**。
したがって、recursive fingerprintとtranspileは完了し、その後のstrict numerical upperの認証で停止した。
sourceはそのcombined upperを固定`1/5000000 = 2e-7`と比較している。

このcombined upperはOperator residual、native forward arithmetic guard、ideal reference radius、
angle lowering／global phase allowanceを含む。**失敗したcombined upperの値および内訳は保存されていない**。
ledgerは計算後、guard通過時だけnative rowへ追加されるため、r6 controlledのledger／ordered native IRは保存結果にない。
どの項が支配的だったか、また実際の演算子誤差が2e-7を超えたかは確認できない。
「認証できなかった」ことと「実際に不正な演算子だった」ことを区別する。
不足値を取得するためのOperator再評価、target matrix計算、再compileは今回実施しない。

旧v4のoutput guard停止とは異なり、今回の保存量は**24,075,195 bytes**。
output、log、CPU、wall、RSS cap超過は記録されていない。Supervisorの`CapFailure`名は
child失敗を伝播するwrapperでも使用されるため、今回を資源上限超過と呼ばない。

## 到達した範囲

| stage／項目 | 保存記録 |
| --- | --- |
| terminal completed_stages | `CD_SAVED_REFERENCE_ONLY`のみ |
| CD | 既存6行のread-only参照。再計算0、旧run再分類0 |
| E-r3 child | exit0、8 compile、8 stored accepted native rows |
| E-r6 child | exit1、2 compile、prefixにA uncontrolledのaccepted row1件 |
| E-r6 controlled A | compile/cost returnまで。numerical guard失敗、accepted rowなし |
| native compile合計 | **10/22**：A4、B6 |
| raw observation | 10件。すべてsingle fingerprint・single transpile、例外なし |
| stored numerically accepted native rows | 9件。全runの科学的勝敗には使用しない |
| E-r6 B | 未到達 |
| synthesis inventory／T jobs／T sequence | **未作成、0/64 calls** |
| T accounting／coverage判断 | 未到達。Tの優劣を判断する資料はない |
| retry | **0** |

入力は保存済みE-r3/r6、rank2×3、seed7101107の元生成順のframe、sign(+1,-1,+1)、
theta1/32、kappa4、u1/128、uncontrolled/MSB-controlled。
Qiskit1.3.0、basis rz/sx/x/cx、opt0、transpiler_seed20261011を維持した。
Tはpygridsynth2.0.0、strict phase、dps100、gate precision1/40000000000を固定したまま未実行。
native/synthesisは各2workers・numeric threads1、phase overlapなし、GPU query/useなし。
CDとEの異なるfixtureを混合してPR総費用を算出しない。

## 資源記録

| 指標 | 外側process receipt | runner Supervisor receipt |
| --- | ---: | ---: |
| wall seconds | 57.7941260999 | 55.1531115500 |
| CPU seconds | 68.903263、reaped runner tree | 68.311253、parent/reaped children |
| peak sampled aggregate RSS | 737.12890625 MiB | 737.12890625 MiB |
| retries | 0 | 0 |

外側はrunner起動からidentity preflight等を含む。Supervisorはその初期化以降を測る。
RSSは50ms samplingであり、instantaneous ceilingや隠れた短時間peakを証明するものではない。
原resultは219ファイル、log合計1,684 bytes。最大physical fileは262,144 bytes。
64 MiB output、4 MiB log、256 KiB metadata reserve、native worker RSS1024 MiB／aggregate2560 MiB、
total RSS3072 MiB、既定CPU/wall上限を変更していない。

## 保存値とprovenanceのread-only監査

監査namespace：`artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v2/`。
`audit_saved_outputs.py`はstdlibとlossless codecだけを使用し、
matrix／circuit／synthesis／LPを計算しない。
**22,821 PASS、0 FAIL**。gateごとの21,768項目は集計数を保存し、残る1,053項目は個別に保存する。

- 原219ファイルのbytes/SHA256と集合が一致。削除・追加・上書きなし。
- root→snapshot/history→blob/chunkの完全graphを照合し、latest logical JSONのbytes/hashを検証。
- source manifest705対象を照合。実行監査時は全一致。
  結果公開に伴う6個のindexの追記は実行後のdocumentationのみで、元prefix・全scientific code・contract・manifestは保持。
- A5のsole-parent、authorization-only paths、approval/runs1/retry0/STOP、marker／terminal／外側provenanceを照合。
- raw fingerprintを保存payloadから再hashし、input arraysの保存bytesとframe contract identityを照合。
  同じQ/T/U、Operator、回路を再生成した検証ではない。
- 保存された9 native IRのordered digest、wire/order、gate count、phase、RZ inventory、accepted error上界を照合。
- CDの旧6行参照・no-reclassification・no-E-cost multiplication、T inventory/jobs未作成を確認。

新serializerは今回の保存prefixとstate historyをcap内で保持したが、E/T full completionの保証や科学的改善を示したものではない。
`COMPLETE_E_T_COMPARISON_NO_AUTOMATIC_GO`には到達していない。

## GPTへ戻す事項とSTOP

今回のrunはA/B資源比較を完了できず、候補の優位性・非優位性・新規性は未判定。
直接停止箇所は判明したが、失敗した誤差上界・ledger・compiled IRの欠落によりcomponent attributionは未完了。
既定誤差保証を緩めたり、他のprecision、fixture、compilerへ変更して通すことはしない。

次に数値guardの妥当性や誤差会計を変更するならGPTの科学的判断が必要である。
保存・観測だけを増やす技術準備でも別scopeが必要であり、この一回実行承認で自動的に修正・再実行へ進まない。
旧v4および今回のmarkerはconsumedのまま、原terminalとTECHNICAL_INCONCLUSIVEを保持する。

結果・外側status・audit・取得manifestをcommit/pushし、remote SHAとGitHub再取得を照合して引き継ぐ。
**mandatory STOP。新しい科学実行、retry、追加T keys、G10/RA-D0、分子、GPU、sampling、LPは行わない。**
