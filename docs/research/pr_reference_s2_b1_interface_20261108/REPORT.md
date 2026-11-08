# 新A B1：保存S2 1 recordの局所interface correctness

2026-11-08 JST、SERVER 2 (`GPUA100-gleap02`)。基準 **ca1e71b3624705d7a942572b91b0b50ce4eeb713**、[固定scope §3](../pr_reference_s3_s2_reuse_20261108/NEXT_VALIDATION_PLAN.md#3-b1保存候補1-recordの有限interface検証)。Branch `codex/pr-reference-s2-b1-interface-20261108`。

**保存1 record・固定4 fixtureの局所correctnessはPASS。B1完了、mandatory STOP。** 新43-key policy、S1上界改善、full SOURCE phase、m17／全18roundのRPE成功・総Tは認定していない。初回の参照fixture不備と補正結果を分け、採用結果は[verified_final/RESULTS.json](../../../artifacts/pr_reference_s2_b1_interface/2026-11-08/verified_final/RESULTS.json)とする。

## 1. 固定入力と実装

対象はS2で既に保存されたframe候補のうち、次の1 recordだけである。Guard/T frontierは再計算も最適化もしていない。

背景の研究条件はLinear H4／1.00 Å／STO-3G／DF12、Partial S2／LD3／h=.02／K2、元H厳密ground供給・準備費除外、compiler Aである。今回のoperator数値はこの物理系を新simulationした値ではなく、以下の人工1q／3q fixtureだけの診断値である。

- Literal：`0x1.0b1d9883f9213p+1`。
- Archive：`artifacts/track_c2_b_pilot/2026-11-06/raw_synthesis.tar.xz`。
- Member：`synthesis/1c223a1ac0ccd6b7e2729dae93da988f4bf9d3bbc10ac1bba0500ed4ebc3029f.json`。
- Backend spec：`0x1.0b1d9883f9213p+1:10^-13:pygridsynth2.0.0:strict:seed17`。
- Positive sequence SHA256：`981736be8f2688787aa2e1da4598b9d42b016abf32171fcc27038f88f55c0155`。保存W=7、T=140は元字段の引用だけであり、新しい価格集計ではない。

元record全体、raw member／archiveのhash、固定4入力、SOURCE、runtimeは[初回INPUTS.json](../../../artifacts/pr_reference_s2_b1_interface/2026-11-08/INPUTS.json)に保全した。SOURCE対応例はB′ `catalogue/rare0_Y0.json`／`[false, false]`、basis `df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2`のnative position423、wire2である。保存crosswalkとのmetadata一致を確認し、whole basis／m17の回路はbuildしていない。

[SingleSavedRecordOverlay](../../../src/trottertracks/pr_reference_s2_record_overlay.py)をfixture専用に追加した。元spec・guard文字列・positive word・T/T†/Wを保持し、設定したliteralの正負だけを受け付ける。Backend exponent=-13とconsumerのsequence-only transport key=-11を区別する。Negativeにもpositive sequenceを渡し、変更していない`lower_native`が全体のactual adjointを一度だけ取る。元recordのguardは引用し、新しいstrict guardを計算しない。

既存DictionaryProvider／C0 cache／85 certificate／43基本word、emitter、consumerを変更していない。Overlayのselectorにarm／roleの引数はなく、B′／Referenceに同じrecordを提供する仕様である。今回は人工local fixtureでの接続確認であり、両armのwhole回路を実行したという意味ではない。

## 2. 固定4 fixtureの結果

`a`は上のbinary64 literal、時間順は固定scopeどおりとした。

| Fixture | Native入力 | 結果 | Wを含む保存word参照とのspectral診断差 |
|---|---|---|---:|
| standalone positive | 1q、RZ(+a,q0) | PASS | 1.773059745376315e-14 |
| standalone negative | 1q、RZ(-a,q0) | PASS | 1.9676090553073168e-14 |
| joint positive | 3q/1c、reset全3q; SX0; CX02; RZ(+a,q2); CX02; SX0; measure0→c0 | PASS | 2.1115476018034196e-14 |
| joint negative | 同じjoint列、RZだけ-a | PASS | 2.0261773261985206e-14 |

入力global phase=0はこの人工fixtureの定義であり、SOURCE templateの欠けたphaseを補った値ではない。Positiveの出力phaseは`0x1.5fdbbe9bba775p+2`（7×binary64 π/4）、negativeは`0x1.921fb54442d18p-1`（binary64 π/4）。全wordのactual adjointのexact cyclotomic identityも確認した。

Gate順序、named inverse、wire、register、reset／measureの順序とclbitを照合した。Jointのconsumer出力は、単独emitを接続した別の参照構築と全named命令列・保存phaseが一致し、両jointのoperator診断差は0.0だった。Unitary segmentは各1、inplace join各1で、copying compose・新QPY・新controlは0。Raw命令列とconsumer auditは採用RESULTSに保存した。

Operator参照は既存`exact_word_matrix`のexact cyclotomic→100 dps interval変換から中点を取り、Qiskit／numpyで比較した。元RZ targetに対する誤差を新計算したのではない。閾値1e-10はN4-B emitter診断を再利用した。**これらは有限の数値correctness結果で、新しいstrict guardやfull-law保証ではない。** Named理想gate・既存IEEE phase-path条件下のemitter allowance（CACHE_EMITTER_UPPER）とconsumer shellの条件付きSOURCE論証を維持するが、新しいcombined guard／RPE予算を評価していない。数値行列積の丸め誤差を厳密emitter誤差そのものと扱わない。

## 3. 初回の参照fixture不備と技術補正

[初回runner](../../../scripts/validate_pr_reference_s2_b1_interface.py)では単独正負2件がPASSした後、最初のjointで`ins(out)==ins(independent)`が失敗した。Consumerはnative SXを既存`lower_native`によりH,S,Hへ展開するが、私が独立参照側にSX命令を残していたことが原因である。元record、W、consumer又はSOURCEの不一致を示す結果ではなかった。

初回[RESULTS.json](../../../artifacts/pr_reference_s2_b1_interface/2026-11-08/RESULTS.json)・[STOP.json](../../../artifacts/pr_reference_s2_b1_interface/2026-11-08/STOP.json)・INPUTSと実行済みSOURCEをそのまま残した。初回結果を採用結果に書き換えていない。

[補正runner](../../../scripts/validate_pr_reference_s2_b1_joint_corrected.py)は独立参照のSXだけをH,S,Hに直し、同じ固定joint2件を完了した。成功済みstandalone2件は保存named stream／phaseを再利用して再実行していない。元native fixture、overlay、word、emitter、consumer、比較閾値は不変更。これは承認scope内の参照構築の技術的修正であり、旧研究の科学的STOPや科学条件を解除・変更するものではない。

補正SOURCE・固定入力・初回証拠hashは[verified_final/INPUTS.json](../../../artifacts/pr_reference_s2_b1_interface/2026-11-08/verified_final/INPUTS.json)に保存した。保存word行列参照は初回と補正で計2回、最初のjointの停止した照合試行1回もcall記録に残した。別word／角度／fixtureへ探索を広げていない。

## 4. 再利用・検証と実行境界

元の`lower_native`、`exact_word_matrix`、N4-Bの`ins`／`expected_stream`を変更せず関数単位で再利用した。Unrelated runnerのmodule初期化を実行しないためのAST読込は従来方式を継承した。Consumer／dictionary本体も基準commitとのbyte一致を確認している。

[人工metadata test](../../../tests/test_pr_reference_s2_record_overlay.py)は5件PASS。正負の同じpositive word／全体adjoint、未知literal拒否、spec／hash／W破損拒否、backend／transportの区別、外部dict変更の非伝播を確認した。初回test起動はPYTHONPATH不足でimportに失敗し、`PYTHONPATH=src`で同じtest SOURCEを実行してPASSした。人工testのguard文字列は人工ラベルであり、科学的近似の証明には使っていない。

RuntimeはPython3.12.3、Qiskit1.3.0、pygridsynth2.0.0、mpmath1.3.0、numpy1.26.4。専用の既存venvを読取利用した。起動時にCPU affinity64、利用可能メモリ、loadを観測し、空き資源を確認した。4件は小さく、joint参照が単独出力に依存するため同一processで実行した。CPU/RAM/workerの独自cap、共有設定・package・他ユーザーへの変更はない。

新規合成・strict guard・native compile・新RZ target誤差評価・T価格取得／再集計・A1 basis defect・S1/S2再実行・precision変更・RPE再会計は全て0。旧SOURCE／raw／result／grant／ledger／marker／STOPを保全した。現在のB1結果は[verified_final/STOP.json](../../../artifacts/pr_reference_s2_b1_interface/2026-11-08/verified_final/STOP.json)でmandatory STOPとしている。

## 5. 次の担当と未資格事項

**次担当はGPT／ユーザー。** この1 recordを再合成せず現emitter／consumerへ接続できる局所interfaceを確認できた。43-key policyを採用するか、A1の現word defect評価又は全体actual-adjoint adapterの検討へ進むかは、別途の判断・対象scopeの実行承認が必要である。今回その先へ進んでいない。

43 literalがS1のframe maxへ与える効果、Dのguard/T tradeoff、全support policy、B′ scalar／Reference vacuum／eventのdomain、central、emitterの未照合経路、shell／terminal、旧diskとの合算、保存626／630 shotsの十分条件、全18round総Tは未解決のまま残す。既存word再利用の局所PASSを、RPE成功・総T削減・一般的優越又は新規性GOへ格上げしない。
