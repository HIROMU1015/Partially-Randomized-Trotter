# P3 Stage 3：H4 native preflight PASS・23/36価格保存、compileで停止・run02

2026-11-02 JST。**native preflight 81項目PASS、価格23/36件保存、`Already mutably borrowed`によるSTOP。**
ユーザーの追加実行指示に基づく技術修正後の1attempt。旧run01のFAIL/STOPは保全し、閾値は変更していない。

## 固定条件・SOURCE・実装variant

Linear H4／隣接1.00 Å／STO-3G／中性singlet、system8/full256、保存DF tol-only1e-8/rank10。案A/T.8/S2/q8/r4/K4/M_B/M_C、P=0。
共通Gaussian保持T_effとscalarを共有し、Stage 2のPauli184/DF360成分・DF U/ηを使用した。
各arm4seed、flat32 event/trajectory、両軸同一wrapper。方式間は別標本で、paired比較とは扱わない。

- 科学SOURCE：`9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c`。公開・固定後に実行。
- Stage 3原契約：`be3de04e6fb501fecc4a07f38ceea7c24ba10472`。同じ科学tuple/seed/compilerと数値閾値を維持。
- 旧失敗結果：`aab0b483fdd8e57fe17305a11c5cb750a9c80a4d`。one-particle差2.376341682004808e-13によるFAILはそのまま保持。
- Stage 2結果：`8bcf5851f99f980dc08c5885ace54874b554c815`。保存入力22件と旧STOPは保全。
- project内Givensをexact-zero/atan2へ安定化。U.T・消去順・RZ/XXPlusYY familyを維持。共有qiskit-nature/NumPyは変更していない。
- 新native合成variantであり旧経路と同じ価格programとは呼ばない。旧run01は価格未取得なので旧価格を混合していない。

## 共通保持とnative correctness

共通保持basisのone-particle差は`4.0478224233294404e-16`、同じd8 PASS上端`7.1054273576010161e-15`以内。
actual adjoint、inverse合成、unitarity、vacuum、共通T半指数、controlled half、DF10 basis、scalar、F/E/R/Emptyを検査した。
全**81項目PASS**。完成したcontrolled bodyを事後phase alignmentしていない。
小さい人工例で旧loweringの近零/微小角の弱点を再現し、同じ入力・閾値のH4新attemptで回復した。旧差を単一の内部演算だけへ因果分解した結果ではない。

| 共通保持検査 | spectral差 | 判定 |
|---|---:|---|
| half | 6.8105919646146522e-16 | PASS |
| actual_inverse | 1.0953844902519636e-15 | PASS |
| controlled_half | 8.3387783330480689e-16 | PASS |

| native program | spectral差 | engineering不確かさ | 判定 |
|---|---:|---:|---|
| DF_0/F | 1.0060722297090791e-14 | 2.0560472207592453e-09 | PASS |
| DF_0/E | 8.1494637509295533e-15 | 9.8562255364346294e-10 | PASS |
| DF_1/F | 1.0669295263877352e-14 | 2.011515574185045e-09 | PASS |
| DF_1/E | 1.0115422814213533e-14 | 9.4107085038172772e-10 | PASS |
| DF_2/F | 9.8970467109741067e-15 | 2.1372104840255762e-09 | PASS |
| DF_2/E | 1.0866522384977679e-14 | 1.0668818775815754e-09 | PASS |
| DF_3/F | 9.7489237217112136e-15 | 1.9276808191385171e-09 | PASS |
| DF_3/E | 9.4294855243433405e-15 | 8.5775478409856915e-10 | PASS |
| P_0/F | 1.1292998680675151e-14 | 1.422689840100218e-09 | PASS |
| P_0/E | 6.712966674796883e-15 | 3.5277144495403796e-10 | PASS |
| P_1/F | 1.0609218119626868e-14 | 1.4259582447432312e-09 | PASS |
| P_1/E | 6.6291570884989235e-15 | 3.5616254106361942e-10 | PASS |
| P_2/F | 7.1056400882776989e-15 | 1.3720469207732997e-09 | PASS |
| P_2/E | 2.4546950210470195e-15 | 3.0220432074349244e-10 | PASS |
| P_3/F | 7.4402088341906497e-15 | 1.3895871988135015e-09 | PASS |
| P_3/E | 3.957071046984905e-15 | 3.2032460171773329e-10 | PASS |
| R | 6.5345886027227327e-15 | 1.0692767831004162e-09 | PASS |
| Empty | 0 | 4.0194366942304714e-14 | PASS |

数値目安はcomplex128のengineering検査でありrigorous certificateやcompiledρ上界ではない。

## 固定trajectory・coverage

| arm/index | seed | order 0/2/4の位置数 |
|---|---:|---|
| P_0 | 310201 | 30/2/0 |
| P_1 | 310202 | 31/1/0 |
| P_2 | 310203 | 31/1/0 |
| P_3 | 310204 | 30/2/0 |
| DF_0 | 310301 | 30/2/0 |
| DF_1 | 310302 | 31/1/0 |
| DF_2 | 310303 | 29/3/0 |
| DF_3 | 310304 | 30/2/0 |

標本を見た再抽出・追加trajectoryなし。order4が未出現でも小規模人工fixtureで構造を検査し、H4標本へ強制eventを追加していない。

## 取得済みcompiled価格（measure込み、P=0）

Qiskit1.3.0、qiskit-nature0.7.2の既存primitivesとproject内stable Givens、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing追加指定なし。
各F/Eは同じ保存eventから構築。共通R/Emptyは両arm共用。Fは共通保持/scalarと残差、Eは残差/event phaseのみ、Rは保持/scalarのみ。
入力/出力QPYと全histogram・compiler/fingerprintは保存されている。

| target | RZ | CX | depth | size |
|---|---:|---:|---:|---:|
| DF_0/E/cosine | 4832 | 1551 | 2197 | 9479 |
| DF_0/E/sine | 未取得 | 未取得 | 未取得 | 未取得 |
| DF_0/F/cosine | 9231 | 2991 | 4215 | 17985 |
| DF_0/F/sine | 9231 | 2991 | 4215 | 17985 |
| DF_1/E/cosine | 未取得 | 未取得 | 未取得 | 未取得 |
| DF_1/E/sine | 4617 | 1445 | 2129 | 9027 |
| DF_1/F/cosine | 未取得 | 未取得 | 未取得 | 未取得 |
| DF_1/F/sine | 未取得 | 未取得 | 未取得 | 未取得 |
| DF_2/E/cosine | 6210 | 1914 | 2748 | 12111 |
| DF_2/E/sine | 6209 | 1914 | 2748 | 12110 |
| DF_2/F/cosine | 9978 | 3170 | 4502 | 19381 |
| DF_2/F/sine | 9978 | 3170 | 4502 | 19381 |
| DF_3/E/cosine | 未取得 | 未取得 | 未取得 | 未取得 |
| DF_3/E/sine | 3655 | 1178 | 1728 | 7191 |
| DF_3/F/cosine | 未取得 | 未取得 | 未取得 | 未取得 |
| DF_3/F/sine | 8092 | 2628 | 3749 | 15766 |
| Empty/cosine | 0 | 0 | 1 | 1 |
| Empty/sine | 2 | 0 | 4 | 4 |
| P_0/E/cosine | 未取得 | 未取得 | 未取得 | 未取得 |
| P_0/E/sine | 155 | 155 | 224 | 362 |
| P_0/F/cosine | 未取得 | 未取得 | 未取得 | 未取得 |
| P_0/F/sine | 3070 | 1163 | 1579 | 5988 |
| P_1/E/cosine | 131 | 178 | 229 | 358 |
| P_1/E/sine | 130 | 178 | 228 | 357 |
| P_1/F/cosine | 未取得 | 未取得 | 未取得 | 未取得 |
| P_1/F/sine | 3044 | 1186 | 1571 | 5978 |
| P_2/E/cosine | 99 | 121 | 155 | 241 |
| P_2/E/sine | 未取得 | 未取得 | 未取得 | 未取得 |
| P_2/F/cosine | 2992 | 1121 | 1507 | 5806 |
| P_2/F/sine | 未取得 | 未取得 | 未取得 | 未取得 |
| P_3/E/cosine | 112 | 118 | 169 | 263 |
| P_3/E/sine | 111 | 118 | 168 | 262 |
| P_3/F/cosine | 未取得 | 未取得 | 未取得 | 未取得 |
| P_3/F/sine | 未取得 | 未取得 | 未取得 | 未取得 |
| R/cosine | 579 | 340 | 511 | 1114 |
| R/sine | 579 | 340 | 511 | 1114 |

機械読取表は[compiled_prices.csv](compiled_prices.csv)、取得済み23件のmetricsは[prices.json](prices.json)。欠測はCSVの空欄で保持し、価格0としていない。
これは正常返却・保存された個別価格の部分証拠。失敗と並行して同じprocess内で得た結果であり、全cohort比較・compiler作用保証を取得したとは扱わない。
母平均・費用tail・統計的方式順位・同精度総N/Gは評価していない。

## Compile停止・未取得13件・非加法性の未完了

Qiskit1.3.0のhigh-level synthesisが`self._equiv_lib.keys()`を参照する箇所で`RuntimeError: Already mutably borrowed`が発生した。
同一process内の36 ThreadPool compileという並列化方式で共有内部状態へのborrow競合が生じた。native基底の数値不一致による停止とは別問題。
36 call開始・23 call正常保存。残る13件には成功metricsがなく、個別の全例外traceは保存されていないため、同一例外を13件全てへ帰属させない。
最初に観測した例外と全thread回収後の件数はSTOP/execution_statusに保全。取得済み23件を破棄・上書きせず、自動retryもしていない。
未取得targetは[pending_targets.json](pending_targets.json)に同じ保存input QPYのhashとともに固定：

- `DF_0/E/sine`
- `DF_1/E/cosine`
- `DF_1/F/cosine`
- `DF_1/F/sine`
- `DF_3/E/cosine`
- `DF_3/F/cosine`
- `P_0/E/cosine`
- `P_0/F/cosine`
- `P_1/F/cosine`
- `P_2/E/sine`
- `P_2/F/sine`
- `P_3/F/cosine`
- `P_3/F/sine`

`F−R−E+Empty`の全予定診断は未完了で、interaction.jsonは作成されていない。
共通Rだけでは削除によってhalfが隣接しcompiler簡約が効く。単独RをFullへ単純加算しないという設計上の注意を維持する。
次の具体的な修正案は、保存済みinputを使い、processごとに独立したcompiler状態で欠測13件だけを取得すること。新sampling・native再構築・高精度化・既存23件の再compileは不要な案。
この案の実装/再実行には新しいユーザー指示が必要であり、今回のSTOP後に起動していない。


## 実件数・保全・公開証拠

人工検査は新規安定化8＋既存Stage 3 12＋bit-copy回帰1＝21 PASS。科学1batchの内訳：

```json
{
  "scientific_batches_started": 1,
  "T_eigendecompositions": 1,
  "DF_eigendecompositions": 0,
  "native_bases": 11,
  "reference_half_exponentials": 1,
  "sampling_calls": 8,
  "trajectories": 8,
  "event_positions": 256,
  "native_body_programs": 18,
  "wrapper_targets": 36,
  "compile_started": 36,
  "compile_completed": 23,
  "scientific_retries": 1,
  "new_signal_B_N_C_G": 0,
  "finite_bit_mean": 0,
  "SCF_FCI_DF": 0,
  "new_state_reference": 0,
  "user_directed_followup_attempts": 1,
  "automatic_retries": 0
}
```
主な実評価カウンタ：

```json
{
  "T_eigendecompositions": 1,
  "native_basis_lowerings": 11,
  "native_operator_evaluations": 44,
  "reference_half_exponentials": 1
}
```

wall `652.444724`秒、process exit1、全thread worker join済み。`scientific_retries=1`はユーザー指示のrun02、automatic retryは0。
基底や半指数をcacheし、空き資源に応じ独立targetを並列化。実行が長くても正常workerを中断していない。共有設定/環境の変更なし。
旧run01・Stage 1/P1/P2/Stage 2のSOURCE/input/raw/result/grant/ledger/marker/STOP、main/元研究branchは保全した。

[native_evidence.zip](native_evidence.zip)は112 memberの可逆保存。native/、prices/、trajectories/の全bytesを含み、
[archive_members.json](archive_members.json)にmemberごとのSHA256を記録する。QPYはgzipされた原値のまま格納し、展開QPYのhashもprice/native metadataへ対応づけた。
元のlocal filesは保持し、Gitではarchiveへまとめた。展開は空の専用directoryで行い、旧証拠へ上書きしない。
全input/compiled QPYの可逆性・hash・SOURCE/旧STOP保全は[verification.json](verification.json)に記録する。

## 残る交絡・次の担当・mandatory STOP

保存BはDFの方が小さくても、本結果はBと単回価格を結合した同精度資源比較ではない。
finite biasのStage 2 UNRESOLVED、finite-bit law平均、全lawρ、compiled作用精度、母平均費用/tail、shot/RPE全費用は未解決。
artifactに保存したcompiled global phaseは原値のまま。人工測定wrapperのcompiler回帰はobservableで確認しており、H4全compiled operator保証を取得したとはしない。
次の担当はユーザー：processを分離して同じ保存inputの未取得13件だけを再compileする案を実行するか判断する。Codexは新たな明示指示後だけ、その限定回復処理を行う。
**公開・remote照合後mandatory STOP。未取得13件の自動retry、追加標本・新条件・Stage 4・主要claim採択・論文改訂へ移行しない。**
