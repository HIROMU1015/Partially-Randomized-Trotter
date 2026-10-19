# H2 A66：FP・D依存結合と最終seal（2026-10-19）

**FINAL_SEAL_MANDATORY_STOP。Aの科学実行は未承認・未実行。**
利用者の「既存のA66契約にFP・Dの依存SHA、結果監査、保存PF対照を結合し、seed衝突・SOURCE・call予算を再確認して最終sealして」に基づき、
元A66の科学SOURCE・候補・n8・compiler・予算を維持したまま依存証拠を固定した。
この指示はD完了結果のA依存入力としての受入に結合し、A実行・公開pushの承認には用いていない。
有効grant・A activation・A attempt・ledger・one-shot markerは作成していない。

## 固定SOURCE・入力・候補

| 対象 | 固定値 |
| --- | --- |
| A科学SOURCE | `de72e5d4e48177734c0ddd79dffac14222410658`、既存23 filesを無変更 |
| seal用metadataツールSOURCE | `136e9f78142c3b73f59a0685471d9889af57605a`、新ツール・人工テストの2 files |
| F完了commit | `7ea6df2dbe34e64a34f468c343bb12922d3e2829` |
| FP完了commit | `af724928689be3592b391fab3a14538bf4752b76` |
| D完了commit | `7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb` |
| D再開科学SOURCE | `cf99e4452df96291d7e9d37d76d2e91be5cbc413` |
| private最終契約 | `.h2_pr_resource_private/2026-10-19/A_seal01/A_spec_FINAL_PENDING.json` |
| scientific attempt予約先 | `.h2_pr_resource_private/2026-10-19/attempts/A_run01`、未作成 |

H2専用branchのremote SHAを読取確認し、D完了commit `7b202a1a…` と一致した。
各FP/D正本は保存private結果・公開用copy・完了commitのGit blobでSHAが一致する。
今回の新しいseal成果はH2 branchへlocal commitし、新規pushは行わない。

対象はlinear H2、距離1.0 Å、STO-3G、tol-only DF1e-8の自動決定rank3、4 qubits、
Nalpha=Nbeta=1のspin sector次元4。保存F状態の再生成・再正規化は行わない。
T=.8、P=0、ε=.05/.01/.005/.001/.0005、通常signed bias、allowance倍率1。
full Hadamard wrapperのcosine/sine各compiled RZ×各必要shotの和で会計する。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing未指定。
symmetric directional、original scalar＋tail identityは1回だけ、full-H referenceは無回転。

Hamiltonian hashは `d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80`、
state hashは `080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf`、
snapshot SHA256は `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa`、
compiler profileは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
SOURCE23・provider262・固定A kernel4・runtime12,202 filesを照合した。

元の[FP/D/A事前登録](track_a_h2_fp_d_a_preparation_20261019.md)のA66を変更しない。
B2のsplit L_D=0/1/2、主集合はr4・q1/2/4/8/16・K2/K4の30条件。
補集合はr2/r8・q1/4/16・K2/K4の36条件。delta=T/q=.8/.4/.2/.1/.05。
適格な各候補には独立な新n8、両axisには同じ軌道を使う。
FPのPR n1はsignal/referenceの同一入力での再利用に限り、Aの費用標本へ混合しない。
候補間CRNなし。空tailは不適格として置換せず、不明な適格性はSTOPとする。
full-rank空tailをAへ追加せず、境界からの自動探索拡張も行わない。

## 依存結果・監査・保存PF対照

| 依存 | result SHA256 |
| --- | --- |
| FP | `0b43326c03b1e6786841920d0f38ad5db01fe5c22d9e1172e2354b87987bf80f` |
| D_run02 | `a7d75179415b14f21ec7e77ddc3e95b4dc5c7f406d664fddbf0755431f19454c` |
| 保存PF対照copy | `2695489d241aee8d6b4f75911a89c42231516df4b68a3c7afeed5e5720a79034` |

FP/Dそれぞれのresult、runner completion audit、保存bytes監査、native exit、受入receiptを固定SHAで結合した。
依存のstatus・stage・SOURCE・同一Hamiltonian/state/compiler binding・native exit0・同一親wait4も確認した。
元A pending契約・旧入力7件を保持し、新しい最終pendingは合計28入力に結合した。
実際に参照するdependency辞書も別の固定inputへ保存し、SOURCE/入力/依存先のすり替えを拒否する。

FPの保存完了監査と204 private filesはそのまま再利用した。
旧FP auditorには「D未起動」の時点限定gateがあるため、そのaudit全体を現在へ適用していない。
Dは既存のstdlib完了auditorで再監査し、新たな会計やPF順位計算は行っていない。
D_run01のSTOP・541 files・旧56軸予約は維持し、未受理raw compileを依存結果へ混合しない。
F295レコード、D_run02完了314 private files、元準備sealもhash照合した。

保存PF対照はDの既存winnerをそのままcopyし、元350行への所属と5精度の対応を検査した。
以下は保存値の転記であり、新たなPF探索や再会計ではない。

| ε | PF対照（full PF、L_D=3、S2） | 保存総compiled RZ×shot |
| --- | --- | ---: |
| .05 | B1_p3_q1_2nd | 5,533,984 |
| .01 | B1_p3_q1_2nd | 188,156,192 |
| .005 | B1_p3_q2_2nd | 1,170,534,707 |
| .001 | B1_p3_q4_2nd | 59,673,629,128 |
| .0005 | B1_p3_q4_2nd | 305,695,076,994 |

これはD登録内の対照であり、大域最適性・厳密bias上界を認定しない。

## seed再確認

既存A registryの528 master・3,888整数を変更せず、SOURCEとplanから導かれる登録値と照合した。
既存の72,080整数とFP診断4整数、計72,084整数との衝突は0。
FP時点のfresh exclusionsは自分自身のA予約を含むため、その集合をそのままAの除外集合にはしない。
旧除外集合＋FPの4整数を用い、A自身との誤った自己衝突を避けた。

既知27件のseed originを既存H2 snapshotとread-onlyで再照合し、全件SHAが不変。
geometry登録もSHA不変。元A/FP共通衝突proofが同じA seed digestを包含することを確認し、
6 geometryの7,737,216予約slotとの既存衝突0の証拠を再利用した。
大量の派生seed列の再生成・RTE event sampling・seed redrawは行っていない。

この検査の対象は登録済み既知campaignである。将来の未登録campaignを包含せず、
衝突0を確率的独立性の証明としない。A実行承認後のactivationでは新しい観測receiptで再確認する。
他worktreeのHEAD/statusは凍結条件ではなく、その通常更新や一時的PSIで科学workerを止めない。

## 固定科学call予算

以下は将来のA一回実行に対する上限。**今回の実科学callは全て0。**
適格候補数・実際の軌道数はA signal検査後に確定し、未判定のまま予算を縮小していない。

| category | 固定上限 |
| --- | ---: |
| preparation | 3 |
| primitive probe / spectrum | 90 / 4 |
| 独立tail matrix | 3 |
| 新base signal / independent forward | 64 / 64 |
| 新signed shot endpoint | 320 |
| trajectory / native evolution | 528 / 528 |
| occurrence / RTE event | 3,504 / 16,032 |
| Hadamard wrapper compile軸 | 1,056 |
| control probe / independent event oracle | 264 / 66 |
| endpoint統計会計pass | 1 |
| SCF / 積分 / DF / 保存状態solver / 実quantum shots | 各0 |
| full-H reference / primary column / independent column / truncated reference | 各0 |
| B0I overlay | 0 |

90 primitive probeは既存providerの純stdlib時間scheduleから構造的に再確認した。
新しい分子行列やsignalは生成していない。FPと一致する2候補の保存signalを再利用するため新baseは最大64。
全適格候補のcompile費用は新n8とし、FP n1を混合しない。
候補間・軌道間の計算は実際のaffinityと空き資源に応じて並列化し、
CPU/RAM/worker/wallへの恣意的上限を追加しない。実資源のadmissionは実行承認後に行う。
既存OS制御や共有環境・他jobは変更しない。

## 人工検証・無効grant拒否

新しい人工metadataテスト21件がPASS。Python `-S` のstdlibだけで実行した。
旧D未完了、sector/hash/SOURCE/回収の不一致、PF copy改変、候補・予算・旧入力の変更、
seed衝突・geometry proof不整合を検査した。
既存の人工gate fixtureを再利用し、異なるD再開SOURCEの完了結果をA依存に結合できることも確認した。
元SOURCEの人工54件は保存証拠を保持し、無変更部分を再実行していない。

実際の最終pending契約＋INVALID grantでdriver/worker/adapterの3入口が科学import前に拒否された。
SOURCE・予算・dependency・seed registryの4種のin-memory改変も拒否された。
numpy/scipy/qiskit/pyscfのimport、NPZ decode、sampling、compile、費用再会計は0。
この検査は主担当Codex自身のmetadata確認であり、独立科学レビューではない。

成果物は
[`A_seal01`](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/A_seal01/)の15 files。
最終spec SHA/digest、28入力、依存結合、PF対照、seed receipt、人工検証・拒否検査を保存した。
最終pending spec SHA256は `4baa3f2eabca0538b7ac01f377acf205b00b44212904040934eac717bb1a4687`。
公開用の固定範囲は[`A_final_preregistration.json`](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/A_seal01/A_final_preregistration.json)を参照。
元のSOURCE・F/FP/D結果・grant・ledger・marker・STOP・seed registryは上書きしていない。

## 残る承認と次の担当

最終pendingに残るのは、Aのみの明示承認、fresh resource admission、activation時のseed再確認の3項目。
これは未解決の科学仕様ではなく、実行前に必要な承認・観測条件である。
grantには最終spec全体のdigestと実行承認receiptを結合し、依存辞書・入力・候補・n・予算を変更しない。
one-shot claim後の自動retryは禁止し、異常時は証拠と非返還予約を保持してSTOPする。

| 担当 | 次の作業・判断 |
| --- | --- |
| ユーザー | 本最終sealのSOURCE・入力・予算によるAだけのone-shot実行を承認するか判断する。公開する場合はpush対象を指定する。 |
| Codex | Aの別途明示承認後だけ、最終sealを照合してfresh resource/seed receiptとgrantを結合し、既存H2 runnerでAのみ1回実行・終了監査してmandatory STOPする。 |

A結果・PR対PFの新比較・S_STATS・P科学実行・詳細科学レビューは未実施。
P最終sealへの自動移行や論文最終evidence freezeには進まない。
