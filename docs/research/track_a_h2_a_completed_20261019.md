# H2 A66：新n8のone-shot実行・終了監査完了（2026-10-19）

**COMPLETE_MANDATORY_STOP。Aを1回実行し、native exit0・同一親wait4・592 worker全回収・保存bytes監査がPASS。**
登録66候補の各新n8、528軌道・1,056 compile軸と330会計行を保存した。
S_STATS・P科学実行・詳細科学レビューには進んでいない。新規pushは行わない。

## 承認・SOURCE・入力

利用者は「承認後に資源・seedを再確認してAを実行・監査」を選択し、
「次の作業に進んで大丈夫」と承認した。
[最終A66契約](track_a_h2_a_final_seal_20261019.md)のAのみの実行・監査に結合し、
H2所有の `A_activation01` と未使用の `attempts/A_run01` を用いた。
one-shot markerは消費済み。自動retry・再起動・次stage自動移行は0。

- 科学SOURCE：`de72e5d4e48177734c0ddd79dffac14222410658`、23 filesを無変更。
- 最終seal commit：`a99b23ce270491a52623976ed5e5714e13beed94`。
- seal用metadata SOURCE：`136e9f78142c3b73f59a0685471d9889af57605a`。
- 最終pending SHA256：`4baa3f2eabca0538b7ac01f377acf205b00b44212904040934eac717bb1a4687`。
- F完了：`7ea6df2dbe34e64a34f468c343bb12922d3e2829`。
- FP完了：`af724928689be3592b391fab3a14538bf4752b76`。
- D完了：`7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb`、D科学SOURCEは `cf99e445…`。

linear H2、結合距離1.0 Å、STO-3G、tol-only DF1e-8の自動決定rank3、4 qubits。
Nalpha=Nbeta=1のspin sector次元4で、受入済みFの状態を再生成・再正規化しない。
T=.8、P=0、ε=.05/.01/.005/.001/.0005、通常signed bias、allowance倍率1。
full Hadamard wrapperのcosine/sine各compiled RZ×各必要shotを合計する。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routingは未指定。
symmetric directional、original scalar＋tail identityは1回だけ、full-H referenceは無回転。

Hamiltonian hashは `d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80`、
state hashは `080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf`、
snapshot SHA256は `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa`、
compiler profileは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
固定SOURCE23・入力28・provider262・A kernel4・runtime12,202 filesを起動前後に照合した。
旧H2のrank指定DF・sector6は混用していない。

## 登録集合と再利用

B2のsplit L_D=0/1/2。主集合はr4・q1/2/4/8/16・K2/K4の30条件、
補集合はr2/r8・q1/4/16・K2/K4の36条件。delta=T/q=.8/.4/.2/.1/.05。
境界拡張・候補置換は行わず、登録66候補全てが少なくとも1精度で費用取得適格となった。
各候補に新n8を取得したが、全330 endpointが適格という意味ではない。

FPと一致する2候補のsignalおよびfull-H referenceは保存値を再利用した。
FPのPR各n1は診断cohortのまま保全し、Aの費用標本には混合していない。
Aの528測定pairは全て `EXPLORATION_N8`。各候補のreplica0～7は独立した登録masterを使い、
同一軌道のcosine/sineをpairにする。候補間CRNは用いない。
Dの5精度PF対照は保存resultから参照し、PF再探索・再会計は行わない。

## 固定予算と実行照合

| category | 保存された非返還予約 | 固定上限 |
| --- | ---: | ---: |
| native preparation | 3 | 3 |
| primitive probe / spectrum | 90 / 3 | 90 / 4 |
| 独立tail matrix | 3 | 3 |
| 新base signal / independent forward | 64 / 64 | 64 / 64 |
| 新signed shot endpoint | 320 | 320 |
| trajectory / native evolution | 528 / 528 | 528 / 528 |
| occurrence / RTE event | 3,504 / 16,032 | 3,504 / 16,032 |
| Hadamard wrapper compile軸 | 1,056 | 1,056 |
| control probe / independent event oracle | 264 / 66 | 264 / 66 |
| endpoint統計会計pass | 1 | 1 |
| SCF / 積分 / DF / 保存状態solver / 実quantum shots | 各0 | 各0 |
| full-H reference / primary column / independent column / truncated reference | 各0 | 各0 |
| B0I overlay | 0 | 0 |

spectrumは実際の登録fragment集合に対して3を予約し、上限4以内で完了した。
未予約の上限枠を返還したとは扱わず、上限を埋める追加計算は行っていない。
必要shotは資源会計用の保存値であり、測定shotを生成していない。

## 保存された5精度の点比較

330行の会計と比較は承認されたA内で1 passだけ実行した。
以下は保存resultの転記であり、終了監査・報告時の新順位計算ではない。
PRのwinnerは全てL_D=1、r2、K4。PF対照はfull PF、L_D=3、S2、q1/1/2/4/4である。

| ε | 保存PR winner | PR総compiled RZ×shot（exact） | 保存PF費用 | PR/PF（保存値の表示） |
| --- | --- | ---: | ---: | ---: |
| .05 | B2_p1_q1_2nd_r2_K4 | 9570223/2 | 5,533,984 | 0.864678 |
| .01 | B2_p1_q1_2nd_r2_K4 | 162650064 | 188,156,192 | 0.864442 |
| .005 | B2_p1_q1_2nd_r2_K4 | 1039663335 | 1,170,534,707 | 0.888195 |
| .001 | B2_p1_q4_2nd_r2_K4 | 48549488218 | 59,673,629,128 | 0.813584 |
| .0005 | B2_p1_q4_2nd_r2_K4 | 497311863209/2 | 305,695,076,994 | 0.813412 |

全5精度で保存されたPRの点推定はPF対照を下回った。
これは登録集合・通常signed/factor1・新n8における有限標本の点比較である。
PRのpaired SE・sample SD・候補内endpoint共分散もresultに保存しているが、
母平均の優位性、厳密bias上界、大域最適性、選択後coverageを認定しない。

選択された2候補の保存stratumはそれぞれ **S0_ONLY 8/8**。
希少S2/S4の費用寄与やtailを、このn8から保証しない。
これはwinnerの2 cohortに関する記録であり、A全候補がS0のみという意味ではない。
S_STATSのn/SD感度・power設計、独立Pの確認結果は未取得である。

## native終了・seed・保全監査

- native exit0・同一driverによるwait4回収。wall `75.48869092576206` 秒。
- native単一processのpeak RSS：401,555,456 bytes。worker合計RAMの値ではない。
- signal64、first-control66、remaining462、計592 workerが全てexit0・回収済み・result hash一致。
- control最大誤差：`2.0293182708196087e-14`、登録閾値1e-9未満。
- primitive30時間×3状態＝90 probe、native/expm誤差 `6.216689976047745e-16`、spectral/expm誤差 `3.384929468473829e-16`。
- driver/worker STOPなし、自動retry・科学runの中断・再起動0。

実行前のaffinity64 CPU、観測された低負荷候補62 CPU、MemAvailable 492,036,472 kBを用いた。
既存owned poolがwaveのtask数と実CPU候補数に応じて並列化し、内部thread1で多重BLAS並列を避けた。
CPU/RAM/worker/wallへの恣意的上限、OS制御の変更・回避、他jobや共有環境への変更は行わない。
既知PIDの観測だけを行い、無関係なworktreeのHEAD/statusや一時的PSIをSTOP gateにしない。

登録528 master・3,888整数は不変。activation直前に既知27 seed originとgeometry登録を再確認し、
旧72,080＋FP4整数との衝突0、geometry7,737,216予約slotの不変proofを確認した。
redrawは0。各保存trajectoryのseed・event digest・q/rのlayout・両axisのfingerprintを照合した。
将来の未登録campaignを包含せず、衝突0を確率的独立性の証明としない。

終了監査は `scripts/audit_h2_a_completed.py`。Python `-S` のstdlibのみで、
固定spec/grant・依存SHA・台帳・worker identity/reap・cohort・保存330行とPF対照のjoinを検査した。
NPZ decode、signal/sampling/compile、paired統計・会計の再実行は行わない。
これは主担当Codex自身の保存bytes監査であり、別reviewerによる独立科学レビューではない。
元科学SOURCEの人工54件・最終sealの人工metadata21件と拒否検査のPASS証拠も保持した。

F295レコード、FP204 files、元D541 filesとSTOP・旧56軸予約、D_run02の314 files、A最終seal9レコードを保全。
今回のA完了5,849 private filesを新manifestへ固定し、監査codeのcopyも保存した。
公開用結果は[`A_completed_run01`](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/A_completed_run01/)の23 files。
有効grant本体・F snapshot・worker raw/logsはprivateに保持し、新規pushは行っていない。
`result.json` SHA256：`898018725d8c50b647cdbc1f3fef006037fd2315bb748503d33a188b2bc86bfb`。

## 比較可能性と次の担当

H4/H6/H8と会計・compiler・native control・通常signed/factor1の規則を共有する。
H2は固有のrank3・sector4・A66を用い、H4のrank10・A50・P標本数を流用しない。
H2のD/Aまで完了したが、S_STATS/Pと同じ証拠段階にはまだ達していない。

| 担当 | 次の作業・判断 |
| --- | --- |
| ユーザー | A完了結果の正式受入と、S_STATS技術準備へ進むかを判断する。公開する場合はpush対象を指定する。 |
| Codex | A受入後、ε=.05/.0005の保存候補・PF対照・SD感度を用いるS_STATS契約と解析予算を固定する。S_STATS数値実行はその個別明示承認後だけ行う。 |

現在はmandatory STOP。S_STATS・P科学実行・詳細科学レビュー・論文最終evidence freezeは実施していない。
