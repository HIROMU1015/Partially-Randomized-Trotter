追記（2026-10-19）：利用者がn80/12を正式採用し、[P最終seal](track_a_h2_p_final_seal_20261019.md)を完了した。P科学実行は別途明示承認待ち・未実施。以下はS_STATS終了時点の記録である。

# H2 S_STATS：技術seal・one-shot数値設計・終了監査完了（2026-10-19）

**COMPLETE_MANDATORY_STOP。保存D/Aの2主endpointから、24 scenario・2方向・12感度設定を1 passだけ実行した。**
基準SD倍率4・相対検出差.02の標本数案は、ε=.05が80、ε=.0005が12。
native exit0・同一親wait4・終了監査PASS。P最終seal・P科学実行は未実施である。
これは正規近似による計画値であり、母平均の優位性や実際のpowerの保証ではない。

## 承認と実行境界

利用者は「S_STATS契約を技術準備」を選択し、「数値実行まで進んでよい」と明示承認した。
これをA完了の受入と、H2 S_STATSの技術準備→SOURCE・入力・予算固定→数値one-shot→監査→STOPの承認として記録した。
既存の承認を固定契約へ結合し、追加の確認待ちを設けず、H2所有の新しいnamespaceで実行した。
今回の承認はsampling・compile・raw再集計・bootstrap・法則再構成・P科学実行・pushを含まない。

最初のmetadata準備では、実n/SD/powerの導出を行わず、保存summaryを転記した。
`NormalDist.inv_cdf`、標本数関数、統計再集計関数を拒否し、数値module・科学packageのimportも拒否するguard下で技術sealを完了した。
続いてfresh resource記録・承認receipt・有効grantを作成し、sealed fieldsの不変性を確認して1回だけ起動した。
`S_STATS_run01/ONE_SHOT.json`は消費済み。retry・再起動・次stage自動移行は0。

## 固定SOURCE・入力・共通実装

| 対象 | 固定commit / SHA256 |
| --- | --- |
| H2 S_STATS科学SOURCE | `e0b959e8a617f7934d05c3e249832ccc1c43909f` |
| そのまま再利用したH4数値kernel SOURCE | `8752489bde3903d80d306c4500535c79ac65ef31` |
| A完了 | `1d2f67ec285956ef4090ea26520d64b11e925f9c` |
| A科学SOURCE | `de72e5d4e48177734c0ddd79dffac14222410658` |
| A result SHA256 | `898018725d8c50b647cdbc1f3fef006037fd2315bb748503d33a188b2bc86bfb` |
| D完了 | `7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb` |
| D科学SOURCE | `cf99e4452df96291d7e9d37d76d2e91be5cbc413` |
| D result SHA256 | `a7d75179415b14f21ec7e77ddc3e95b4dc5c7f406d664fddbf0755431f19454c` |
| 今回のS_STATS result SHA256 | `33e48667c4bb0b3bafc38d4f7a425d217053d1c8191ed54da078e301e5886a27` |

H4の数値kernelのfile bytesと関数codeを再利用した。H2専用adapterは独立した関数namespaceへH2契約を結合する。
H4のmodule globals・SOURCE・契約・結果・cohortや、他worktreeを変更しない。
H2 Aの保存PR/PF比を旧kernel用の方向fieldへ変換する処理は、承認済み数値pass内だけで行った。
H4のrank10・候補数・n1108/78を移植していない。

SOURCE closure17 files、固定入力13件、stdlib runtime269 filesとPython executableのbytesを起動前後に照合した。
入力13件は、Git固定のA/D各result・終了監査・native exit・execution binding・公開manifestの10件と、
選択summary・準備承認receipt・旧証拠保全manifestの3件である。
Pythonは専用環境の固定executableを`-B -S`で使用し、数値package・compilerは呼び出さない。
F/FP/D/AのSOURCE・入力・raw・grant・ledger・marker・STOPに属する旧7,219固有レコードを保持した。

## H2条件と固定2 endpoint

linear H2、結合距離1.0 Å、STO-3G、tol-only DF1e-8による自動rank3、4 qubits。
Nalpha=Nbeta=1のspin sector次元4。受入FのHamiltonian・状態・DF分解は再生成しない。
T=.8、状態準備費用P=0、通常signed bias、allowance倍率1。
保存費用はfull Hadamard wrapperのcosine/sine各compiled RZ×各必要shotの合計である。
保存compilerはQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing未指定。
snapshot・Hamiltonian・状態・compiler hashは公開`selected_saved_summaries.json`のbindingに固定した。

| ε | 固定PR候補 | L_D / q / r / K / delta | 保存PF対照 | 保存PR平均（exact） | 保存paired SE | 保存PF費用 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| .05 | B2_p1_q1_2nd_r2_K4 | 1 / 1 / 2 / 4 / .8 | full PF B1_p3_q1_2nd、S2 | 9570223/2 | 31086.308466452374 | 5,533,984 |
| .0005 | B2_p1_q4_2nd_r2_K4 | 1 / 4 / 2 / 4 / .2 | full PF B1_p3_q4_2nd、S2 | 497311863209/2 | 660361277.5861268 | 305,695,076,994 |

両方とも保存A winnerの新`EXPLORATION_N8`であり、FP診断n1・旧H2・H4等の標本は混合しない。
PFはDの保存対照を使い、PF再探索・再compile・shot再計算・候補の再選択を行わない。
数値passで保存したPF−PRのexact差はそれぞれ`1497745/2`、`114078290779/2`、方向は両方+1。
観測された差を検出差の代わりに使わない。

## 固定した正規近似設計と保存結果

family α=.05、2主endpointに対する片側Bonferroni α/2、設計power=.8。
保存paired SEから`s_plan = SE_saved × sqrt(8)`とし、
`n = max(2, ceil(((z_(1-alpha/2) + z_power) × f × s_plan / (eta × G_PF))^2))`を用いた。
SD倍率f=1/2/4/8、相対検出差eta=.01/.02/.05、n上限なし。
基準f4/eta.02は数値を見る前に登録し、結果を見て選び直していない。

以下は保存resultの転記である。報告・監査でn/SD/quantile/powerを再計算していない。

| SD倍率f | 相対検出差eta | ε=.05のn | ε=.0005のn | P軌道数案 | P compile軸数案 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | .01 | 20 | 3 | 23 | 46 |
| 1 | .02 | 5 | 2 | 7 | 14 |
| 1 | .05 | 2 | 2 | 4 | 8 |
| 2 | .01 | 80 | 12 | 92 | 184 |
| 2 | .02 | 20 | 3 | 23 | 46 |
| 2 | .05 | 4 | 2 | 6 | 12 |
| 4 | .01 | 318 | 47 | 365 | 730 |
| **4** | **.02** | **80** | **12** | **92** | **184** |
| 4 | .05 | 13 | 2 | 15 | 30 |
| 8 | .01 | 1269 | 188 | 1457 | 2914 |
| 8 | .02 | 318 | 47 | 365 | 730 |
| 8 | .05 | 51 | 8 | 59 | 118 |

基準案のP予算はtrajectory/native evolution92、full wrapper compile184軸、occurrence128、RTE event256、
control最大8、独立event oracle最大2、seed slot140、endpoint会計1 pass・比較2である。
PF compile・signal・SCF/DF/状態/reference・bootstrap・shot再計算は各0の案。
これは将来のP契約への入力であり、P最終n・P SOURCE・独立seed・有効P grantは未固定・未生成。
`P_after_S_STATS_DRAFT.json`は`PENDING_USER_ADOPTION_AND_FINAL_P_SEAL`である。

## 希少層・統計的限界

選択された2候補の保存stratumはそれぞれ`{"S0_ONLY":8}`、K4である。
法則確率、希少層を観測するためのn、未観測stratum費用は全てnullのまま保持し、0埋めしていない。
今回、sampler lawからの確率再構成やvisibility設計は実行していない。
f4/f8は未観測tailの分散上界ではなく、この正規近似案が実power=.8を達成するとは認定しない。
独立Pを実行したとしても、この制約の評価は別に必要である。
母平均優位・選択後coverage・厳密bias上界・大域最適性・CI/p値の主張はしない。

## 予算・テスト・終了監査

| 科学call | 固定予算 | 実行 |
| --- | ---: | ---: |
| 数値設計pass | 1 | 1 |
| 正規近似scenario | 24 | 24 |
| 固定PFとの符号比較 | 2 | 2 |
| 感度設定 | 12 | 12 |
| raw再集計 / bootstrap / sampling / compile / signal / shot再計算 | 各0 | 各0 |
| 候補再選択 / FP・旧cohort混合 / P / 新seed | 各0 | 各0 |
| SCF・DF・状態・reference生成 / law再構成 / 希少visibility n | 各0 | 各0 |

人工21件＋既存H4数値回帰10件の計31件がPASS（failures/errors0、実D/A統計の読込なし）。
実pendingの無効grantはdirect/witness両入口で数値importより前に拒否し、output・one-shot markerを作成しなかった。
人工fixtureではceil・最低n・方向・H2 q1/q4のP call・不正cohort/入力・再実行拒否・STOP保全・監査の数値非実行を検査した。

native wallは`1.757040023803711`秒、数値phaseと後検査は`0.8308109194040298`秒、peak RSSは32,931,840 bytes。
同一親wait4でexit0を回収し、terminalは`COMPLETE_MANDATORY_STOP`。
24件のスカラー処理なので1 processを使い、CPU/RAM/worker/wallの恣意的上限は設けない。
空き資源を観測し、PSIは診断として記録した。共有設定・他job・他worktreeを変更しない。

終了監査は`audit_h2_s_stats_completed.py`のstdlib metadata検査で、数値kernelをimportしない。
数値関数・科学module importを拒否するguard下で、SOURCE/input/runtime・承認・grant・marker・台帳・native回収・保存scenarioの対応を検査しPASSした。
提案P callの整数恒等式は検査したが、標本数式・quantile・SD・powerは再実行していない。
主担当Codex自身の終了監査であり、別reviewerによる独立科学レビューではない。

公開用[準備証拠](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/S_STATS_preparation01/)と
[完了証拠](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/S_STATS_completed_run01/)を保存した。
準備・activationの保全manifestと旧7,219レコード、今回のattempt/activation16 private filesを照合した。
有効grant本体・元raw/log・runtime明細はprivateに保持し、新規pushは行わない。
guardの実行記録はprivateの`S_STATS_seal_stdout01.txt`と`S_STATS_audit_stdout01.txt`に保持する。
旧STOPと非返還予約を保持し、自動retry・科学中断・再起動・P科学実行・詳細科学レビュー・最終evidence freezeは0。

## 比較可能性と次の担当

H4と同じ正規設計・f/eta感度・会計規則を使い、H2固有のrank3・sector4・候補・保存SE/PFから設計した。
H6/H8を含む既存結果・cohortは変更・混合しない。H2はS_STATS完了まで到達し、P最終sealが残っている。

| 担当 | 次の作業・判断 |
| --- | --- |
| ユーザー | n80/12の基準案と希少tailの制約を確認し、P最終sealに用いるnを採用するか指示する。公開する場合はpush対象を指定する。 |
| Codex | 採用指示後、独立seed衝突検査・P SOURCE・候補/PF対照・予算・one-shot契約を最終sealしてSTOPする。Pのsampling/compileは行わない。 |

現在はmandatory STOP。P最終seal・P科学実行への自動移行は行わない。
