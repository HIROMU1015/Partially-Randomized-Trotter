# Full-H κ2：固定ceil(3N/2)の1件Q3 preflight

2026-11-15 JST。ユーザーは固定vector・採択済みtarget/law/lossによる最大1新規Q3 callとexact閾値判定を明示許可した。今回指示に従いcount-only包含から必要scopeを実行前に固定した。承認を証明する新たな仕組みは作らない。

## §10.5と物理同一性

既存B branch `codex/rpe-total-dp-acceleration-20261114`、着手HEAD540b54ab2（clean）を継続。元control a82b0f0、2倍条件/結果422db93fc、元signal ce6aacfa、C c628e9e8、統合roadmap518377f1の固定資料・関連local履歴・公開refを確認した。確認範囲では同一Full-H κ2・今回整数vector・decoder/target/law/lossの保存Q3結果は未同定。新namespaceは未存在。自ユーザーの関連job名/cwdを読取確認し、D7価格holdout/publication jobはあったがQ3重複ではなかった。D1/C完了を待たず、それらのprocess/worktreeへ変更を加えない。全非公開資料の不存在を証明したものではない。

元configuration54ac3eb3→total265b90dc、binding579c5de2、全13physical/round ID、元packet/SOURCE/target/law/normalization/scalarは422db93fc INPUTSを正本として保全した。原packetはce6aacfaのFull-H κ2/K2、one-bodyを含む全H、元q/R/grouping/event schedule。既存Full-H routeの`metadata/create/bind_signal`で照合し、同一性を検査した。元rawを再取得せず、scalar/normalizationを再適用しない。R448、PR-DF、Pauli、PFは対象外。

新しく識別するallocationは`FULL_H_KAPPA2_TOTAL_Q3_USER_SCALE_3O2_01`。X/Y各軸の正本整数列は

```text
[96,90,84,78,71,65,57,51,45,38,32,26,18]
```

全roundで`ceil(3*N_control/2)`と一致し、半整数を上へ丸めた。shot policyはUSER_SUPPLIED。control a82b0f0と2倍422db93fcは保存比較に限って再利用し、再評価しない。十分N・最適N・nominal Nは生成しない。

## 科学条件と有限scope

Z1＋T1 total decoder `consecutive_dyadic_nearest_branch_principal_energy_total_v1`、kernel710aa69b、accelerator9e294778を変更せず再利用。保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象loss `(Ehat−Eproxy)^2`、EXACT_AFFINE_OF_BINARY64_RAW、保存ideal finite-K rawとshots/X/Y/round間IID二項モデル、十進εE=0.00159360010199040のexact二乗を使用する。原target資格、Minkowski参照、未認証allowanceを保全し、AδEを加算しない。

各roundのCartesian count数C_mは

```text
[9409,8281,7225,6241,5184,4356,3364,2704,2116,1521,1089,729,361]
```

| 条件 | 実行前固定値 |
|---|---:|
| phase radius（全13round） | 1.0 rad |
| maximum_round_outcomes | 9,409 |
| maximum_verification_pairs | 273,175,586 |
| maximum_frontier_states（非終端のみ） | 8,192 |
| maximum_transition_pairs | 353,666,241 |

既存`finite_bounds`（run_pauli_q3_three_quarter.py）の純整数式で、max C_m、Σ C_(m−1)C_m、C0+8192*Σ C1..12を確認した。全count/隣接組を包含し、grouped outcomes≤C_mかつ非終端frontier≤8192より遷移を包含する。m0新supportは上限C0=9409から始める。旧controlの2593初期状態、2倍の10081 grouped outcomes、旧閉包PASSは今回の証拠として使わない。実際の閉包・mass・残余品質は1call内で評価する。

## 実行経路・人工検証・性能

`run_full_h_kappa2_scale3o2_q3_fixed.py`は実行SOURCE8a8871021の2倍callerから固定INPUTS・Full-H metadata・既存worker/superviseを再利用し、allocation/科学scope/namespaceだけを今回へ接続する。元3候補/2倍executeは呼ばない。実行前commitで新caller/INPUTSを固定する。run01は新規専用で、再実行やresume入口はない。

高速backendは実績のある8worker/batch8を初期設定として再利用し、恒常CPU/RAM/worker上限としない。手法入口docs/PERFORMANCE_METHODS（未導入branchの固定参照）と既存acceleration_casebookを調べ、SOURCE・exact整数集約・binary64順序・pool contextの適合を確認した。観測時64 CPU、available約498,524,804 KiB、load約1.58。共有設定を変更しない。既報の2倍wall約1106秒を新vectorの所要時間保証に外挿しない。

既存の原API return・partials・RESULT・CPU/RSSを**worker自身で保存**する経路を維持。親monitor故障fixtureを再利用し、未開始拒否・開始済み保存を確認した。新callerには数値本体を変えない協調stage observerを付け、伝播開始/終了、その間の集約・次round準備区間、終端処理、DP returnを記録する。親は自worker subtreeのみを資源観測する。

`ObservedExecutor`は同じProcessPoolExecutorを継承し、shutdownで回収した科学worker/DP子processの終了code・signalを記録する。非同期frame dumpやtimer threadを使わない（Python3.11.0rc1の既知faulthandler不具合を回避）。observerと実際のsupervisor→worker呼出しcontextを人工データで確認し、原API outputの完全一致、段階log、終了code0、context復元を確認した。新caller4＋再利用worker5の**9test PASS**。新1.5倍の実H4科学callをtest代わりに呼んでいない。

## gate・保存・STOP

call内で候補固有ρ閉包、round/DP exact質量保存、nonfinite、principal guard、残余loss、数値allowance、上下界とmomentsを確認する。m0含むfrontier除外質量を捨てず、終端へ非終端capを課さない。上界≤τなら条件付きモデル適合、下界>τならモデル不適合、下界≤τ<上界ならINCONCLUSIVE、gate不成立ならUNKNOWN。原API保存状態・process終了/監視状態と数値判定を分離する。

最大1callの終了・同等結果再利用・重大異常でmandatory STOP。無断retry・範囲事後拡大・別倍率/round探索・新signal/sampling/compile/価格/Gは不可。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`、ground/実sampler移送/厳密数値certificate未認証を維持。元control/今回/2倍をSOURCE付き表へ並べるが、単調性・最小N・未評価vectorへの補間は主張しない。ロードマップ本体は更新せず、差分をHANDOFFへ渡す。
