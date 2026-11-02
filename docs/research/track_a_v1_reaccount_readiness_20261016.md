# V1：保存データによる5精度費用再評価の実行契約

終了追記：[5精度再会計・結果と限界・完了STOP](track_a_v1_reaccount_results_20261016.md)。以下は実行前固定の比較・契約の履歴。

利用者の「その方針で進めて」に基づき、Phase BでPASSした固定モデルを用いたone-shot再会計を実行する。対象はlinear H6/1Å/STO-3G/tol-only DF19/12 modes/sector400、T=.8、q1/2/4/8（δ=.8/.4/.2/.1）。新しいsignal・scalar overlay・primitive・native回路・compile・fit・sampling・V2計算は0とする。

[機械可読の計画と入力binding](../../artifacts/resource_applicability/track_a_v1_reaccount_readiness/2026-10-16/README.md)。[再会計runner](../../scripts/track_a_v1_reaccount.py)、[入力materializer](../../scripts/prepare_track_a_v1_reaccount.py)、[人工tests](../../tests/test_track_a_v1_reaccount.py)。[Phase B結果](track_a_v1_phase_b_run01_results_20261016.md)はcommit `66caef169940320eb263d1e6fd8e87d1068d73f2`。

## 固定する比較と入力

ε=.05/.01/.005/.001/.0005、経験的allowance倍率1/10、各軸α=.025、u/headroom≤.01、既存shot算術順序を保持する。主費用はP=0の`N_real*RZ_cosine + N_imag*RZ_sine`、symmetric_directional full-wrapper。compilerはQiskit1.3.0・rz/sx/x/cx・opt1・seed17・topologyなし、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

|scope|決定論候補数|追加する候補|
|---|---:|---|
|old_grid_S2|32|B0/B0+IのS2 p5/10/15と既存full S2/S4|
|fine_grid_S2|96|B0/B0+IのS2 p6–14（旧p10を二重登録しない）|
|fine_grid_S2_plus_truncated_S4|136|さらにB0/B0+IのS4 p11–15|

全scopeにfull S2/S4を含む。scope名のS2は打切り格子の意味であり、旧full S4を比較集合から外さない。通常biasの適格性はprefix単調と仮定しない。正確な有理数のweighted RZ、次にcell ID辞書順で最良を選び、許容差tieは作らない。

PR主比較は既存確認union8・各32のpaired標本平均。探索72候補・各8は別scopeとして保存し、確認標本と混ぜない。PR p3/4は外側S2 biasの診断のままであり、正式候補にしない。PRの標本平均・標本SE・母集団最良の保証を区別する。

入力はPhase A保存104 signal、旧H6保存92 signal、旧B0+I保存12 signal、旧exploration/confirmation wrapper CSV、旧D0会計・B0+I会計・旧比較表、固定model、Phase A/B完了receipt。旧B0+Iは元screen/scalar_signalから既存signal欄だけをcopyし、新たな位相回転やsignal生成をしない。D0入力receiptへ旧rawのbytes/SHAを照合し、全26入力receiptをprivate configへ固定する。NPZ/archiveのdecodeは行わない。

通常biasについて、旧104候補×10条件、Phase A追加104候補×10条件の適格性・shotを保存値へ照合する。旧費用のweighted RZと確認8候補の会計を再照合し、旧比較集合の10条件の最良候補・費用が旧公表表と一致した場合だけ新比較を受理する。不適格を0費用にせず、適格な費用欠測・数値未解決・入力不一致でSTOPする。

## モデルと実測の優先順位

固定model SHAは `406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211`。登録全4軸のPASS・Phase B正常終了・全回収を入力gateに要求する。S2/S4の式・4個の独立stratum・phase/profile区別を維持し、再fitしない。
**保存実測を優先**する。したがってPhase Bで測定したB0 p11q1 S4とB0+I p15q8 S4の4軸は実測を使い、その他の適格な追加候補だけを固定モデルで推定する。モデル推定は`VALIDATED_MODEL_ESTIMATE`、実測は`MEASURED_COMPLETE`、PR有限標本平均は`SAVED_PAIRED_SAMPLE_MEAN`として来歴を残す。

4軸PASSは全prefixに対する2%保証ではなく、旧B0 binary profile同値も未証明。model適用による選択は宣言された条件付き点推定であり、matched B0/B0Iの因果比較や母集団の最良保証ではない。

通常signal biasに加え、保存signed分解の各軸絶対値和による経験的非相殺診断を出す。分解未保存の候補は`UNRESOLVED_SIGNED_DECOMPOSITION`とし、欠けた候補を落とした全domain最良値・費用比は表示しない。厳密bias上界のcertificate、これに基づく保守的適格性は引き続きnull/未解決である。

## 技術検証、実行資源、STOP

pinned PythonによるV1関連71人工tests PASS（今回の再会計14件を含む）。全候補を人工metadataに置換した通し検査、旧比較不一致・Phase A会計変更・不完全Phase Bの拒否、実測優先、exactな近接値/tie、欠測STOP、非相殺分解欠測時の最良null、scope包含、grant/stale gateを確認した。primary agentによるlocal検査であり、外部再現・CI・独立レビューとはしない。

科学計算を並列起動する必要はない。保存値を一度parseしてindex化し、各精度/biasの会計を全scopeで共有する。driver1・observer1、各thread1・別idle CPU。driver AS2GiB/RSS1GiB、observer AS/RSS512MiB、総RSS1.5GiB、CPU300秒、wall600秒、output64MiB、stage256MiB。監視は既存owned launcherの1秒間隔・最大gap5秒、PSI/OOM増分・disk/inode・CPU/thread/lineage条件を継承する。

one-shot pass1、accounting evaluation上限4160、model axis estimate上限208。SOURCE/入力/model/既存結果/grant/ledgerとruntime34,273ファイルを実行前後にhash照合し、fresh gateは30秒以内。旧grant・carryをrefund/resetせず、新しい専用lock・grantだけを使う。

中間精度で旧比較に対するPR削減率の変化が10 percentage points以上、または緩/厳精度でPR優位の符号が変わる場合は結果報告へ明記する。この報告triggerは追加計算・新しい科学条件の自動承認ではない。全表と保全監査を保存し、commit・push・remote SHA確認後mandatory STOPする。V2・H5・他作業と既存結果を保全し、次の実験や再fitへ進まない。
