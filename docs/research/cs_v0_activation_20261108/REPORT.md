# C-S V0：採択条件・最終activation・科学SOURCE固定

2026-11-08 JST／SERVER1 GPUA100-gleap-01。
**V0条件は今回採択済み、最終activation完成。技術状態TECHNICALLY_READY（実H4 preflightは初回実行時）。科学実行は未許可・0 campaign／0実H4 call。公開照合後mandatory STOP。**

## 既存SOURCEで足りなかった範囲

基点606c40dの各関数は利用できるが、旧実行草案に「薄いserializerから呼ぶ／自動科学CLIなし」と明記され、Stage A–D全体の呼出し・保存・例外STOPは未接続だった。この不足だけを埋める `cs_v0_execution.py` と `run_cs_v0.py`、追加人工testを作った。元のcs_v0.py/cs_v0_native.py・operator/law・synthesis provider・価格器は変更しない。

既存Parent科学pipelineは登録G9 operator・旧実験の契約/marker/資源設定を前提にし、このH4 V0の起動入口として無変更では利用できない。そのrunnerを再開せず、operator_target、Calls、atomic_json、native.acquire/local_native/native_accounting、old static preflight、CPU tick parserだけを再利用した。

新規関数のlocal comparison/finite resultは、既存strict RZ target式の2×2interval serializationとoperator_targetへのcallback接続である。新しい物理operator engine/solverではない。旧float/Spectrum経路を混ぜず、実H4への呼出は0。

## 人工検証と重複排除

新activation用**10件が初回PASS、0.293秒**。完了順序、跨specのkey dedup、negative actual adjointの引渡し、保存、default CLI不起動、preflight/scope/finite/local cap STOP、guard失敗時のraw先行保存、保存例外・中断のprefix/UNKNOWN、再実行拒否を確認した。

その後、新SOURCE/remoteと旧preflightを結ぶreturn metadataを仕上げ、対応する人工preflight test1件だけ追試PASS（0.003秒）。10種類の人工test、2 invocationであり11種類の独立検証としない。実backendはmock、実H4入力不使用、人工以外のoperator/合成/価格0。

[主log](ACTIVATION_TESTS.log)、[局所追試](PREFLIGHT_CONNECTION_FOLLOWUP.log)。旧18件・9件は型/値/SOURCE不変で再利用し、再実行なし。Parent/RTE/SP/F1/F2/P3/SCV/C1等の旧suiteも0。旧preflightの同条件実データ再実行0、provider再install/再合成0。OS資源のread-only snapshotを一回確認しただけで、新scienceや他processへの操作はない。

## 固定した条件と初回に残るもの

保存linear H4／1Å／STO-3G／legacyDF12／8system／全4電子参照／L_D3、m0,q1,δ.02,r1,K2/coefficient_atol0、N1 actual Q／selected fullを維持する。

ユーザー採択値を[実行契約](V0_EXECUTION_CONTRACT.md)へ固定：first D/R factor0各1、π保持nearest1/256rad/tie-away、residual r1/K2/root256/proposal160/ηρ10^-12、strict pygridsynth2.0.0/seed0/dps100/10^-6、finite cap10^-12/local cap3×10^-6、unique keys≤8、campaign≤1/retry0。新しい科学値への変更はない。

実H4のslot適格性、strict acquisitionの成立、finite/local cap・局所価格は未取得。初回Stage Aで新SOURCE/remote・13入力bytes・runtime/既存profile・fresh output/cache・空き資源/非干渉を照合する。旧engine保証の変更後C-Sへの継承、full価格/shot/N/Gは保証しない。OS強制消失時の最終STOPは未保証で、未返却callはUNKNOWN・再実行なし。

## 公開・保全・次の担当

base=`606c40d7809a5941a76c54d0bc2dd8e1a5bc1818`、branch=`codex/cs-v0-activation-20261108`、worktree=`/tmp/cs-v0-activation-20261108`。最終科学SOURCEは本directoryを公開したcommit全treeで、最終チャットのSHAを正本とする。SOURCE.jsonに実装・契約hashと既存input/profile bindingを固定した。

元main HEAD/status/3,087既存files、旧606c40d技術SOURCE、N1/N2/engine、SCV raw、旧result/grant/ledger/marker/STOP、F1/F2/P3/Parent/SP/PS48/C1凍結を保全。変更は新activation・人工資料と索引/statusの追記だけ。共有OS/venv/cache/他jobの変更0。

GitHub APIで専用branchへ公開し、remote SHA/parent/tree・全変更本文/blobを照合する。今回採択は実行許可ではないため、本番outputの予約・起動も0。

次担当はSERVER1 Codex。必要な明示許可の対象は、**この最終固定SOURCEと採択済みV0条件で、最大D1/R1・8positive keys・1campaignの限定科学callを一回実行すること**。条件の再採択や承認証明/grantは不要。許可までは実H4 binding/operator/finite mean/strict acquisition/local priceを開始しない。

**Mandatory STOP：技術activation・人工検証・公開照合までで終了。V0科学実行、V1、他研究へ自動進行しない。**
