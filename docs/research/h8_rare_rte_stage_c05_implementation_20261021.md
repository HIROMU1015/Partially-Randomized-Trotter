# H8希少RTE C-0.5：本番adapter実装・技術報告

2026-10-21 JST。**C-0.5実装・人工検証・技術preflight完了、科学実行未承認、mandatory STOP**。実H8の法則計算/sampling/simulation/signal/native回路生成/compile、保存値の新規科学数値解析は0。有効grantの発行・科学one-shot消費は0。

## 完了した実装

C-0 commit `136bd2730e49938bae697ed32674dfcc53e63f07`を継承し、旧設計・旧STOPを書き換えず新SOURCEを追加した。linear H8/1 Å/STO-3G/tol-only DF28/T=.8/P=0、p8q2r4K4とp7q16r4K4、元signed bias/allowance1・analytic shots/PF・native/scalar/control・Qiskit1.3.0/rz,sx,x,cx/opt1/seed_transpiler17を維持する。

|実装|接続・機能|
|---|---|
|[sampling.py](../../scripts/h8_rare_c05/sampling.py)|全支持integer count/CDF、均等位置、旧component正規化、元_make_event/request接続、q2/q16、private seed registry|
|[provider.py](../../scripts/h8_rare_c05/provider.py)|承認後だけimportする原H8 loader・純粋prefix復元・native evolution・full measured wrapper・compiler/fingerprint・原controlへのlazy bridge|
|[adapter.py](../../scripts/h8_rare_c05/adapter.py)|親のlaw/count/component表共有、evolution各1回、同event両軸、call予約、cache、native/会計/final byte gate|
|[parallel.py](../../scripts/h8_rare_c05/parallel.py)|動的trajectory processと軸fork/COW、予算内の統合admission、atomic出力、途中失敗STOP・全所有子孫回収|
|[resources.py](../../scripts/h8_rare_c05/resources.py)|affinity内のphysical core idle、有効quota/memory、I/O pressureをread-only観測し投入数を調整|
|[accounting.py](../../scripts/h8_rare_c05/accounting.py)|明示承認gate・新grantのみのO_EXCL消費、非返還ledger、完全identity cache、paired pilot統計と独立integer moments|

[人工tests](../../tests/test_h8_rare_c05.py)、[成果物packet](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/README.md)。実APIは元のclass/functionへ接続するコードを実装したが、C-0.5ではsymbolic providerで検証し、実H8 packageをimportしていない。人工PASSを本番成功とはしない。

## 数値規約の具体化と未採択事項

[数値law・pilot設計](h8_rare_rte_stage_c05_numeric_pilot_20261021.md)に候補ID、整数weight、PCG64 raw bit消費、正規化、丸め・支持・underflow・旧NumPy choiceとの差、law差分engineering gateを一意に記した。新方式は保存binary64の有理数正規化モデルを抽選する。Bのcanonical有限lawとの費用bias保証やbit stream同一性はない。

`numeric_model_approved=false`。元の実分子法則への新しい数値モデルの正式採用・H8での確率再計算は行わない。law対応の実データ診断は将来C-1の別科学callとして必要で、今回STOP境界に残す。新量子係数・normalization・shot則を採択していない。

## 元nativeと並列実行

親が保存snapshot/state/block exportを一度読み、元の純粋なprefix復元でp7/p8 hashを照合する。元exportの`保存prefix完全一致=false`を保持し、完成済みprefix objectを読むだけとは言わない。SCF/DF/state solver/Gaussianの再生成はしない。decode/prefix復元も科学callとして事前予約する。

親でcount/CDF・component表を準備しCOW共有、workerはevent列とevolutionを一度だけ作る。両軸は同event digest・同evolutionから、測定を含むfull wrapperとして構築/compileする。component順・basis reuse・phase・scalar・control policyは元APIへ委譲する。state preparationは費用外。元の`max_instructions=1000000`は実装STOP guardであり、compiled RZ上界ではない。

外側trajectoryのCPU/memory予約には両軸の需要を含め、内側はその予約の中で1/2軸を動かす。nested processの数だけでCPUを重複割当しない。read-onlyの実CPU idle・physical topology・quota/memory/I/Oから新規投入を制御し、worker数・thread数・RAM/wall timeに固定の低い上限を設けない。未知のprivate peak/thread/IO需要を勝手に0や小値で補わず、承認したbootstrap見積りがなければadmissionを拒否する。現在の空き観測は未来の予約を保証しない。

single-thread fork親を受入条件とする。unsafeならSTOPする。spawn＋mmap/shared codecは大型入力のpickle/再復元を避ける代替候補だが、現在の元Qiskit/preparation objectに同一identityで使えるcodecを未検証のため自動切替しない。この場合は別技術契約が必要。GPUは元compilerを置換しない。

schedulerは自分のprocess groupとPID/start/UID identityだけを扱う。coordinator自身へのLinux child-subreaper指定で、親workerの異常終了時にも所有孫processを回収する。OS全体の設定変更ではない。途中のCPU競合/一時PSIだけで正常workerを強制終了・再起動せず、新規投入を待つ。失敗では部分結果・非返還予約・STOPを残し、retry/resumeなし。

cacheはSOURCE/provider/runtime/library・Hamiltonian/state/snapshot/preparation・q/r/K/delta・完全event digest・scalar/control・axis/measurement・compiler/fingerprint・数値規約が一致する場合だけ使用可能。library封印未完了なので契約のcost cacheは初期無効。同じdrawが重複しても標本行を削除せず、cache hitとcompile missを分ける。高速化率は未測定で、toy所要時間をH8へ外挿しない。

## read-only照合と来歴

C-0報告をGitHubから今回実取得し固定blobとbyte照合。C-0の6 remote取得receiptを参照継承し、関連固定SOURCE33 blobを再照合した。[取得記録](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/source_acquisition.json)。段階B/先行調査/H8 SOURCE/packet/profile補足の固定commitはC-0の来歴を維持し、再調査で上書きしない。

[runtime照合](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/runtime_readonly_audit.json)：記録runtime34,273 fileとnative config provider265 fileは全SHA一致。原P封印のstage489＋provider442＝931 fileも両条件で一致した。同じ内容の重複照合を独立931実験などと扱わない。

初期のhelper相対pathをprovider rootへ解決した試みでは21 locatorが未解決だった。その後、元Pの封印済みstage rootで21件の現在bytesを照合し、固定Gitでも20件を9d18cdd6、h8_ab_signal.pyを後続6197e97dへ対応付けた。欠測/回復の時点を新receiptに残し、旧statusを修正しない。

ファイルbytes一致は新C-1 import解決、SessionEquivalenceLibraryの実mutable状態、loader後のfork安全性、旧compiled出力bit同一性の保証ではない。C-1で新SOURCE/import manifestを固定しfresh gateする。原profileは5f267e96で取得済みという現行状態を保持し、未取得へ戻さない。

[seed除外receipt](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/seed_exclusion_receipt.json)：保存原P registryのexcluded setと96登録taskの元master→step規約から、旧seed除外情報をprivateに準備した。これは既存seed registryの技術的な階層復元で、RTE次数/component/event抽選ではない。新C-1 seedは0、実seed値の公開0。将来registryはSOURCE/input/condition/stratum/phase/replica/epochで封印し、旧集合と新全派生streamの衝突を拒否する。

## 更新契約と必要な選択

[粗精度案](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/contract_H8_0.05_PROPOSED.json)、[細精度案](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/contract_H8_0.0005_PROPOSED.json)はC-0案の参照継承＋今回の更新。PROPOSED/approved=false/runtime_authorization=false、有効grantなし。

第一候補はS2_SINGLEの独立variance/load pilot。全3層pilotを選ぶ案も提示した。multi/S4を未知項として保持し、有限pilot最大値を全support上界にしない。本数は相対分散精度とモデル仮定から決める方式を示し、最終値は未採択。productionの95/99%・費用差1/4または1/2の半幅も選択肢に留め、S0 n48の誤差を含めて将来設計する。

[承認事項](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/approval_requirements.md)は、数値候補の採否、scope、pilot本数/計画信頼水準、control/oracle、法則診断・数値会計範囲、bootstrap footprint、条件別開始順/同時実行、完成SOURCE/manifest/新grant、公開範囲。pilot完了後はSTOP、productionは別承認。

## 人工検証・終了監査

55件PASS。[log](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/synthetic_tests.txt)、[mock native receipt](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/mock_native_receipts.json)。toyのq2/q16×3層、整数count対全word列挙、極小正支持、CDF境界、PCG raw interface mock、元request/measurement/scalar/controlのsymbolic接続、同event両軸、予算並行予約、seed/cache拒否、途中失敗/取消/親急死と孫回収、final byte gate STOP、paired分散対独立raw moments、未承認実行拒否を検査した。

Qiskit・numpy・trotterlib.rteの実importなし。PCG64 raw interfaceはmock wordで検査し、実分子streamを生成していない。担当Codex自身の別実装数値検査で、独立AI科学reviewではない。

[preflight](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/preflight.json)と[保全/構造終了監査](../../artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/completion_audit.json)を保存。旧B運用5file＋既知input82file、旧C-0/B成果物、SOURCE/raw/grant/ledger/marker/STOPを保全。新共有設定/他process・別4次PF研究への変更0、push0、詳細科学review/final freeze0。

専用branch `codex/h8-rare-rte-stage-c05-20261021`へローカルcommitし**mandatory STOP**する。次の担当：**ユーザー**が数値規約・pilot/精度・validation・公開範囲を選択。**Codex**はその判断で完成契約を封印できるが、明示承認までC-1 sampling/compile/実値解析・productionへ進まない。
