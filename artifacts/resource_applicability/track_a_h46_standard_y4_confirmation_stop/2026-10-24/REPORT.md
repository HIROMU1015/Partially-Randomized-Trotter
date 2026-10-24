# H4/H6 標準Yoshida4 独立確認：技術エラーで停止

**承認された一回の実行を開始したが、追加したruntime adapterの参照不備により両系ともworker入口で停止した。親contextの準備は両系で完了したが、新Y4 trajectory・sampling・compile・費用会計は0。方向の独立確認は未評価であり、成功やINCONCLUSIVEとは判定していない。科学retryは行わずmandatory STOPした。**

## 実行と消費範囲

2026-10-24、利用者の「この固定契約のD・E・F一括実行のみ承認する」を受領し、元の固定契約をそのまま一回使用した。

| 項目 | 固定値 |
|---|---|
| 科学SOURCE | `a94735946e8cf15ffea5c93ae99000f7ab83d23a` |
| exact契約SHA256 | `6d08c2531bc009042fb58926b7eaa5d45e26348457a865783d825a2af97df794` |
| 入力manifest SHA256 | `de1996e3d9317074c2a8a8ad0037d87cf51f0dcf3baf94cde1b0d20348ad9e80` |
| 承認receipt SHA256 | `cfe8c4ec43d70c3bdf258bf349337688c6ae0bcbb2d53bb03df29049baadc43a` |

科学条件はlinear H4/H6、隣接1 Å、STO-3G、tol-only DF1e-8、rank10/19、T=.8、ε=.001、α_axis=.025、P=0。対象は標準Y4/q4/prefix3/5の新45/32軌道と、既存独立S2/q8の25/23軌道の固定N再会計であった。条件・n・候補・科学関数を変更していない。

| 実施内容 | H4 | H6 | 合計 |
|---|---:|---:|---:|
| 親context台帳entry | 9 | 9 | 18 |
| 新trajectoryのTASK予約 | 0 | 0 | 0 |
| request sampling / RTE event | 0 | 0 | 0 |
| native evolution / wrapper / compile | 0 | 0 | 0 |
| trajectory statevector / oracle action | 0 | 0 | 0 |
| 受入済み新trajectory | 0 | 0 | 0 |
| paired統計・方向比較 | 0 | 0 | 0 |

各系でbootstrap1/setup1/saved snapshot restore1/saved reference read1/occupation oracle初期化1/Gaussian prepare2/matrix-only tail coefficients1/tail spectrum1を実施し、prewarm receiptと9 entryの完全一致を確認した。したがって、科学計算を一切行わずに止まった実行ではない。新shot・signal・reference再計算・S2再取得・S2再会計は実施していない。evaluation rootは未作成。

新grantは2件ともCONSUMEDのまま保全し、未使用へ戻していない。残予算を返金・再利用していない。[消費・終了・保全監査](end_audit.json)、[STOP](STOP.json)。

## 原因と人工確認

エラーは`'NoneType' object has no attribute 'runner_namespace'`。新adapterの`worker()`が`RT.RUNTIME`を参照していた。再利用した`rebind()`は、関数が使うglobals辞書と、その初期値をコピーした`SimpleNamespace`を返す。遅延初期化後、globals辞書の`RUNTIME`には実体が入るが、コピー側の`RT.RUNTIME`は初期値Noneのままだった。同じ不備がcache guardにもある。

これは追加adapterの実装不備である。準備段階のmockは`RT.RUNTIME`を直接埋めたため、実際の遅延初期化を再現できず見逃した。Hamiltonian・sampling law・compiler数値結果の不一致を示すものではない。

停止後、**元SOURCEを変更せず**、元の`rebind()/get_runtime()`を使う人工fixtureで同じエラーを再現した。`get_runtime()`が返した実体をadapter側に保持する修正候補について、遅延初期化後のworker・cache参照、人工partial12/wrapper2、wrapper直前の2予約を確認した。[人工再現コード](reproduce_binding.py)、[確認結果](binding_reproduction.json)、[未適用の修正候補](binding_fix_UNAPPLIED.patch)。

修正候補は封印済みSOURCEへ適用していない。科学grantの再発行・実H4/H6の再実行・新しい費用統計は行っていない。人工確認は一次担当による技術確認であり、別担当の科学レビューではない。

## 終了・保全

fresh admissionでH4 17/H6 27 workerを並行起動した。H4のnative launcherはexit1、wall約9.23秒、H6はexit1、約26.66秒。全44 workerはexit0で回収済み、両native launcherとcontrollerも終了・回収済み。processの正常回収は、科学取得の成功を意味しない。既存end auditとcontext auditは両系ともSTOPであり、PASSへ変更していない。

固定SOURCEの6ファイルとsealed入力504 fileを再照合した。停止run・controller・authorityの既知64ファイルをprivate保全snapshotへ記録し、SHAを照合した。[停止証拠の相対名・SHA](stopped_run_file_hashes.json)。元SOURCE・input・結果・raw・grant・ledger・marker・過去STOPに変更なし。今回のsealed SOURCEもそのまま保持した。共有環境・他processの変更、強制kill、PSIを理由にした中断は0。

private authority・seed・内部絶対pathは公開せず、集約状態とSHAだけを保存した。旧strict COMPAREはBLOCKEDを維持し、詳細科学レビュー・最終freezeは開始していない。

## 次の担当

**Codex**：継続を依頼された場合、別worktreeで修正候補を適用し、実際の遅延初期化とfork workerを通す人工回帰検査を追加する。旧run・使用済みgrant・SOURCEを保全したまま、新SOURCE・新契約・新grantの未実行案を固定する。

**ユーザー**：その修正後の固定契約に対し、科学再実行を別途明示承認する。今回の承認を新SOURCEのretryへ自動転用しない。

現在は**mandatory STOP**。新Y4標本がないため、ε=.001の独立G/SE、S2との比・差、Full/PRの方向再現について新しい結果はない。
