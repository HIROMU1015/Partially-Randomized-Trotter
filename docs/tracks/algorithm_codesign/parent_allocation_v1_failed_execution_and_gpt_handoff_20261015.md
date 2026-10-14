# Parent allocation v1：one-shot技術停止・保存監査・引き継ぎ

2026-10-15 JST。最終分類 **PARENT_ALLOCATION_TECHNICAL_INCONCLUSIVE**。
固定sourceを一回起動し、初回key job作成時に停止した。retry0、marker消費済み、mandatory STOP。
科学的な配分利益・非利益、ordinary/P5の優劣、研究の継続・終了を判断できる比較結果は取得していない。

## 正本・来歴

| 項目 | 固定identity |
|---|---|
| Source S | `8dfec590a8d760eb15932d862698b270649e9bcf` |
| Authorization-only直接子 A／実行HEAD | `66d8c026210a56108f2c443695c6d9f52f59fe0b` |
| branch | `track-b-parent-allocation-one-shot-execution-20261015` |
| Contract SHA256 | `42187e9c8cb387ea72ce0d93b45e2e4c40ee002a1170e6d940b386e13facf5c6` |
| Authorization SHA256 | `09c45fc766ecb9462fc7eea275cc39536d6b83634ffac845a1bf7bea66426d68` |
| Source manifest SHA256 | `6011ae3dbab5ed392cac2394ca0b74c8e8e68c14d8514b12560b131e3af547af` |
| Marker SHA256 | `257112f17667e42550b434ced6aea4cb926ebad939adbbba0e0c806d4e01ad7c` |

利用者の原文「次の作業に進んで大丈夫です」を、直前の固定S one-shot承認待ち報告に対する実行承認として
[authorization receipt](parent_allocation_execution_authorization_receipt_20261015.md)とJSONへ記録した。
Aの唯一のparent=S。Aの変更はauthorization JSONとreceiptの2 pathのみ。
launch前にHEAD=A、remote SHA=A、clean、161 critical hashes、固定runtime、marker未存在を確認した。
準備branchの公開確認HEADは実行HEADに使用していない。

[原結果ディレクトリ](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v1/)を正本とする。
[terminal](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v1/terminal.json)、
[inner terminal](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v1/inner_terminal.json)、
[outer process receipt](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v1/outer_process_receipt.json)、
[STOP](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v1/STOP.json)は不変。
本書は結果や分類の置換ではない。

## 実行範囲と到達段階

既知3-qubit distinct-basis synthetic、p=(1/5,3/10,1/2)、x=5/7、状態000、σ=+1、r=1、共通finite P5。
これは分子DF実験ではない。primitive1e-6、K256/H160、rho=eta=1e-12、guard100dps、axis精度1/200、alpha_axis1/40を維持。
合成worker上限2、numeric threads1。固定CPU/RSS/output/log/call上限は変更していない。

| 段階 | 保存された到達状況 |
|---|---|
| source／runtime／authorization gate | 通過 |
| exclusive marker／ledger start | 各一回保存 |
| 使用旧unique sequenceのstrict再guard | 20 receipts保存 |
| 旧common価格に基づく候補・C* freeze | 15構成のmetadata保存。新価格による選び直しなし |
| inventory freeze | unique36＝旧20＋planned新16。key上限18内 |
| 登録root controlled診断 | planned72、実施0 |
| 初回新key job作成 | TypeErrorで停止。jobファイル0、worker起動0、backend reservation0 |
| 新合成 | 0 |
| 完成比較law／N／資源比較表 | 0。resource_map、rows、laws、eventsディレクトリ未生成 |
| quantum shots／sampling／trajectory／LP／DF／GPU／旧runner再実行 | 0 |

freezeや旧guardは認可された本runのprefixとして保存する。これを勝敗、非改善の証拠、完遂済み比較として利用しない。
planned key数・予定matrix件数は実取得数と区別する。新合成0なので2 worker並列の実行速度は測定できていない。

## 停止原因

保存された例外：

```text
TypeError: dict() got multiple values for keyword argument 'parent_pid'
```

[runner](../../../src/trottertracks/algorithm_codesign/parent_experiment/runner.py)は外側owner PIDを`ledger_job`の`parent_pid`へ入れる。
ledgerがそのjobを`run_pipeline`へ渡し、候補／inventory hashを追加したidentityにも`parent_pid`が残る。
[pipeline.acquire](../../../src/trottertracks/algorithm_codesign/parent_experiment/pipeline.py)の初回key作成は次の形になっている。

```python
job = dict(**identity, key=key['key'], ratio=key['ratio'],
           parent_pid=os.getpid(), job_index=index)
```

この二重指定が、key JSONの保存・Popen・backend呼出しより前にTypeErrorを生む。
通常のPython辞書による人工再現、保存済みledger identityの`parent_pid`、初回job未生成、保存例外が一致するため、原因の確信度は高い。
ただし本runnerはstack tracebackを保存していない。正確なline tracebackが保存されたという主張はしない。

事前71 testsは数学・ledger・guard・marker等を検証したが、人工pipeline接続テストは取得関数をmockし、非空inventory＋ledger由来`parent_pid`を実際の`acquire`へ渡す組合せを覆っていなかった。
これはsourceの技術的なcoverage不足であり、科学仮説や精度の問題ではない。

## Resource・内外status

- runner呼出し一回、outer exit1、inner／outer／terminal全てtechnical inconclusive。
- outer wall約3.636秒。completed ledger tree CPU約3.621秒、outer CPU約0.412秒。観測CPUは別のsampled値3.9秒。
- sampled aggregate RSS peak 57,479,168 bytes（約54.82 MiB）、completed child peak35,144 KiB。
- resource_failure=null。CPU/RSS/output上限超過による停止ではない。
- outer receipt時点のoutput121,389 bytes。terminal／STOPまで含むraw最終ファイル合計123,875 bytes。
- retry0、成功prefixの科学利用禁止、次stage未認可。

## Read-only保存監査

[監査JSON](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v1/saved_output_audit/saved_failure_audit.json)と
[実行した監査script](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v1/saved_output_audit/audit_saved_failure_as_executed.py)を保存した。
23 checks PASS。source/A/contract/manifest、marker・terminal・STOP・内外status/hash、inventory範囲、旧guardの保存binding、取得件数0、親worktree保護を照合。
161 critical paths／92旧保護対象に変更なし。親HEAD/index／既存69 status entriesのfile hashに変更なし。
旧guard監査は保存metadata/hash/countsの確認だけで、operator guardを再実行していない。
新しい係数評価、回路生成、matrix評価、合成、N計算、science runner再実行は監査中に行っていない。
全rawファイルは512 KiB以下でGitHub接続から個別取得できる。auditはそのpath・bytes・SHA256を索引化している。

## 最小修正案と次の境界

技術的な修正候補は、key jobでledger owner PIDを上書きできるmapping形式へ変更すること。
例えば`dict(identity, key=..., ratio=..., parent_pid=os.getpid(), job_index=...)`なら、重複kwargsを避けつつworkerのppid検査を保持できる。
outer→ledgerのparent bindingとledger→keyのparent bindingはそれぞれ維持する。

修正準備には、非空inventory・既存`parent_pid`・mock processで初回job作成へ到達する登録外回帰テスト、marker／job ownership・改ざん検出・失敗時保存の確認が必要。
今回source、contract、authorization、markerは修正していない。修正を採択・実装したとの意味でもない。

採択済みの科学比較、精度、baseline、capsを変える必要は現在確認していない。技術修正はCodexで扱える範囲だが、凍結Sの改変やこの消費済みrunの再試行はできない。
将来実行するなら、別source S2・更新manifest／テスト、新しい未使用namespace／marker、S2のauthorization-only直接子と別途明示one-shot承認が必要。
現在のAやmarkerを再利用しない。

**今回の実行と保存監査をここで終了し、mandatory STOP。**
重要科学レビューで勝敗を判定できるnative比較はまだない。修正準備・新実行の必要性を次に判断する材料として本書を返す。
GPTの科学的採否・新規性・研究方針判断や新しい詳細レビューは開始していない。
