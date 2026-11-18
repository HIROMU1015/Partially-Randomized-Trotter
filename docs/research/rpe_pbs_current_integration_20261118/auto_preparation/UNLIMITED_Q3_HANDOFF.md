# Q3件数上限なしのH4探索とローカル0統合

RomuTailのcurrentは `romutail-auto-preparation-5b48aabbc`。固定controller SOURCEは `5b48aabbcb8079c6e5c5fc7e0f944fd1e15b2d67`。切替元は `romutail-auto-preparation-a4bb8e7c8`。切替時に稼働中のcontrollerとrelayがないこと、および30件の既存runの固定設定が同一であることを確認した。中央状態の正本は `.rpe-central/state`。新しいmanifestとホスト別検査は `.rpe-central/fleet/romutail-auto-preparation-5b48aabbc/`。

科学Q3は `28d976b99cdf8329adcc1b79327e329e947ef774`。timed adapter、準備wrapper `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2`、価格worker、各providerと工程別runtimeは既存の配置を使用する。RomuTail、qg2、Romulaptop WSL、PBS gatewayで配置照合PASS。server1は停止扱い。local0-wslは受付ONのまま。ホストのCPU・メモリ・同時枠を維持した。

## 操作

```powershell
$run = 'search-0edbfa5f4df9cb2257f5'
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId $run
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Stop -RunId $run
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId $run
```

通常Startには `-RequestFile` で今回の探索範囲を明示した。Q3の件数制限を外す指定は `scope.stop.max_new_q3_per_method` の文字列 `UNLIMITED`。明示的な0と省略は準備工程ごとに区別する。既存runは自身の固定releaseを使用する。終了済みの `search-0edbfa5f4df9cb2257f5` のStatusは固定 `a4bb8e7c8` を返し、Windowsでは `TECHNICAL_FAILURE` と表示する。このrunのResumeには未確定Q3の扱いの確定が必要。

新しい継続設定と§10.5照合は `.rpe-central/deployment/romutail-auto-preparation-5b48aabbc/authorized-unlimited/`。確定した設定のPreview IDは `search-ef6754a5ec267a136aea`。Windowsの専用ディレクトリ `C:\Users\romuk\rpe-unlimited-preview-5b48aabbc\` の `CLIENT.json` と `REQUEST.json` で通常入口を使う。後継Previewは照合保留中で起動していない。旧runへ未起動の後継を重ねない。

## 実探索の結果と保留

対象は直線H4、隣接距離1 Å、STO-3G、DF rank 10、`tol_only 1e-8`、M=12、binary64 h=0.1。shot配分は全方式auto。固定の合格判定を通過した下表の費用は探索時点の条件付きRZ費用。比較点は[保存比較点](AUTHORIZED_SAVED_COMPARISONS.json)のFull PF M5・16hとB0I p7・Yoshida4・32h/5。

| 方式 | 指定範囲 | 開始時に再利用 | 新規Q3の受領／採用 | 暫定最安の条件とRZ費用 | MSE下限〜上限の表示値 |
|---|---|---:|---:|---|---|
| PR-Pauli | LD4〜7、絶対κ [2,64]、h/2h/4h/8h | 32 | 9／8 | LD5、2nd、h、398,050,902.43 | 2.5134114164075188e-06〜2.5134114164075235e-06 |
| B0I | p6/7/8、2nd・4th・4th(m5_best)、h/2h/4h/8h、36構成 | 16 | 7／7 | p8、4th(m5_best)、8h、1,041,325,366.00 | 2.4986203267724546e-06〜2.4986203267724546e-06 |
| PR-DF | LD7〜9、h/2h/4h/8h、保存κ規則 | 23 | 8／8 | LD7、605,742,808.37 | 2.5371513471782050e-06〜2.5371513471782050e-06 |

中央発行の元runは `search-9263482821b08a34a00a`、`search-8c5b56e7d2e6d5d599c7`、`search-4bd3a237a18ed0429847`。元の上限32・16・16と消費済みの64件、固定SOURCE、既存STOPを保持した。Windowsの通常Startから `search-0edbfa5f4df9cb2257f5` を起動し、同じStartの再送が同じrunの `ALREADY_RUNNING` に解決した。新規Q3は25起動、24成果受領、23採用、2件未起動取消。24成果はRomuTail、qg2、Romulaptop WSLから回収し、採用の内訳はRomuTail 12件、qg2 4件、local0-wsl 7件。

B0Iのp8・4th・4hは信号1、compile 2、監査・集計4の実行段階タスクで準備を完了した。compileはqg2、信号と監査・集計はRomuTailで実行した。予算の集計単位は段階タスクで、compile wrapperの数とは別。追加準備はsignal・compile・auditに `UNLIMITED` を適用し、今回の消費はそれぞれ1・2・4。元の消費台帳には書き戻していない。停止した版が列挙したB0Iは12構成だった。後継では固定索引 `10703e4d969c42d0b088795811411945aed12d20` と登録済み36件が一致し、この準備済みp8・4th・4hの原成果を再利用する。

PR-Pauliでは準備済みLD4・LD5のhから探索し、選択済み8件の追加準備が旧controllerのSHAをproviderで解決できず待機した。後継の準備入力は固定wrapper `d887abdca` を明示する。PR-DFの新規評価はLD7・h。LD8・hはdescriptorのexact λ照合、LD9・hは既存価格compileのSTOPを保全。LD7〜9の広い幅は固定整数グリッドのmapperで保留した。幅や時間は変更していない。

停止を起こした `bf164759c491f8d2c68b` はB0Iのp8・4th(m5_best)・8h。Romulaptopの同じattempt `5062201d1f98428b8486f0fbd4d693d4` で実Q3開始と途中ファイルを確認した。終了receipt、完全なOUTCOME、稼働中のsupervisorは確認できず、`UNKNOWN_RESULT` と `V2_RESULT_REQUIRES_REVIEW` を保全した。新しい§10.5照合では既存25起動分も調べ、再利用候補95件（Pauli 41、B0I 23、DF 31）を識別した。この1件のため開始判定はHOLD。再投入は0件。件数上限による終了も収束判定も発生していない。

## ローカル0と配布の検収

- 固定 `c9c9d0bf07da4c4cd96143c5e8c466d6940bbd8e` のクライアント、人工検収、bundle診断を統合した。WindowsとRomuTailの通常Previewで新release、3方式の `UNLIMITED`、`RPE_LOCAL0_PROGRESS_V1` を確認。Windowsの元runの技術停止表示も確認した。
- `prep-c19ba7776dc9905cf3ac5906` のWSL検収は2回の新規workspace、固定入力105件、作業入力27件、追加取得0件を再利用した。人工jobの結果hash照合とackも再利用した。共通bundleは省略した外部delta baseへ依存せず、履歴の `commit:path` に必要なtree/blobを同梱する。通常workerと同じ新規clone、fetch、sparse展開、固定SOURCEと入力の照合を生成時に使う。履歴blobが欠けた人工cloneで受入を確認した。
- LD6の固定入力は `d7a976a629a6db68197b24f8f86734e109c8bf5d:docs/research/rpe_split_preparation_20261116/LD6/PREPARATION.0.CONFIGURATION.json`、SHA-256 `b5e198fc98c900d96fcd5937daf9bb0a5f46e1ee1741f690e6db18bcbe02cc27`。中央で特定した。旧WSLの5件の保存REQUESTと、元bundleが揃わない6件は一括再検査していない。元REQUESTと過去のbundleは固定のまま。
- PBSの新規実行は0件。後継設定のPBS時間はQ3 14,400秒、保存receiptがある準備工程についてprovider・SOURCE別に見積りを設定した。既存の24時間allocation、96 CPU、CPU総上限100、drain条件を保持する。時間は受付用の余裕値で、実測の上限ではない。PFのcompileは既存のqg2限定経路を保持した。
- PRの `SCHEDULE_WIDE_V1` とshotの `METHOD_MINIMUM_PRIORITY_V2` を保持する。内部区切りは `WORK_SLICE_YIELDED`。候補規則が尽きたら `REQUESTED_WIDTH_RULE_EXHAUSTED`、保留が残る場合は `SCOPE_HELD_INPUTS`。大域最適性を保証しない。通常入口・配布の回帰52テストと3サブテスト、旧上限を超える人工探索、候補枠再配分、通常Resumeの二重起動防止を確認した。

公開済みrunは結果branch `9f86f209aff04bd5b9960968ce1a2c0597072106`、Cの地図は `cdad25ee87ffc18084de4b3569486fcfd50c9d86`。Romulaptopの `C:\Users\romuk\OneDrive\ドキュメント\ChatGPT\研究\H4-search-map\index.html` に当該runが表示され、最終同期が `FINAL_RESULT_PUBLISHED` であることを確認した。
