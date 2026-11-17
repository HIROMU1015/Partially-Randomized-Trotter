# Romulaptopの保存準備入力の検収

対象は `search-b1a74e00581b89ea42db` の `PAULI_COMPILE`、`prep-c19ba7776dc9905cf3ac5906`。準備SOURCEは `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2`。外側bundleのSHA-256は `553f73133aa53361eb902a46738cabc45e0e6876c3432beffb4bacc858e05612`、原REQUESTのSHA-256は `0086a136e783487f175041c4f78cdf5004ec0bb782cfe359aacb827cb81803de`。

## 原因とローカル0の復旧

対象profileが使う固定checkoutから新規共有cloneを作って再現した。外側bundleの `git bundle verify` は0.018秒で通り、続く `git fetch <bundle> HEAD` は3.122秒で226件の未解決deltaを報告した。前提commitの存在を確認するverifyでは、受け手のGitストアにthin packのdelta baseが全てあることまでは確認できない。外側のbundleを修復すると、内側のprovider bundleと、価格証拠および出力先を含むsparse checkoutからも不足objectが見つかった。

RomuTailの固定release `.rpe-central/releases/95a74e448-pbs-current/.git/objects` と元runのcontroller checkoutでobject IDと固定入力hashを照合した。そのうちlocal0-wslに不足していた外側base 110件、内側base 3件、固定証拠の参照に必要なobject 5件、通常のsparse展開に必要なobject 1,275件を専用bare Git領域へ補完した。合計1,393 object、9,529,163バイト。元runのartifact cacheから作業入力の不足23件も既存cacheに追加し、個別hashを照合した。

補完先はlocal0-wslの `.romutail-controller/worker/local0-bundle-inspections/git-object-supplements/`。固定view `.romutail-controller-acceptance/source-d887abdca` のGit alternatesから参照する。通常workerが新規workspaceを作るときも、既存の共有cloneによって同じobjectを参照する。`install_git_object_pack.py` は固定HEAD、pack SHA-256、indexに含まれるobject IDを検査し、専用領域を登録する。再実行時に全objectが既に見えていれば取込を省く。

## 検収

`inspect_saved_preparation.py` には原REQUEST、実際のworker profile、固定transportのcheckoutとSHA、新規出力先を渡す。固定版の通常workerと同じ `dry_check`、`_clone`、`_include_declared_outputs`、作業入力のmaterialize、SOURCE guardを使い、provider起動の前で終了する。

| 検査 | 初回 | 再実行 |
| --- | ---: | ---: |
| worker事前照合と固定入力105件のhash照合 | 4.430秒 | 4.363秒 |
| 新規clone 2箇所と作業入力27件の展開・照合 | 8.131秒 | 8.166秒 |

両回とも同じ保存bundleをfetchし、SOURCEとPAULI_COMPILEのPython 3.12.3を照合して通過した。再検収の追加転送は0件。別の保存要求19件のうち、両bundleがある13件を通常workerで事前照合した。8件は通過し、5件は固定commit `d7a976a629a6db68197b24f8f86734e109c8bf5d` のLD6設定JSONの参照で保留。残る6件は両bundleが揃っていない。事前照合のみの13件には新規workspaceの展開結果を付けていない。

別の人工jobは配置manifestとruntimeを照合してlocal0-wslで1回実行した。重複submitは既存attemptを返した。RomuTailの人工台帳で返却結果SHA-256 `b85e95e8d42be80c933dd920168ef74a1f8207200fbc1a4fa298dd9b982ee97c` とworker ackを確認した。受付ON、同時Q3枠auto・初期1、DP並列16を読み戻した。実信号・実compile・実Q3・本番キュー投入は0件。

## 中央担当への引き渡し

- `tools/local0-search/tests/Test-SearchClient.ps1` は既存SSHクライアントから隔離した人工bootstrapを呼ぶ。未準備の候補を含むJSONの原本byte、準備予算の未指定と明示0、候補ごとの進捗、準備待ちと並行する評価、Windowsクライアント終了後の継続、再送・再開での起動1回を検収した。
- [応答契約](../local0-search/PROGRESS_CONTRACT.md) に従い、中央の台帳から `RPE_LOCAL0_PROGRESS_V1` と継続元run IDを返す。既存の旧runはクライアントの状態照会で技術停止と表示した。継続と共有currentの採用は中央の通常手順で行う。
- 共通側ではbundle生成・配布時に、前提commitに加えてthin packのdelta base、固定証拠の `commit:path`、sparse展開に必要なtreeとblobの到達可能性を検査する。workerの `_metadata_input_repository` と `_clone` が受け取る固定storeを使い、捨てられる新規cloneでverify、fetch、固定hash照合、sparse展開までを配布前検収に含める。欠けた依存は固定された配布物として宣言し、既にあるobjectは再利用する。今回のlocal0側はこれらの共通worker関数と生成処理を変更していない。
