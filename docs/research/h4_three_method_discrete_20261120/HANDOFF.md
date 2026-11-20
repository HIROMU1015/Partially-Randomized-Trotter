# H4の3方式・離散130条件

通常入口 `scripts/run_rpe_search.py` で次の範囲を指定する。
ユーザーはこの範囲の対応と実最適化の開始までを指示している。
既存の中央キュー、科学SOURCE、工程別予算、STOP、未確定attemptを引き継ぐ。

|方式|公式|分割・切断|要求幅 Δ/h|絶対κ|設計条件数|
|---|---|---|---|---|---:|
|Full PF|2nd、4th、4th(m5_best)|全10 fragment|2、4、8、16|なし|12|
|B0I / TRUNCATED_PF|同じ3公式|p=6、7、8、9|2、4、8、16|なし|48|
|PR-DF|2nd|L_D=5、6、7、8、9|1/2、1|1、2、4、8、16、32、64|70|

モデルは現行H4 linear 1 Å、STO-3G、tol_only_1e-8の保存snapshotと10 fragmentを使用する。
hは既存binary64の0.1（厳密には3602879701896397/36028797018963968）を維持する。
要求幅を短い初期roundの実幅に置き換えてscopeを判断しない。各roundのqと実幅は固定mapperが決める。
shotは採用済み `RPE_SCALED_SHOTS_V1` のauto、M探索と測定配分gridも既定値を維持する。
130は設計条件数であり、物理scheduleの同一性による再利用後の構成数やQ3評価件数とは異なる。

## 通常scopeの指定

PFの `per_method.METHOD.delta_over_h_values` は正の厳密文字列の有限リスト。
PRでも同じ項目を指定でき、固定hとの積を厳密な絶対要求幅へ変換する。
区間 `delta_over_h`、絶対幅 `delta_pf` との同時指定は拒否する。
PFの明示離散指定では、過去のconfiguration IDリストによる偶発的な狭窄を解除し、
今回指定のp・公式・幅からcanonical metadataを作る。
保存catalog、PF索引、registryにも同じ離散制限を適用する。

PR-DFの `kappa={"mode":"absolute","values":["1","2","4","8","16","32","64"]}` は、
Full PF・B0Iと同時に指定できる。上下限区間への置換はしない。
finite-grid plannerはこの7点だけを提案し、経験的な上方停止や近傍細分でgrid外へ出ない。
保存候補、継続snapshot、liveで準備を終えた候補にもL_D・公式・幅・κの同じ制限を適用する。
物理的に同じscheduleに写るκも指定値として記録し、既存の同一性照合で成果を再利用する。

## 実装と検査

既存 `pf_candidate_space` は16hを受け取れるため再利用した。新しい分子計算器やPF公式は作らない。
明示幅がある場合は旧PF索引にないS2・標準Yoshida4の16hもcanonical metadataとして列挙する。

- `tests/test_rpe_discrete_scope_entry.py`：通常scopeの3方式同時解決、厳密なhとの積、曖昧な表現の拒否。
- `tests/test_pf_discrete_scope.py`：Full PF 12件とB0I 48件、旧索引外の16h、保存候補の離散制限。
- `tests/test_rpe_discrete_kappa.py`：7点の有限探索、grid外の保存候補除外、pendingの保全、非実行PlanOnly。

計算と配置の経過・run IDは開始時の追記で示す。人工テストは科学的性能を示さない。

## 旧ローカル評価の失敗を保全する開始照合

初回Start `search-5c9a54cac2fe1630160c` は科学起動前の§10.5で保留された。
旧server2取込み `prior-c11a81adff03` のPR-DF4件にOUTCOMEはないが、元ホストと
取込みの双方に終了コード1・`automatic_retry=false` のPROCESS_RETURNが存在した。
元ホストで対象runの稼働プロセスがないことを照会し、原ファイルのハッシュも一致した。
旧attempt、原結果、STOPは変更せず、同じ構成・測定配分を再利用も再投入もしない。

既存 `verify_prior_q3_holds` に `LEGACY_FAILED_RETURN` を追加し、取込みprovenance、
固定問題の同一性、原REQUEST・起動・失敗receipt、元hostの状態照会を照合する。
これは失敗を成功や科学結果へ変換せず、該当配分だけを既存shot探索のheld集合へ渡す。
他の独立した条件の通常開始を妨げない。旧4件の再試行は許可しない。
人工回帰 `tests/test_rpe_legacy_failed_hold.py` は、原本不変、改変・稼働中・再試行の拒否、
同じ配分がshot候補にあっても新規評価されないことを検査する。

## 通常Startと実Q3開始

固定controller `d9a4c2ac47c72c7f42190b5a0ab88313ba1e1cd2` を使う
release `romutail-h4-discrete-d9a4c2ac47` を新規run向けcurrentへ採用した。
科学Q3・価格・信号workerとruntime、PBSの需要連動配分は前の混在版を維持する。
scope原本はRomuTailの
`.rpe-central/planning/h4-three-method-discrete-20261120/SCOPE.json`。

通常Startのrunは `search-906c799cfd87c62160ce`。
§10.5はPASS、B0I保存評価255件の再利用と旧失敗4配分の個別保留を確認した。
Full PF・B0I・PR-DFの全3方式で新しいQ3の実行・結果回収が始まった。
観測時点では9件が中央ACCEPTED（Full PF4、B0I3、PR-DF2）。
この9件の判定は `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD` で、成功解の発見とは異なる。
server2とWSLに加え、PBS `267545.CPUcluster-PRISM` でもFull PF1件を実行・回収した。
同PBS枠は適格な需要に応じた2 CPU・9 GiBで、96 CPU運用の検収ではない。
依頼は開始までであり、担当の判断でStopせず探索を継続している。

一部の新規PR-DF候補はdf-planで元の絶対パス
`/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/H4_context.json` を解決できず
候補単位で保留された。新しい信号・価格準備の全条件が通過したという主張はしない。
既存成果を使う評価と独立した候補の探索は継続する。後続の技術作業は、
このcontroller側metadataサービスへの保存入力パス受渡しの修正である。
失敗した候補の原記録を保全し、同じ科学処理の自動再試行を追加しない。

通常操作：

```bash
.venv311/bin/python -B .rpe-central/releases/romutail-h4-discrete-d9a4c2ac47/scripts/run_rpe_search.py \
  --config .rpe-central/CLIENT.json --action status --run-id search-906c799cfd87c62160ce
```

実装は親Codex、PFとPR backendおよび人工回帰は委任Codexと共同で担当した。
親Codexによる関連66テストと4subtest、委任Codexによるcontinuation14テストが通過した。
実環境への配置・元hostの終了確認・通常Start・Q3結果回収の確認は親Codexが担当した。
独立した重要科学レビューを実施したという意味ではない。

## 2026-11-20：運用障害の修正と継続準備

上記runはPF準備の `check_pf_preparation_owner.py` が、新releaseの出力先を旧PF専用Pythonの
書込み範囲外として拒否して終了した。Q3は20件を回収済み（Full PF9、B0I8、PR-DF3）。
PBS枠267545は終了F・Exit_status=0、中央キューはACCEPTED20・CANCELLED1で未確定attemptなし。
元runの記録は変更しない。

修正版は新しいPF専用出力設定、PR metadataへの保存入力path map、DF_DESCRIPTOR専用SOURCE、
RomuTail Q3の固定adapter checkout参照を使用する。新しい科学条件は追加しない。
`OPERATIVE_METADATA_REPAIR` 継続は保存されたPF metadata障害・終了・キュー・scope・予算を
照合し、明示STOPと科学STOPを維持する。PRのpath障害は3工程すべてがscience=falseの保存記録を
持つ場合だけmetadataを再準備できる。旧科学処理の失敗配分は保留を引き継ぐ。
継続用snapshotは `continuation-inputs/h4-discrete-906c799/` に元記録のまま保存する。
実装・人工検査は親Codexと委任Codexが担当し、重要科学レビューではない。
再開runと実worker確認は配置後に追記する。

最初の修正版07961cefe4のrun `search-7d6ddb1b06ef919fac24` でQ3を19件回収した
（Full PF9、B0I6、PR-DF4）。RomuTail自身でFull PF6件の実行・回収を確認した。
追加descriptorのbundle照合先をmetadata作成用controllerと共用したため、controllerの
metadata commit後に固定HEAD照合が保留された。通常Stopで起動済み処理を回収し、
EXIT=0・全キューACCEPTED・全PBS枠F・relay終了を確認した。STOP原本を残す。
後続配置ではdescriptorの専用exact checkoutを全hostに分離し、科学stageに入っていない
recoverable operational holdだけ継続する。保存snapshotは
`continuation-inputs/h4-discrete-7d6ddb/`。既存の科学失敗保留は引き継ぐ。

## 最終の再開配置

SOURCE `b0c04ae1915661af8de4f34957304e2cb0e7aed9`、設定release
`romutail-h4-repair2-b0c04ae191-r3` で、通常入口から
`search-9c1865e787c6aab3d251` を開始した。scopeは同じ130条件であり、予算も引き継ぐ。
直前の2配置は保存状態の読込優先順位・検査出力を固定入力へ含めた設定不整合により
科学起動前に保留された。LAUNCHを持たず、既存attemptを再試行したものではない。
最終配置では親の全状態を先に読み、過去の候補STOPを保全して、検査出力を固定入力から除く。

起動前の継続照合と§10.5はPASS。RomuTailの新Q3を中央ACCEPTEDで回収し、server2とPBSでも
Q3の実計算開始を確認した。修正したDF_DESCRIPTORはWSLで実行に入り、RomuTailでは
PR-DFの価格準備compileが始まった。準備stageは回収まで別途確認する。
旧20件と直前19件の原結果、通常Stop、旧科学処理の失敗保留は保持した。
親Codexの最終関連回帰は37 passed・8 subtests passed。人工テストは科学性能の証明ではない。
科学SOURCEや進行中runは変更せず、通常Stopも追加発行せず探索を継続する。

準備stageの回収結果は成功とは区別する。DF_DESCRIPTORはWSLで実行されたが、保存loaderが
間接importする `matplotlib` が専用runtimeに存在せず終了コード1となった。
PF信号準備も保存receiptの元の絶対パス未解決で失敗した。PR-DF価格較正は8 compile完了後、
`INDEPENDENT_COHORT_REQUIRES_EXPLICIT_INPUT_ADOPTION_NO_POOLING` によりSTOP_PARTIAL
となった例を確認した。これらのraw・receipt・失敗候補を残し、自動再試行はしない。
全130条件の準備成功・探索完了・科学的成功解を主張しない。
実行可能な候補のQ3は継続している。次はCodexが専用runtimeの不足、PF保存入力の経路、
価格較正の独立コホート採用条件を整理し、固定契約の変更や再実行が必要な範囲を区別する。

## 2026-11-20：利用者の修正・再実行指示への対応

利用者が残件の修正と再実行を指示した。元9c runは通常Stopで回収し、EXIT=0、
全25タスクのACCEPTED24・CANCELLED1を照合した（Q3は11件、準備は13件）。
旧attemptと結果・STOPはそのまま保全し、選択した終了済み準備失敗だけを新runで再実行する。
過去の旧Q3失敗4配分など、この修正と無関係な保留は解除しない。

- DF_DESCRIPTOR：matplotlib・openfermionpyscfのimportを割当前に照合し、PBSとWSLは
  その依存を満たす既存専用Pythonを使う。既存環境にパッケージを追加しない。
- PF_SIGNALS：固定科学loaderを維持し、保存snapshot・receipt・contextをhash付きで配送し、
  既存path resolverを固定loaderのI/Oへ接続する。
- DF価格：固定gridの同じ有限schedule(q/R/K/time等)を確認して、候補ID・宣言κだけの違いを
  同じ保存コホートとして扱う。SOURCE・compiler・seed・役割の照合とpooling禁止は維持する。
- DF価格のλ照合：default/NEHALEMでLD7/8に1–3 ulp差が生じた。HASWELL・BLAS1 threadを
  NumPy読込み前に適用すると、LD7・8・9で既存の保存λのhexと一致した。既存guard・許容値は変更しない。

元runのDFコンパイル40軸を原receiptへ照合し、`continuation-inputs/h4-rerun-9c/price-recovery/`
にbyte同一で保全した。監査済み価格やモデル成功に変換せず、通常の厳密な保存軸照合へ渡す。
PFの完了済みcalibration（SOURCE raw_commit `960c0497534bc13924f41be4c7ca22c437653ac6`）も
元bundleとreceiptを照合し、候補ID別の保存calibrationとして再利用する。
新しいsignal/holdout等の不足分だけを許可済み130条件内で実行する。

修正版13bの `search-2bf68d58d3d7c32243c8` ではPF信号とdescriptorを実行・成功回収した。
引継ぎ設定が旧READY recordを方式で絞らずPR-Pauli packetまでDFリストへ加えたため、
5候補のdf-planがmetadataのみでKeyErrorとなった。担当の設定ミスとして修正する。
DF専用リストへ絞り、誤方式を明確なエラーで拒否する入力検査を追加した。
実際の7候補のdf-planを局所の修正版で通し、科学stage未起動の旧3metadata receiptを保全する。

価格用の回収軸は、RMSE要求IDと有限回路IDを区別して再利用候補へ渡す。
実データの読み取り検証では40回収軸と既存LD9の8軸を厳密照合して再利用でき、HOLDは0件だった。
HASWELLの追加reference確認ではLD5・6も保存λとtail scalarへ厳密一致した。
新holdoutの独立性は維持し、保存軸を監査済みモデルや科学的成功へ変換しない。

## 2026-11-20：保存prefix信号の修正と実経路の再検収

SOURCE `cac95fd9c4` の `search-eea38701fa5a6a593e9f` では、DF価格較正3件と
PFモデル固定1件を正常回収した。保存prefixのDF信号は旧context絶対パスの未解決で失敗した。
正常Stop・EXIT=0・全5タスクACCEPTED・relay終了を確認し、新規DF較正32軸を
`continuation-inputs/h4-rerun-eea/` に元のバイトで保全した。旧40軸も保持している。

SOURCE `dac4f3c98b` は保存prefixにも既存のパス変換付きcontext loaderを使用する。
入力ハッシュ、元のlambda/scalarとSOURCEを照合し、固定科学loader・providerは維持する。
同じ評価の終了済みrunの中央CASとSHA・サイズが一致する重複キャッシュも回収できるようにし、
server2で約5.89 GB、WSLで約3.64 GBを整理した。未確定attemptの入力・原結果は保護した。
常設の所有ユーザー回収サービスもこの固定版を使用する。

`romutail-h4-rerun3-dac4f3c98b` は4環境の配置検査、明示版・通常版のPreviewを通過し、
通常入口から `search-5f6fb3cdbf9ca0b2f8af` を開始した。Start応答は120秒で時間切れになったが、
指定run IDのStatusで起動済みと照合し、再送していない。
同じH4 linear 1 Å / STO-3G / tol_only_1e-8・10 DF fragments / binary64 h=0.1の130条件を維持した。
server2でLD6・幅h・絶対κ=1とLD7・幅h/2・絶対κ=1の信号を各13ラウンド実行・回収し、
両候補が `READY_FOR_SEARCH` に進んだ。LD7の較正は保存16軸を再利用し、held_axes=0だった。
独立holdoutと後続Q3も実行・回収した。閾値超過のQ3を成功解や科学的性能改善とは扱わない。

別途PBSのPF_MODEL_FREEZE 1件は、PF専用Pythonが監督用 `run_rpe_q3_worker.py` を拒否し、
科学adapterの起動前にUNKNOWNとなった。元のUNKNOWN・所有・ログを保全する。
修正切替のため通常Stopを行い、EXIT=0、ACCEPTED18・UNKNOWN1、relay終了、
専用PBS割当267558のFを確認した。次版ではtransport監督にnative Pythonを使用し、
科学providerの専用Pythonと固定入力は維持する。限定した起動前障害の継続照合を通してから再開する。

直前runの新Q3は4件を回収し、すべて `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD` だった。
別の2要求のdry-check保留は `COUNT_ONLY_SCOPE_EXCEEDED` であり、PBSのtransport障害とは区別する。
これらの数値範囲制限と元の保留は維持する。
再開用派生bundle一覧は `method=PR_DF` の3件READY・1件WAITINGだけを使用する。
元STATEに残る旧PR-Pauliの5件READYをDF入力一覧へ追加しない。

保存処理中にRomuTailのファイルシステム空きが0になったため、全run終了と未確定入力の保護、
中央CAS原本とのSHA・サイズを確認してworker側の重複23件・4,545,400,832 bytesを整理した。
回収処理には実ファイルシステムの空きも見る条件を加えた。50 GiBの受付上限は変えず、
既存の所有・ロック・保護・内容照合を満たす複製だけを整理する。
元のSOURCE、raw、UNKNOWN、失敗記録を削除・書換えしていない。
受付保留と復帰の確認手順は[共通運用ルールの容量確認](../rpe_distributed_operations.md#空き枠容量障害の切り分け)を参照する。

実装・人工検査は親Codexと委任Codexが担当した。親Codexは4環境への配置、通常入口の操作、
実処理と中央回収を照合した。独立した重要科学レビューは実施していない。

## 2026-11-20：最終修正版での実再開

最終SOURCEは `1355ad65faffbfb9a03cff17df4091e7c8f1c93e`、採用releaseは
`romutail-h4-rerun4-1355ad65fa`。通常入口で
`search-e6361f01b6a480718127` を開始し、Statusで `started=true` を確認した。
Startの応答時間切れでは再送していない。指定の130条件、shot規則、工程別設定、
予算と既消費、旧科学STOP・元UNKNOWNを保全した。新runに追加のStopは発行していない。

RomuTail・server2・PBSで新Q3の実行と中央回収を確認した。
PBS workerも最終SOURCEで起動し、job `ee22369b2692ebd85b49` / attempt
`d874ee8153c44d4888628f797fa27c35` のQ3を正常回収した。
PF_MODEL_FREEZE・PF_HOLDOUT_AUDITはRomuTailで正常回収され、
PF wave `2b1dbdd20db18e8adf7a3a8c10d39ee9240ab2b839ed377337c5fb540eba86ca` は
`READY_RETURNED_TO_SAME_OPTIMIZER` に進んだ。
WSLは配置検査済みであり、この最終runでの新規科学処理成功とは区別する。

検収時点では準備10件・Q3 16件をACCEPTEDで回収し、新runのUNKNOWNは0件だった。
16件中14件はモデル閾値超過、2件は `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。
後者は上記H4条件でのPR-DF / S2 / LD7 / Δ=h/2 / 絶対κ=1 / M=12、
`MODEL_MSE_FILTER_ONLY` の固定条件付きproxy判定である。X/Yの各round0..12のshotは、
PBSの `ee22369b2692ebd85b49` が `[36,33,31,28,26,24,21,19,17,14,12,9,7]`、
`5841cce04ae322680c85` が `[30,28,26,24,22,20,18,16,14,12,10,8,6]`。
最終科学結論・全130条件の終了・総cost評価完了を主張しない。

この区間のPBS割当は需要に応じた各2 CPUであり、96 CPUを実使用した検収ではない。
267559〜267562はFとなり、不要な専用割当が返却されたことを確認した。
実ファイルシステムの空きが不足すれば新規受付だけを `FILESYSTEM_SPACE_LOW` で保留し、
既存処理、他host、内容照合済みの重複整理は継続する。50 GiB上限は変更していない。
容量不足が再発した際は標準 `git prune-packed` で正式packと重複するloose objectのみ整理し、
22,706,360,320 bytesを回復した。2772個すべてのobject IDが整理後も読めることを確認した。
固有object、正式pack、科学原本を削除していない。保存失敗した作業用HANDOFFは固定SOURCEから復元した。

人工検査はworker21件・PBS12件、起動前継続22件（別の1件を除外、9 subtests）、
複製回収6件・ファイルシステム受付2件が通過した。
Windows側の初回C同期成功と、このrun専用の600秒監視がRUNNINGであることを確認した。
確認時の公開INDEXには新run IDが含まれておらず、探索マップへの掲載・表示確認は残る。
次段階はCodexが新runの結果蓄積、保存容量、公開INDEX・探索マップへの掲載を照合する。

## 2026-11-20：容量回復後のWSL・PBS復帰

利用者の復帰指示に従い、同じ `search-e6361f01b6a480718127` を通常Stopで回収し、
EXIT=0とrelay終了を確認して通常Resumeした。新しいinvocationは
`ecf4258b51a2487484b37f9372cc95a8`。探索の固定SOURCE `1355ad65fa`、130条件、
予算、既存結果・科学保留は変更していない。
旧ローカルcompile `prep-11e8784777b094024911f73d` は終了receiptがないためUNKNOWNを保持し、
再投入していない。

WSLは50 GiB上限に対し約50.2 GiBで受付停止していた。Git標準の `.rev` 補助索引があると
既存圧縮処理が見送ることを確認した。元の索引214件をSHA付きで別保管し、既存の排他・
全object照合・原子的置換を使って21,658,415,104 bytesを回収した。
全26,453 objectとUNKNOWN 20件の内容を保全し、使用量約30.1 GiB、空き4枠・受付trueを確認した。
現runでQ3 `58038c749175e3a6d2f9`、再開後はDF_CALIBRATION
`prep-c6f63084d29b397923c65f1e` とDF_MODEL_FREEZE
`prep-5297bd873f9889442903076a` をWSLで実行・中央回収した。

PBSは容量不足後に止まっていた管理heartbeatが通常Resumeで復帰した。
旧割当267563の終了Fと、新割当の通常投入・返却を照合した。
再開後のPBS Q3 `b20683451b06902532f6` / attempt
`49eb64f9d39b4d33b4c18a534f604cc8` は中央ACCEPTED。
判定はMODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLDであり、科学的成功解とは扱わない。
PBSの実使用CPU数はその時点の適格需要に従い、96 CPUの実使用を主張しない。

再発防止として、対応するpack/idxを持ちGit検証を通る `.rev` を既存の損失なし圧縮で扱う
修正を追加した。合成検査は委任Codexが24件実行しPASSした。親Codexが新しい整理サービス専用
配置 `worker-storage-recovery-9c7c246657` へ反映した。探索の固定コードは置換していない。
通常入口の応答が計算開始後も同期hook待ちで120秒を超えるため、既存CLIENT設定の待機を
300秒にした。科学計算の時間上限・予算とは別であり、時間切れ時にResumeを重ねていない。

実処理・回収・保全の記録は現runの `recovery/wsl-pbs-20261120/RESTORED.json` と関連JSON。
旧UNKNOWNの保全を続け、以後はCodexが蓄積結果と探索マップの掲載を照合する。

## 2026-11-20：準備・信号の遠隔割当待ちの調査と配送改善

通常の手順は[共通運用ルールの割当前照合](../rpe_distributed_operations.md#割当前に確認する入力と実行環境)を参照する。

利用者の指示に従い、現run `search-e6361f01b6a480718127` の工程別台帳と
workerの入力照合を調べた。4環境とも準備・信号providerを設定済みである。
WSLは空き4枠でもQUEUED準備がない区間があり、保存された
PREPARATION_PENDINGの44件を未処理44件と数えてはならない。
PBSは他環境の適格な空き枠で処理できる仕事を除いた需要に割当を増減する。
PBSで準備のreadiness成功はあるが、照合が戻る前にRomuTailで約3.3秒の処理が
終わった例があった。96 CPUを使用した検収ではなく、全環境で信号の実処理を
確認したという意味でもない。利用可能な他環境を待たせてPBSの実績を作らない。

WSLのDF信号入力bundleは約192 MBで、遠隔relayは最大4件のbundle配送と照合を
終えるまで取得に進めなかった。配送を準備1件ずつの照合・報告へ変更し、
照合済みの1件が残りの転送を待たず通常の取得へ進むようにした。
繰り返しHOLDする要求はホスト別の照合回数で交替し、Q3の照合batchは維持する。
JSON stdoutに返るworkerの失敗理由も、stderrが空の場合に限り長さを制限して表示する。
元のattempt所有、observe_only、原結果、科学SOURCEと実行条件は変更しない。

この変更はrelay配送層に限定する。現runへの適用時はrelayのみを停止して終了を
照合し、固定commitから別配置したrelayで同じCONFIG・共有mailboxを再開する。
workerが受領済みの計算は継続した。通常Stop/Resume時の配送版選択は、後述の統合で自動化した。

調査・人工検査は委任Codex、適用・実機の回収照合は親Codexが担当する。
独立した重要科学レビューではない。調査と配置の原記録は
`.rpe-central/planning/allocation-readiness-20261120/` に保存する。

配置は `relay-readiness-854de00f7c`。旧relay PID2888560の停止・終了を確認して
PID3016994へ切り替え、server2・WSLの新しい状態照会がともに失敗0で進むことを照合した。
委任Codexが追加4件を含むrelay人工検査15件を実行しPASS。親CodexはWSLの保存済み
準備2要求を新経路へ渡す乾式確認を行い、先頭1件だけの固定入力照合成功、
observe_only 2回、attempt追加0、科学呼出し0を確認した。
通常Statusは同じinvocationのRUNNING・controller_alive=true。
元UNKNOWN1件と信号失敗の自動再試行禁止は保持する。PBSの準備・信号の実実行は
未検収であり、適格な新規要求がない区間の空きと配送不良を混同しない。

## 2026-11-20：通常Resumeへの配送版の統合

前節の手動再適用を通常入口へ統合する。中央bootstrapのSERVICEに
`relay_transport_command` を設定し、Start/ResumeがSTARTEDまたはALREADY_RUNNINGを
返した直後、公開同期より先に `ensure_rpe_relay_transport.py` を呼ぶ。
これは固定runの科学コードを変更せず、指定runの配送プロセスだけを選ぶ運用設定である。
Status/Stop、開始拒否、完了済みrunでは呼ばない。切替できなければ起動状態とは別に
`relay_transport.status=RELAY_TRANSPORT_PENDING` を返す。

配送選択は既存REGISTRY.lockで通常Stopと直列化し、登録runの場所、現在invocation、
生存controller、relayのCONFIG所有を照合する。旧relayを正常停止して終了を確認し、
修正版を起動する。すでに修正版なら再起動しない。SOURCE・scope・結果・キューの所有と
UNKNOWNを変更しない。共有の常駐サービスや新しい承認機構は追加しない。

追加人工検査9件はPASS。親Codexは配送選択5件と全9件、委任Codexはbootstrap4件と
component bindingを検査した。広めの既存27件では26件PASS、既存の
`test_status_reports_completed_exit_and_preserves_scientific_stop_reason` が
UNKNOWN_RESULT時の旧期待値COMPLETEで失敗する。現行実装はRECONCILIATION_REQUIREDへ
保留するため、この無関係な科学STOP判定を弱めてテストを通してはいない。
通常Stop/Resumeの実機検収と配置記録を本節へ追記する。

実機Stopの回収中、同一driver invocationでも処理区間の切替時にrelayが作り直される
ことを確認した（06:05:53Z停止、06:06:03Z旧版再起動）。再開直後だけの切替では
不十分なため、同じrun・invocationに限って配送版へ追従する小さな補助プロセスを
追加する。通常Stop、controller終了、invocation変更で終了し、区間間にrelayが
存在しない間は起動しない。すでに修正版なら何もしない。新しい区間で旧版relayが
実際に起動したときだけ、上記の同じ排他・所有照合で配送版を選ぶ。

区間追従を含む追加人工検査14件がPASS。親Codexによる通常入口・registry・配送の
関連34件もPASSした。上記既存テストの期待値をRECONCILIATION_REQUIREDへ修正し、
UNKNOWN_RESULTの保全・Resume拒否を引き続き検査する。科学判定の実装は変更していない。

実機検収を完了した。通常Stopは旧invocation `ecf4258b51a2487484b37f9372cc95a8` を
EXIT=0で回収し、専用PBS割当もすべて終了した。通常Resumeは同じrunを新invocation
`cd6f74384447408ab92e73d5348b37ad` でSTARTEDとし、応答の
`relay_transport.status=RELAY_TRANSPORT_CONFIRMED`、action=REPLACEDを確認した。
配置は `relay-resume-fd761eb2b8`、配送PID3070438・追従PID3070579。
固定controller SOURCE1355ad65faとcomputation SOURCE28d976b99は維持する。
APPLIED_SCOPE・ROUND.requested・PROBLEM・CONTROLLER_CONFIG・ROUNDの全hashは変更なし。
UNKNOWNは同じjob・attempt・要求・保留理由で維持し、通常Resumeによるupdated_epochだけが更新された。

新しい配送版からserver2で7件、WSLで1件のQ3結果を中央へ回収した時点を保存した。
再度の配送選択はrelay・追従ともALREADY_CURRENTを返し、PIDの追加・再起動はなかった。
区間追従の複数世代とStop終了は人工検査で確認しており、実機の全区間を完走した主張ではない。
これで現runの通常Stop/Resume後の手動再適用は不要になった。
公開同期は従来のRESULT_SYNC_PENDINGであり、本件の配送検収成功と分ける。
原記録は現runの `recovery/relay-normal-resume-20261120/`。
次段階はCodexが通常探索のPBS・WSLで準備・信号の実実行と回収を確認する。

## 2026-11-20：離散κの後続候補を準備待ちに補充

利用者は指定済み130条件の中で、独立したタスクを切らさない方を優先した。
現runの第5区間ではPR-DFの準備対象8候補のうち、WSLのLD5・絶対κ16の価格較正が残り、
同じLD5のもう一方の幅は入力共有の順番を待った。空いている別splitの枠にも、
次区間の候補は現区間が終わるまで登録されなかった。

分散準備・live refill・work-conservingを有効にした実行では、選択に結果を要しない
明示的な離散κグリッドの未登録点を準備待ちリストに補充する。既存のsplit別の順序、
同時準備枠、保存入力照合、工程依存、予算、登録済みの保留を適用する。
保存状態を使った登録の比較では、次の候補が8件から28件になり、既存点はすべて一致した。
これはκ16・32・64の指定点の登録で、実行件数や性能の測定ではない。
候補補充・離散scope・後着READY・並列準備・工程別予算の人工検査35件はPASSした。

現runの固定版は稼働中。既存の継続照合は、終了receiptのない準備attempt
`prep-11e8784777b094024911f73d` を `UNKNOWN` として検出するため、新版への切替は保留する。
元のattemptと結果を保持し、現runを継続する。新版での実割当・回収の検収は、
既存の継続条件を満たして切り替えられる段階で行う。

### 同じrunでの適用

利用者は現runへの適用を明示した。通常Resumeは、中央SERVICEで対象runに指定された場合に限り、
別配置したスケジュール入口を選ぶ。既存のGateway、同じ台帳、固定設定照合と§10.5を通り、
科学provider・Git locators・import先は元のcheckoutを使う。追加の入口はrun ID、元SOURCE、
対象入口、版のファイルハッシュを照合する。元の固定ファイルは書き換えず、実際のスケジュール版を
`NORMAL_DRIVER.json` の `implementation` に明記する。元attemptを別runへ移す必要はない。

通常入口・固定版検査・再開時の配送を含む人工回帰は38件PASS。旧環境を要する1件はskip。

### PBSの短時間再割当を削減

補充の適用後、PBS割当267589、267590、267591は科学タスクを所有する前に
`PBS_IDLE_ALLOCATION_RESIZE_DRAIN` で終了した。最初の2割当は確保228 CPU秒に対し
実使用16 CPU秒だった。準備にメモリ実測がない場合、350 GiBの枠を単独で予約するため、
先に詰めると同じ割当のQ3が入らない場合がある。

PBSの選択版では、workerが受付可能になってから従来のreadiness待機時間を使って初回の照合と
取得を待つ。現在の割当に適合する要求がある場合は、要求全体の合計メモリだけで縮退・再投入しない。
複数の需要があるときは使用量を見積もれるタスクを先に詰め、未計測工程のみの場合は元の
メモリ全量保護を適用する。施設のCPU・RAM、walltime、適格性、既存attempt所有は維持する。

中央の通常Resumeで対象runだけにPBS管理版を指定し、計算ノードの固定provider・worker、
PBSの固定設定と元の科学checkoutは引き続き使う。関連人工回帰40件PASS、旧外部配置を要する
1件skip。別の旧 `test_rpe_pbs_release_binding` 2件は実装が使う `_validate_acquisition` と
テストが呼ぶ旧 `_acquire` の不一致で失敗し、今回変更していない。実機のCPU時間と回収は別に確認する。

### 価格較正・RomuTailの増枠・WSLキャッシュ照合の試行

利用者は現探索の資源配分の試行を指示した。次の通常Resumeでは既存のPBS管理版も維持し、
RomuTailの `DF_CALIBRATION` の同時軸数を2→4、server2を3→6にした実行profileを使う。
既存の科学SOURCE、計算条件、回路・seed・結果の所有と予算は保持する。
server2は従来の30 CPU上限と自動枠を保持し、コピーしたプロファイルを通常のreadinessで照合する。
RomuTailの2枠目は、独立軸並列対応の価格較正同士で、既存のCPU枠内かつ前回の計測ピークを
基にしたメモリ余裕がある場合に限る。Q3と未計測の準備工程には単独枠を維持する。

WSLのrelayは新しい状態応答で入力bundleが揃っている要求を同じ確認回数の候補で先に扱う。
キャッシュ照会の往復を省き、workerでの既存固定入力照合は実施する。WSLは既存の4枠を使用する。
76件の人工回帰PASS、旧外部配置を要する1件skip。保存要求でRomuTail4軸、server2 6軸の
実機metadata照合が両方PASS。稼働中のattemptを中断せず、次の通常Resumeで新しいprofileを選ぶ。

### 2026-11-20 23時台の候補復帰と配送の短縮

対象は `search-e6361f01b6a480718127`。運用版 `9cd4b4729f` でPR-DFの受入済み構成・
信号locatorをQ3配送に渡し、PBSのgatewayで既知の割当の照会をまとめた。保留済み3要求の
新旧locatorはGit blobのバイトが同一だった。各要求で旧参照の `Canonical bridge mismatch` と、
受入receiptの参照を使った `DRY_CHECK_PASS` を再現した。固定計算は既存の通常runから実行した。

同じrunを通常Stopで排出してから通常Resumeで継続した。元の `APPLIED_SCOPE.json`、
`CONTROLLER_CONFIG.json`、`PROBLEM.json`、`ROUND.json`、`ROUND.requested.json`、
PBSの固定設定、および既存のUNKNOWNのjob・attempt・要求SHA・detailを照合した。
元の科学controller SOURCEは1355ad65fa、科学Q3 SOURCEは28d976b99。中央選択と
受入・配送の実使用版は `NORMAL_DRIVER.json` と `distributed/relay/TRANSPORT.json` に残る。
停止・再開には通常入口と同じrun IDを使う。

PBSの既存5割当の照会は同一内容で、個別照会の中央値1.051秒から一括照会の0.223秒へ短縮した。
実探索でPBS投入から使用可能になるまでの時間は、対照267651・267652の102.4～102.6秒から、
変更後267653・267654の67.3～71.5秒となった。267653では2 CPUを220秒確保し、
実CPU時間44秒で10.0%。対照267651は4 CPUを342秒確保し、実CPU時間48秒で3.5%だった。
タスク数とCPU数が異なる実測である。次の割当267654は16 CPUで8件のQ3が同じ割当に入り、
全件を中央で回収しPBSの終了状態 `F` を確認した。確保291秒、実CPU時間253秒で、
確保CPU時間に対する実使用率は5.43%だった。

server2とWSLの中継は `fd853684d9` を同一invocationへ適用した。既存経路の人工通信で、
server2のSSHは中央値0.402秒から0.013秒、WSLの小さいJSON往復は2.251秒から1.899秒になった。
23:20の通常中継計時では、server2の状態照会中央値は0.965秒（21回）から0.586秒（46回）、
WSLは1.566秒（9回）から1.103秒（29回）だった。
通常Resume後、22件をRomuTail、15件をserver2、15件をWSL、9件をPBSから回収した。
旧入力参照で保留されたPR-DF 7件を4環境で回収し、同じjobの重複割当・受領・採用は0件だった。
固定科学条件と容量上限を保持し、PBSは準備済み需要に従って資源を決める。
候補・PBS管理・通常入口の人工回帰133件と中継の人工回帰19件が通過した。

実機の選択前後と性能照合の保存場所はRomuTailの
`.rpe-central/planning/rpe-fleet-dispatch-efficiency-20261120/` の `MEASURED.json` を参照。
共通手順は[分散探索の運用ルール](../rpe_distributed_operations.md#空き枠容量障害の切り分け)を使う。

### 2026-11-20 23:41 JSTのQ3まとめ配送とPF compile配置検収

同じrunのrelayを `5d29988230` に更新した。23:41 JSTまでにWSLへQ3 14件を7回の
まとめ配送で送り、同じattemptで全件の実行開始・中央回収が確認できた。配送往復を7回削減した。
同じ人工メッセージ4件の現地試験では、server2の中央値が2.429秒から0.599秒、
WSLは5.339秒から1.447秒になった。各3回。実runのWSL PR-DF割当から開始の
中央値は19.827秒（16件）と20.295秒（9件）で、実開始待ちは同程度だった。
server2では同じ方式別の観測でPR-DFが9.721秒（12件）と6.679秒（5件）、
TRUNCATED_PFが10.075秒（5件）と5.686秒（5件）だった。各観測の処理内容は異なる。
人工回帰99件が通り、まとめ配送の応答欠落でも6件の所有と回収は各1回だった。

PF_COMPILEとPF_HOLDOUT_COMPILEは、保存済み要求でRomuTail、WSL、PBSの
固定SOURCE・Git入力・実Pythonのdry-checkを各2件通過した。
PBSの作業専用checkoutでは不足する4つの前提コミットと、385個の履歴コミットを
既存Gitキャッシュへ補い、固定HEADとscienceファイルを照合した。専用Pythonとして
既存の `venv311rc1-pauli-signal-chem05` を指定した。
このrunの次回通常Resumeに4ホストの配置を登録した。PBS予約は標準較正で12 CPU・32 GiB、
独立検証で2 CPU・8 GiBとした。固定providerのround数と既存メモリ見積りを根拠にした初期設定である。

23:41 JST時点では、現行controllerが稼働中で、compile配置の実運用比較を通常Resumeへ残した。実機記録は
`.rpe-central/planning/rpe-fleet-compile-batching-20261120/` の
`TRANSPORT_MEASURED.json`、`MEASURED.json`、各ホストのmetadataとResume選択記録を使う。

### 2026-11-20 23:46 JSTの通常Resumeと実行確認

通常Stopは23:44 JSTに終了コード0で完了した。旧PBS割当 `267659.CPUcluster-PRISM`
の終了と、旧invocationの実行中・転送待ちが0件であることを照合した。通常Resumeで同じrun
`search-e6361f01b6a480718127` を `5d29988230` で再開した。固定入力と既存の未確定
attemptを保全した。23:51 JSTにこのinvocationの中継を復旧し、4ホストのQ3の中央回収を確認した。

通常キューがRomuTailへ割り当てた `PF_COMPILE` 2件は、固定providerの子がサーバー2の
絶対パスからorigin configを読んだため、回路生成前に終了した。中央は失敗結果を回収した。
2件のattemptと予算消費を保持し、自動再試行は行わない。旧版のdry-checkでは子の起動時に
必要な設定が要求へ宣言されていなかった。修正版では、origin config、固定provider、
保存H₄ snapshotとreceiptを要求へ含める。子プロセスは照合済みのホスト別パスを使って
従来の科学providerに委譲する。保存入力と固定providerの照合および人工回帰を実施した。

再開後、PBSは2 CPU割当でQ3を回収し、その後、適格な待機Q3 3件に合わせて6 CPUを
確保した。途中時点の計時は `LIVE_RESUME_MEASURED.json` を参照する。
compileの実完了および6 CPUに増枠した後の定常利用率はこの時点では判定できない。
修正版のcompile adapterは次の配置用で、現在のinvocationは再起動していない。
