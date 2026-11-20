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
