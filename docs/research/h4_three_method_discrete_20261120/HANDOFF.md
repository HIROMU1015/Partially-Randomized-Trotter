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
