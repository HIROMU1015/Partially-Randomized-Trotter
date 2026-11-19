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
