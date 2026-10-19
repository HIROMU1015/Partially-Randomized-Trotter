# Global Bundle Bridge V1：ローカル独立技術受入

**実施範囲の独立技術検証を完了。新しいProducer不整合は確認されなかった。証拠不完全26件と科学入口閉鎖を維持し、正式採択・科学GOは行わない。**

[Q1–Q10](handoff.md)、[機械判定](verification.json)、[43ケース](artificial_43_case_verification.json)、[SOURCE](independent_source_verification.md)、[契約](global_bundle_contract_acceptance.md)、[storage/post-close](storage_post_close_acceptance.md)、[費用](global_cost_and_budget_acceptance.md)、[テスト](artificial_regression_results.md)、[残件](pending_and_required_decisions.md)を参照。

|集合|ローカル結果|
|---|---:|
|公開43件|bundle正常2・単条件正常2・技術失敗13・証拠不完全26|
|公開35 mutationの独立再読|integrity mismatch32・証拠不完全3|
|追加fresh人工bundle/fault|正常2・証拠不完全6（元43件とは別）|
|新独立mutation62|integrity mismatch58・証拠不完全4|
|独立回帰の重複しないID|**845 PASS = 旧原版623 + 既存V2独立57 + 新165**|
|別枠：既存Storage V2 Producer unit tests|41 PASS|

43/35/fresh8/新mutation62は新165の内数で、加算しない。Producer報告の828/107や過去集合も加算しない。選択した独立845件のFAIL/ERROR/SKIPは各0。固定server補正版labelの環境差による1 FAILは別集合として明示する。

直接基点は`49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866`。Producer公開HEAD`d24cccbecdf0928fe977080dd7b28c591df0089e`、測定SOURCE`e099dcc461247e81eff981867dc28c3f3b16df5e`はread-only参照で、merge/rebaseしない。source release candidate digestは`f827a93e74e791e58f8c5ec30b73d2165b049fa218eaf600b134ec17fa9a2c6e`であり、Git commit・正式release・実行credentialとは別である。

743固定ファイルの公開bytes/SHA256、元archive27,677,690 bytes・全2,022 memberを独立照合した。bundle n256のcall countersは全16,896。公開人工result15,502,271B・log43,190,850Bはreserved/observed/physicalが一致。実G7の費用・保存量上界やCPU CIではない。

ローカルPython3.10.12、Producer申告Python3.12.3。実行環境差は記録したが、別物理ホストであることはNOT_ESTABLISHED。旧server runtimeのexecutable/library実体をローカルbytesで代用していない。freshローカル測定時のruntime62ファイル等は別途実物照合した。

実G7/F3 import/sampling、新科学計算、保存科学値再解析、正式science SOURCE採択、実authorization/grant/claim発行消費、native quantum処理はすべて0。run_ready=false/science_release=nullを維持。

再現は[手順](reproduce.md)。原証拠は`fixed_reference/`、新人工rawは`raw_parts/`の6partと[manifest](local_artificial_raw_manifest.json)。[per-case](case_reports/)にverification/acceptance/比較CSVを保存し、通常経路でProducer/scheduler/Admission/self-checkerをimportしない。

次の担当はユーザーとGPT。技術契約と未採択条件を判断する。Codexはcommit/push・公開bytes/HEAD・保全照合後mandatory STOPし、実API/authority接続や科学実行へ自動移行しない。
