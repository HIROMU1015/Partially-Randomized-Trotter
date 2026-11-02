# H6：丸め差許容・保存tail基底対応・4 worker一回探索

[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_roundoff_preparation_v1/2026-10-12/README.md)。
利用者が小さい丸め差を許容して本計算を指示。旧環境export待ちを解除し、保存tail basisへnative固有値を対応付けた。
450の元seed/order/choice/support/signを保持。Float metadata30件の差最大2.8e-17、追加重み付き演算差合計4.9e-12。
Gate生成matrix-byte digestの差は同じgate名/positions/angleに結び付けて対応を記録。大きな数値差とdiscrete変更はSTOP。
科学224/基盤233を維持。新portの原cost logicはbyte gateへの限定丸め比較追加以外不変、compiler/科学plan/input generation変更なし。
Deterministic preparation4種は同native hashで、server447を含む受理450group/904wrapperを再利用し、原142の探索だけを一回実行する。
旧/new compiler outputの完全同一性・量子costのbit同一性を確認済みと扱わない。環境来歴を別に保存する。
67 local限定検査PASS。独立agent review/CIではなく軽量差分review。既存H4/venv/共有設定/他job/GPUは変更しない。
新exact grant/source/profile/input/archive/carry/fresh-resource/output/controlにbinding。CPU4worker/AS32/RSS9/aggregate40/RAM64と既存PSI/OOM/5秒guardを維持。
過去actual total=null、停止下限4/32は独立予約として保持、今回の診断2trajectory/16occurrenceは確定消費へ追加。旧消費の返却/resetとcap拡張なし。
一回142の完了/STOP後終了、自動retry・selection/独立32確認/H8へ進まない。

旧strict-source b53271f5/REVIEW1293109eと証拠を保持した。今回SOURCE0365f0a0。

2026-10-12 一回起動確認：coordinator4085189、4worker4085412/4085413/4085414/4085418、450旧partial gate通過。起動直後のresource STOP理由なし。SOURCE0365f0a0、起動REVIEWc613a617。監視下で継続し、完了/STOP後終了。
