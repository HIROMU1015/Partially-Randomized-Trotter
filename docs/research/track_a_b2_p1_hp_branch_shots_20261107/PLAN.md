# 新HP強PF 5 arm：保存branch・energy・shot会計

2026-11-07 JST / Codex B主エージェント。ユーザー指示による固定5 arm×18roundの保存数値再解析・条件付きshot会計・旧3 armの転記比較・実装/人工test/公開の許可範囲。

- HP入力：`3e316e071fe671aa61b31bc89f83bf4380d0e92f:artifacts/track_a_b2_strong_pf_hp_requalification/2026-11-07/run01/result.json`。科学SOURCE `d401d82ab9e77ab43c6eec3483b3086633dcea6d`、precision SOURCE `75cb38e686ad8b82d00d787531d767e816e1331d`。
- 対象：Full S2 s4/s8、Yoshida4 s2/s4/s8、各m0..17/Q=2^m。旧3 armは`55fa5f5eb121ee01ae963da7ab6e5bbe9a507173`の保存結果をコピーし、decoder/shotを再実行しない。旧5 STOPは過去の別SOURCEのまま。
- 元H_phys/state/DF12、sector/JW/order、P1 `7bc952772e451f27c1654dab6803192f2b204c9f`、ε_E=.00015936001019904 Ha、δ=.02、β=.4、prior[-128,128]/shift0、95%目標、各軸α=.05/36、位相予算.02/.06/.04/.24/.02/.02、u=1e-8・独立閾値1e-9を維持。
- 旧`b2_p1_strong_pf_branch_shots.py`/人工testは固定55fa版を原内容で再利用。新規はHP保存schema接続・scalar変換安定性・全8 arm比較だけ。既存decoder、shot式、PF/HP科学SOURCEを変更しない。
- 既存HP記録/差分helperで80/120桁資格・HP→scalar誤差を確認。保存HP位相を既存decoderへ直接渡した場合の枝と、complex scalarからの枝を照合する。既存tie許容値と変換誤差に対してbranchの距離余裕を確認。既に選ばれた枝のHP liftはenergy変換の診断だけで、新しい枝選択をしない。
- A=1、disk=0、raw=z_PF、保存binary64 PF半径下限を旧`account_saved_round`→`account_radius_round`へ渡す。HP半径と保存scalar半径を囲む隣接floatの両端も同じAPIで評価し、ceilが変わらないことを確認。新しいshot式・数値閾値・予算再配分はしない。
- 5 armのdecoder完了/STOP後にだけ既知energyを含む旧3 arm summaryと元H energy recordを読み、事後照合に使用。保存平均のenergy精度とfinite-shot/実機保証を区別する。
- armは独立processで処理し、worker数は独立5 taskと実OS affinityから選ぶ。任意のCPU/RAM/BLAS上限・共有設定変更は行わない。
- 新出力：`artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/`。5 armのJSON/CSV、summary、旧3+新5のSOURCE付き比較CSVを保存。成功した科学実行は追加しない。
- 共有SOURCE/input/P1不一致なら全体STOP。arm固有のbranch/energy/正半径/shot/変換不安定はそのarm STOP、理由・保存数値を残し他armを継続。HP/PF信号を再生成・正規化・位相補正して救済しない。
- private入力、新step/PF simulation、旧3 arm再計算、新HP再適格化、測定sampling、回路build/compile、価格MC/CX/RZ/depth価格、総費用・最適化・詳細科学レビューは行わない。Codex A/Cのworktree/SOURCE/途中成果を変更しない。

固定SOURCEと人工確認後、上記範囲を追加承認なしで実行。結果・比較表・簡潔REPORTを公開し、remote SHA/tree/全変更内容を確認したらmandatory STOP。価格・総費用へ自動進行しない。
