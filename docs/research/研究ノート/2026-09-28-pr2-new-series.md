# 2026-09-28 PR-2別系列再開ノート

## 08:56 JST V0–V3完了・V4前停止

GPTレビューは旧S0をPASSへ変更せず、`AMEND_AND_RESTART_FROM_NEW_S0`を推奨した。旧pilot入力の
read-only一巡監査では完全配列を回収できなかったため、旧`STOP_INPUT_REPRODUCTION_MISMATCH`と
`S1_authorized=false`を維持し、保存済みdevelopment snapshotを別系列
`pr2-rebaseline-de7a5492-v1`の入力とした。これは性能で選んだ入力でも、旧pilotと同一と認定した入力でもない。

外部レビュー、V0監査、新系列amendment v4、authorization manifestをspecification commit `30ea857`、
専用module/runner/testをsource commit `ef86868`で結果前固定した。専用testは7 passedだった。

V1ではsnapshotの二回load digest、shape/dtype、finite値、Hermiticity、state norm、sector整合、Rayleigh
residualを検査して通過した。V2ではrank 3/6/9のB2-G/B2-Wについてexact cover、sampling sign、確率和、
identity coefficient、repeat preparation、`H_D+H_R=H`を検査して通過した。G/Wは全rankで同じordered
prefixとなり、`collapse_B2_G_and_B2_W=true`である。

統合statusは`S0_PRIME_PASS_V4_REVIEW_REQUIRED`、result fingerprintは
`b210b394e9cd5a8eded947b0fd12cefe19ce9f27eb6b8140b3e863df73ea7961`、artifact SHA-256は
`0eb22c813eb838169eb455334146140467ebbc5636db78bd923b1e6bdaed46d8`。分子計算、signal、trajectory、
compile、quantum shot、held-out NPZ loadはいずれも0件である。

`V4_authorized=false`、`S1_prime_authorized=false`、`automatic_next_stage=null`として停止した。これは
resource優位性またはheld-out transferの結果ではない。次は結果packetを外部確認し、V4/S1′を行うなら
別の明示的authorizationを結果前に固定する。
