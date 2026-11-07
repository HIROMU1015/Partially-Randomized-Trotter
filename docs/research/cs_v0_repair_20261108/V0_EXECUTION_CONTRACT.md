# C-S V0：保存schema修復と限定再実行

2026-11-08 JST。ユーザーは今回「意図しない中断は修正して再実行してよい」と明示した。この許可に基づき、型例外で停止したrun01を保全し、修復SOURCE公開照合後にrun02を実行する。別の承認・grant発行は不要。

科学条件は[0bd18bbの採択契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0bd18bb7d982bad32f795558a2caf27d34ba3778/docs/research/cs_v0_activation_20261108/V0_EXECUTION_CONTRACT.md)を変更しない。保存linear H4/1Å/STO-3G/legacyDF12/8system/全4電子参照/LD3/m0q1/δ.02r1K2/coefficient_atol0、N1 actual Q/selected full、D1/R1 factor0、nearest1/256rad/tie-away/π保持、residual r1/K2/root256/proposal160/ηρ10^-12、strict pygridsynth2.0.0/seed0/dps100/10^-6、finite10^-12/local3e-6、最大8positive keys。state/Born/RNG/shot/full compile/N/Gは0。

変更点は以下だけ。

- 保存parameterを既存basis restoreと同じfloat codecで復元してからinverse thetaを反転する。XXPlusYY beta、保存hex、位置・role・phase・gate集合を維持。
- 旧SOURCE_AND_CHECKSは変更せず、新SOURCE.jsonの完全なeffective_base_source_sha256で変更2ファイルを明示的に固定する。入力13件・runtime/profile検査を省略しない。旧既定preflightの動作は不変。
- 新source branchはcodex/cs-v0-repair-20261108、outputはartifacts/cs_v0_science/2026-11-08/campaign_run02、cacheはその内部。旧run01/STOP/ログ/費用を変更しない。旧run01ではnative取得0のため再利用・重複取得はない。

元run01のretry0は履歴のまま保持し、今回は明示許可された技術再実行として別記録にする。採択条件・誤差cap・support・science targetの不成立は科学的STOPであり、格子/r/K/precision/seed/slot変更による救済はしない。新たな技術的中断だけは、原因の証拠・修正人工test・新SOURCE固定を経て復旧する。無条件の科学探索/追加campaignは行わない。

起動は新公開SOURCE SHAを--sourceへ指定し、既存run_cs_v0.py --execute-one-campaignを一回呼ぶ。Phase A–D、call/原本先行保存、positive-key dedup、actual adjoint、UNKNOWNと結果公開は元契約のまま。完了または科学的STOP後、公開照合してmandatory STOP。
