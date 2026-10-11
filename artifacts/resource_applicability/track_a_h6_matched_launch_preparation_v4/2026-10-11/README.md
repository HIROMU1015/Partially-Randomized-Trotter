# H6補完：起動準備v4・科学runner v3不変

**PREPARATION_ONLY / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / mandatory STOP / next_stage_authorized=false**。
[契約・認可対象・起動順序](../../../../docs/research/track_a_h6_matched_launch_preparation_contract_v4.md)、[旧起動STOP](../../../../docs/research/track_a_h6_matched_supplement_launch_stop_v3.md)。
science source `5a903aea2492c2d211c1e15e7737df0d50694c12`、metadata source `48b489ca3465ef7f4bb202f61c8ca45e9455271e`。
新出力 `launch_v2`、親を準備・write probeしleafは未使用。旧v3科学source、92候補/shot/compiler/AS/caps/reuseを維持。
科学runner/signal/sampling/build/compile/DF/state生成0、新approved grant0。

- [sealed manifest](sealed_manifest_launch_v4.json)：digest `d350e7dcca318eb729e5cfe43fad0d313cb49ec604f79560aeb3e74f28ce2865` / SHA256 `c7e4dcc354515dd9cdd1bf656aa6cea0cf323fefd204585fc801a1e3f5adb617`
- [identity比較・変更field限定](identity_comparison_v4.json) / [静的source/input/reuse検証](preparation_verification_v4.json)
- [parent準備](output_parent_preparation_v4.json) / [parent再検査](preflight_parent_recheck_v4.json)
- [20 synthetic/process PASS](synthetic_validation_v4.json) / [JUnit](synthetic_pytest_v4.xml) / [log](synthetic_pytest_v4.log)
- [未承認grant template](authorization_request_template_v4.json) / [baseline](baseline_before_preparation_v4.json) / [保全](preservation_v4.json)
- [独立remote照合](remote_preparation_verification_v4.json)
- [preflight source](../../../../scripts/resource_applicability/h6_matched_launch_preflight_v4.py) / [metadata sealer](../../../../scripts/resource_applicability/prepare_track_a_h6_matched_launch_v4.py)

旧92 signal/84 B3/449 group/902 wrapperとmissing143を維持。原447を先行し成功時だけ残142、全592後に原上位候補×独立32確認する計画。
現在は別の一回実行認可待ち。旧grantは使用しない。科学runnerを準備で起動しない。
GitHubでは空parent directoryが保存されないため、起動直前にこのpreflightで親・権限・leafを再確認する。
20 local検査はfilesystem/metadataの証拠で、実H6 q8のメモリfeasibilityや新費用の証拠ではない。
