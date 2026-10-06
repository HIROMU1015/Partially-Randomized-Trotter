# R1 v2 source review request — 2026-10-06

**準備完了、R1 science未認可、mandatory STOP。**
Base `61dd534567fda5c7348fdc688814089eb26a3561`。
Branch `track-b-rte-reallocation-r1-source-preparation-20261006`。
publication/source Sのfull commit SHAとGitHub固定URLはcommit/push後の利用者向け報告に従う。
source Sは本文を含むcommitであり、自己参照SHAをauthorizationへ埋めない。

## Reviewの読み順

1. [Preregistration v2](rte_reallocation_r1_preregistration_v2.md)：狭いRQ、baseline、context role、resource vector、cap。
2. [Native/phase/error semantics](rte_reallocation_r1_native_semantics_v1.md)。
3. [Source](../../../src/trottertracks/algorithm_codesign/rte_reallocation/)と
   [runner](../../../scripts/tracks/algorithm_codesign/run_r1_rte_reallocation.py)。
4. [Contract](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/contract_v2.json)、
   [126-key inventory](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/synthesis_key_inventory_v1.json)、
   [source manifest](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json)。
5. [Focused test receipt](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/focused_verification_v1.json)、
   [test output](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/focused_tests.txt)、
   [runtime receipt](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/runtime_identity_check_v1.json)。

## 旧案からの変更と未決事項

- ordinary/PTSC-K0/Aのsame-target I0比較、common PauliだけCTS collected controlを追加。
- distinct-basis **controlled** rowsをprimaryに指定。ordinary circuit rowsはdiagnostic。
- canonical samplingのみ。IS/PAIやdictionaryを追加しない。
- 同じx/sign/context/precisionを保持したが、arm追加で264 rows、42 angles/126 keysへ結果前に改訂。
- 旧channel projective guardを転用せず、up_to_phase=falseとstrict operator guardへ変更。
  joint controlled synthesisの測定biasはfactor2で戻す。
- group midpoint rounding→exact rational IID lawを保ち、numerical displacementをbiasへ戻す。
- primaryはresource vector、secondaryは共通finite-confidence forecast。
  materialityの自動GOはなく、成功terminalもawaiting GPT reviewで必ずSTOP。

このdistinct-basis toyもPauli展開可能。I0/I1の情報境界は今回のcontractで指定したもので、
I1取得の不可能性やDF scaleのclassical-cost advantageは示さない。
native countsもadditive synthesized primitivesであり、強いwhole-circuit optimizerとの優位は未検証。
**この限定scopeでR1を行う情報価値があるかをGPTが確認する。**
axis accuracy / confidence / cap / resource-vector routeは今回の具体的技術案で、
GPT source reviewで採否を閉じる。publication priority、論文claim、DF適用判断は未確立。

## Test/source stateと次のauthorization

27 focused tests PASS。phase、basis、complement、finite mean、rounding/canonical consistency、
strict phase guard、confidence bias、zero baseline、launch/one-shotをoff-domain synthetic fixturesで検証。
tool metadata/source identity一致、install/registered synthesis/resource evaluationは0。
共通API変更0、旧R0/R0.5・SP/BF/BM/BS結果・marker・authorization・Track Aは保持した。

authorizationはpending、source_commit=null、science_execution_authorized=false。
本資料はR1実行を承認しない。GPT source reviewを通過した場合だけ、
source Sのdirect child Aでauthorization JSONと任意receiptのみ変更し、
利用者の明示的なone-shot実行指示を記録する。
S上やreview grandchild、dirty tree、source変更を含むA、既存marker、retryはrunnerが拒否する。

GPTには `PASS_FOR_SEPARATE_R1_AUTHORIZATION` または必要なsource/contract修正を返してほしい。
reviewに通っても、本turnで科学を実行せず、別authorizationへ戻す。
