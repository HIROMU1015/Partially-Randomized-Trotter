# 2026-10-10 Track A：H4限定science manifest固定・seal

利用者の継続指示を直前の「manifest固定・seal、未認可gate確認、公開後STOP」へ適用する。
[新契約・報告](../track_a_ax2b_h4_limited_seal_v1.md)を追加し、元H4-P親STOP/実行source/結果/freezeを保存する。
再監査済みboundsをexact copyし、現在のscience source/input/env/CPU3/専用output/旧8 cell・capsを束縛する。
科学sourceとH4-P実行sourceのclosureを別に記録。今回追加するのはstdlib seal工具と合成testsだけで、科学backendは変更しない。
35合成tests pass。execution_plan_sealed=trueは条件固定だけで、science_authorized=false、launch_allowed=false。
専用outputはmetadataとして固定し、凍結v2 launcherの別grantへ同じpathを結び付ける後続作業は残る。今回grantを作らない。
新分子計算/native準備/数値array decode/probe/sampling/circuit build/compile0、H4-P再実行0。
公開後mandatory STOP。H4 scienceは別の明示認可、H6/H8/GPUは未認可。H6_NOT_AUTHORIZED、DRAFT_NOT_AUTHORIZATIONを維持する。
