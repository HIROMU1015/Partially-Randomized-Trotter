# 2026-10-06 Track B SP-1 one-shot結果とGPTへのSTOP

source S=`0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`、review R=`d0502f22039dc0b5de227ce4b7b5a229c55f1e74`。
利用者のPASS_FOR_SEPARATE_SP1_AUTHORIZATIONは準備P=`c2a49f72eaffba4514743689eb8ccee02ca060a8`へ記録済み。
別の明示実行指示を受領し、Sから独立branch/worktree `track-b-sp1-one-shot-execution-20261006`を作成。
Sの直接子A=`9630e06122172af238ac5219bec6dfc8b01ca837`はauthorization JSON＋receipt二pathだけを変更。
source/contract/tool/threshold/caps、原SP-0.5結果/marker、旧STOPを保持した。

clean Aとruntime/23 critical identities/fresh marker条件を照合して、固定runnerを一回だけ実行。
SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW。12 wrappers/48 mask rows/96 axesを完了し、mandatory STOP到達。
42 rows適格、6 rowsはモデルshot cap。technical timeout/memory/output/numeric failureなし。
wall約6.97 s、CPU約6.94 s、peak RSS約279.85 MiB。quantum shots/trajectoryは生成していない。

保存値はA/B-DとC-DRでn=8/16のgain→n=32のloss→n=64のcapを示した。
CのD-only/R-onlyに5% material gainは0。raw gain10のうちduplicate4を除いて独立gain6。
この記述から研究方針の採否や新規性をCodexで判断しない。

[結果照合](../../tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)、
[raw result/marker/保存値audit/manifest](../../../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)。
stdlib saved-field auditはPASS。result/markerのhashを保持し、matrix/guard/coefficients/shots/resourceを再評価しない。
追加synthesis、DF/Hamiltonian/NPZ、GPU、whole-wrapper compile、full suiteは0。
sourceの59 focused testsは既存local pass。actual compiled総費用、最小shot数、一般D/R則を主張しない。

既知crossoverの再確認を超える問いが残るか、主線closure/縮小/追加検証の必要性・範囲は
[GPT側の研究再評価](../../tracks/algorithm_codesign/sp1_post_run_gpt_review_request_20261006.md)へ戻す。
結果がpositiveでも次stageへ自動進行せず、過去BF/BM STOPやAの証拠境界を維持する。
