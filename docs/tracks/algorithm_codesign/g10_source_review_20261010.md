# G10 source review / GPT引継ぎ入口

状態：**READY_FOR_G10_SOURCE_REVIEW_SEPARATE_AUTHORIZATION_PENDING**。
利用者採用G9 v2 review §13の一束に沿い、保存policy下界・claim整理・次数比較sourceを準備した。
登録P3/P7のsynthesis/native比較は0、G10 marker absent。m5原result/marker/authorization/STOP不変。

## 読む順序

1. [採用GPT review](../../research/track_b_G9_v2_scientific_review_20261010.md) §13。
2. [保存policy独立監査](g10_saved_policy_and_claim_audit_20261010.md)と[claim/prior-art map](g10_prior_art_and_claim_map_20261010.md)。
3. [数学・実行契約](g10_degree_comparison_contract_20261010.md)、[machine contract](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json)。
4. new modules `g10_generator.py` / `g10_reference.py` / `g10_comparison.py` / `g10_launch.py` / `g10_saved.py`、
   [future runner](../../../scripts/tracks/algorithm_codesign/g10_degree_matched_native.py)。
5. [focused tests](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/focused_tests_v1.json)、
   [provenance](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/provenance_audit_v1.json)、
   [source manifest](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/source_manifest_v1.json)、
   [evidence manifest](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/evidence_manifest_v1.json)。

## 完了した準備

- G9 raw exact値からCTSのreview粗下界を独立確認。945 direct保存IRのTも再計数。
  ordinary/partial/P3にも固定辞書の任意proposal policy分離が残る。general full/closed P5の同familyはこの下界では未分離。
- closed P5+6/7 ordinary pair、general P3/P7、literal P3/P7 CTSを同じ有限operator taskへ接続するsourceを追加。
- m3=5 / m5=6 / m7=6の17 direct rows、34 axes。m5は保存native資料のcommon-policy rebudgetのみ。
- static key upper181/cache192/new acquisition162、support upper11319/cap12000、runtime/capsを結果前固定。
- 41 focused tests PASS：off-domain P7 mean、Q(sqrt2)/独立matrix、negative controlled phase、zero-T、
  proposal補正、参照依存遮断、saved-only rebudget、API string/stub、remote/hash/direct-child/consumed-marker拒否。
- fixed runtime identity PASS。pending launch拒否。G10 registered science/matrix/trace/backend取得0。

## 準備資料のread-only照合

[stdlib verifier](../../../scripts/tracks/algorithm_codesign/verify_g10_source_preparation.py)はhash/AST/保存記録/
未認可状態だけを確認し、登録angle・budget・matrix・sampler・backendを評価しない。
[照合記録](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/preparation_verification_v1.json)。

## 実施時のboundary

現commitはsource preparationで、execution authorizationではない。
[新authorization](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/authorization.json)はnull/falseのまま。
source Sのfinal review後に別authorization-only child Aと新source-bound明示one-shot指示が必要。
このgateは[source contract](g10_degree_comparison_contract_20261010.md)と`g10_launch.py`に固定し、旧G9権限を流用しない。
承認/実行の分割を各testへ増やさず、G10一束で登録比較を行い全outcomeでSTOPする。

source-bound実施コマンド（本準備では未実行）：

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/abe/Project/prt-worktrees/track-b-sp05-economics-preparation-20261006/.venv/bin/python -B \
  scripts/tracks/algorithm_codesign/g10_degree_matched_native.py --source-commit <S>
```

結果後は既定claim/evidenceの集約・構成noteへの縮小・必要な一般化一つの採否をGPTが判断する。
registered canonical lawと arbitrary-proposal lowerを区別し、未分離を失敗/非改善とみなさない。
G10の勝敗・technical failureを理由にm9/新provider/precision/solver/DFへ自動進行しない。
