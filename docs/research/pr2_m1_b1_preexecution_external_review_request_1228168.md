# GPTへのPR-2 M1-B1実行前レビュー依頼

## 依頼

PR-2 matched-accuracy resource studyについて、M1-Aの`SELECTION_LIMITED`後に作成したM1-B1 bounded
direct-compile契約をレビューしてください。今回は契約・実装・schema・zero-compute planだけを確認し、
M1-B1科学計算、trajectory sampling、circuit build、compile、held-out accessは実行しないでください。

## 固定identity

- repository：`HIROMU1015/Partially-Randomized-Trotter`
- M1-A result commit：`3c1831e326c27c5f679b3820997f27916d26ed9f`
- M1-A result SHA-256：`1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086`
- M1-A result fingerprint：`422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e`
- M1-B1 source commit：`12281687fe13c13ac19688d328f9da26a1d63f34`
- implementation authorization SHA-256：
  `ee0301f05fd0dc348cd7a2c522fd030271d0ecd5a892fa615798875d5c7a6146`
- zero-compute plan SHA-256：
  `d0c2234cb787b57af85457a6232504be39bbb2140d60f9b67bbda3d252622e6c`
- zero-compute plan fingerprint：
  `94592dbddce9b21cfe9fd31c61c578943264002655379e255aa36161072b5814`
- status：`M1_B1_BOUNDED_COMPILE_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED`

主要ファイルは次である。

- `docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`
- `src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py`
- `scripts/run_pr2_matched_accuracy_m1_b1_contract.py`
- `tests/test_pr2_matched_accuracy_m1_b1_contract.py`
- `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/`

## 変更しない監査結果

M1-Aは210候補中206候補をaccuracy適格、B2/B3 random候補194/194を適格とした。proxy frontier 64件の
うち52件が旧16-cell cap外に残り、compile 0で`SELECTION_LIMITED`となった。この結果を消去せず、
「粗いaction proxyでは候補を安全に16件へ圧縮できなかった」という監査として保存する。M1-B1は旧selector
を改良した結果ではない。

## レビュー対象のM1-B1契約

1. M1-Aでaccuracy適格だったB2 145＋B3 49、計194 fingerprintを追加・除外せず使う。
2. random各cellは32 trajectoryを固定し、同じtrajectoryをcosine/sine二軸で共有する。
3. random wrapperは`194 × 32 × 2 = 12,416`。
4. B0 12＋B1 4の全16 baseline cellを二軸でcompileし32 wrapperとする。accuracy不適格B0 4件は
   baseline completeness用であり、matched-accuracy frontierへ入れない。
5. 総上限は12,448 full wrappers、future process worker最大6、各BLAS thread 1。
6. candidate fingerprint、axis、trajectory seed/index、compiler identity、source commitが完全一致する
   cache/checkpointだけを再利用し、cross-cell reuseを禁止する。
7. B1ではsignalを再評価せず、32-trajectory actual compiled resource mapを作った時点で停止する。
8. 追加96 trajectory、held-out候補確定・access、transfer、winner精密化、S3を認可しない。
9. B1後は`CONTINUE_RESOURCE_STUDY / NARROW_TO_TECHNICAL_NOTE / STOP_DUPLICATIVE /
   COMPILE_RESULT_INCONCLUSIVE`の研究判断へ戻す。

zero-compute planは194 random＋16 baseline cellと12,448個のunique wrapper cache keyを固定した。専用testは
9 passed、M1全focused testsは35 passedである。development/held-out NPZ load、signal再評価、trajectory、
occurrence、circuit、compiler invocation、full wrapper compile、quantum shot、GPUは全て0である。

## 確認してほしい点

1. M1-A artifactからの194 random fingerprint固定に結果依存の追加・除外が入り込んでいないか。
2. B0/B1全16 cellを測り、不適格4 cellをmatched-accuracy frontierから除く扱いが妥当か。
3. cosine/sineが同じtrajectory seedを共有しつつwrapper cache keyをaxisで分離する規則が妥当か。
4. source/compiler/candidate/axis/trajectoryを含むcache/checkpoint identityは再利用条件として十分か。
5. 12,448 wrapper、最大6 worker、32 trajectoryで強制停止する上限がbounded pilotとして妥当か。
6. B1後の四分岐と、追加96/held-out/transferを別reviewへ送る停止規則が十分か。
7. このbundleを基礎に、別commitでresult-prior M1-B1 execution authorizationを作る段階へ進めるか。

## 回答形式

次のいずれか一つを先頭に示してください。

- `APPROVE_RESULT_PRIOR_M1_B1_AUTHORIZATION_DRAFT`
- `REVISE_CONTRACT_BEFORE_AUTHORIZATION`
- `STOP_OR_NARROW_BEFORE_COMPILE`

その後、重大な問題、必要な最小修正、実行前に追加で固定すべきidentity/resource/testだけを列挙してください。
このreview回答だけではM1-B1を実行せず、別のexecution authorizationをcommit・reviewするまで停止します。
