# 2026-10-12 N3旧条件族診断 v1 準備履歴

## 完了済み科学レビュー受領

S1の保存幅差を支持し、旧族の保存coverageを越えた回収／分離診断を限定継続するレビューを受領した。
利用者はCodexで可能な準備を次の科学判断・明示実行指示の地点まで進めるよう指定した。
レビュー第11節は新診断への別認可を必要とするため、source/独立checker/synthetic/input/plan/公開までが今回の範囲。
[準備報告](../n3_old_family_v1_preparation_20261012.md)、[数学・実行契約](../n3_old_family_v1_mathematical_contract_20261012.md)。

## 実装・synthetic

μ0への単調性による一変数等価性と、PSDなWの両端strict負traceを実装した。
原FIVE→old_candidateの対応を静的確認し、NORM不成立を別に判定する契約を保持する。
23 synthetic PASS。dense9 testに恣意的なcheck-count期待の失敗が一度あり、exact certificate PASSの確認へ訂正。raw logを保持。
新しいS1診断0。元OCC勝者の有理上端・係数・U/cはlosslessコピーだけ。元primary/gate/STOPとN1/N2/別Trackを保持する。

## Source固定と新scope plan

source 1aa12ab0b777b28f1428d9bb8417af93c4376aae、source fingerprint 46baa537269db923dd8b8c95e4788555471c60bad1572d3a01ec1e70245c6488、plan SHA ef3c66d2e01e47027d9d382afc324aab9b7263aab3d550c3576c31e0f5bd6ff5。
S1新診断0、未認可template、Mandatory STOP。準備だけを公開・remoteでidentity確認する。

## 公開・remote identity確認

準備payload 23bf77c6cfffe798d54e5340d59ef5be26e9e9ae、source 1aa12ab0b777b28f1428d9bb8417af93c4376aae。59 blobsをGitHub取得・byte照合、preflight PASS、新S1診断0。
元結果・STOP・manifest98保全。root外部配置整理は同byteの移動51件を確認し、5外部metadata編集を保持。
次は固定一回診断の明示指示、科学レビューの同内容再実施は不要。[publication](../n3_old_family_v1_publication_20261012.md)。
