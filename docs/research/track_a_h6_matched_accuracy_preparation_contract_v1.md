# Track A H6 精度一致資源比較の準備契約 v1

2026-10-11 JST。利用者が共有した[H6 pilot後のGPT独立レビュー](track_a_h6_technical_pilot_post_review_2026-10-11.md)に沿い、次のCodex作業をH6の精度一致資源比較の設計・準備に定める。RQ-Rを主、RQ-P1を補助とし、同じ信号精度でB2の回路短縮が強い決定論PFとの比較でも残る条件を調べる。

本段階はレビューの採用と準備範囲の記録まで。追加科学計算の実行認可、実行用source・plan・入力・環境のsealは未発行である。`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、`next_stage_authorized=false`を維持する。

## レビューと一次結果の来歴

レビュー原文をbytes一致で保存した。SHA-256は`ccab6a911af17092e6baffcad89a8d8de4c026644c4344a4baff050e1a2906fb`。利用者の原文とZIPは編集しない。

固定結果は`eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2`、science sourceは`0b04886869efb9d08b07d6517300da2bc0123f4a`、実行sealは`fc97d46db9dc84d255eaf317d1fea75fd1d1c015`。[原結果索引](track_a_h6_technical_pilot_result_v2.md)と[保存要約](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/result_summary_v2.json)を保持する。

[補助ZIP](../../artifacts/resource_applicability/track_a_h6_post_review_adoption/2026-10-11/track_a_h6_pilot_review_scalar_checks_2026-10-11.zip)、[転記入力とレビュー内算術](../../artifacts/resource_applicability/track_a_h6_post_review_adoption/2026-10-11/scalar_supplement/saved_scalar_review_calculations.json)、[転記照合](../../artifacts/resource_applicability/track_a_h6_post_review_adoption/2026-10-11/scalar_identity_audit_v1.json)を別資料として保存する。ZIPのmanifest hash、7 cellの主要scalar、16件のprimary wrapper費用を既存JSONと照合した。補助コードは実行していない。転記一致はレビューの全導出や数値保証を認定するものではない。

## 採用する部分証拠

原statusは`H6_TECHNICAL_PILOT_STOP`、reasonは`PHASE_WALL_CAP:wrapper_cost`。correctness7/7、保存wrapper32/36、欠測7件を維持する。個別・paired費用として使用し、random母平均の精密評価やPRの一般的優位としない。B2は2 cost trajectory、B3は1 cost trajectoryであり、二軸・control共有wrapperを独立標本として数えない。

B3 replica1と4 wrapperの補完を主検証設計の一律前提にしない。旧runを36/36 COMPLETEに変更せず、失われたworker terminalを後から再生成しない。補完する場合は目的・保存seed・未取得taskだけを結んだ別実行とする。

原runのN/Gはnull、accuracyはUNDETERMINED、数値allowanceとground stateは未認定のまま。後続の保存値再集計・shot会計は別scopeと別成果物にする。

## 維持する科学対象

linear H6、1.00 Å、STO-3G、12 modes、α3/β3、sector400、採用DF rank19、T=0.8、同じ保存stateとsnapshotを使う。全19 signed fragmentと生成順、tol-only `1e-8`、coefficient cutoff0、採用weighted Hermitization政策を継承する。新しいSCF・DF分解・state solverを通常の前提にしない。

有限時間coherent-signal taskの資源研究であり、chemical-accuracy energy estimation、ground-state証明、全RPE段・物理実行時間・fault-tolerant総費用へ読み替えない。

## 比較集合と結果前に決める項目

| 項目 | 採用する方針 | 次の準備で固定する内容 |
|---|---|---|
| 決定論対照 | B1 S2とglobal Yoshida S4を含める | 共通q集合、実装・compilerのidentity、技術失敗の表示 |
| B0とB2 | 共通の少数generation-prefixを使う | prefix5/10/15は出発案。最終集合を結果前に列挙 |
| outer step | qを主要自由度とし、同q比較だけで方式を選ばない | q1/q2をanchor、q4/q8を初期拡張案。最大qと有限候補生成規則 |
| finite tail | R=qr、整数r、finite law・normalizationを維持 | 有限R/K集合とB・finite誤差会計。Rだけの無制限拡張をしない |
| B3 | normalization負担の限定診断を残す | 安価なfinite式の確認対象。family全体を現点で排除しない |
| 要求精度 | ε_sig=0.05/0.01/0.005/0.001を中心にする | 共通の軸配分・成功確率・strict headroom判定 |
| control | symmetric_directionalをprimaryとする | ordinaryの限定paired感度taskと必要probe coverage |
| 指標 | no-prep measured wrapperのRZを主、CX/depth/sizeを副とする | 軸別費用、normalization・bias margin・shot込み会計、共通準備cost Pの感度 |

具体的なgrid・標本数・除外閾値はまだ固定していない。レビュー由来の例を認可済みの全直積としない。最初に適格となる最小qを常に最良とせず、同じ有限集合で統計余裕と回路費用の競合を評価する。

## signal評価と費用取得の分離

固定候補生成、signal/bias/Bと数値診断、精度会計とcoverage確認、actual primary-wrapper費用、主要比較の独立確認標本の順で準備する。全候補集合・signal取得集合・cost取得集合・technical unavailable・未取得を別に保存する。

予測costによるheuristic shortlistingは探索的な選抜である。未取得候補を黙って除外して全method最適性を主張せず、必要なcoverageが不足すればSELECTION_LIMITEDまたはUNDETERMINEDを許す。資源guard超過を科学的不適格や真の高費用の確定値として扱わない。

deterministicではfull-wrapperのzと対応signal recordの照合を追加する設計にする。randomの一つのtrajectoryをensemble平均に等置せず、event・確率・phase・normalizationの接続を維持する。

## 数値幅と費用統計

empirical uと保証付きuを分け、指定target・参照・精度／演算順序比較・適用範囲を保存する。現oracle差の最大値をそのまま普遍的uにせず、u=0のレビュー内J表を正式なshot budgetにしない。headroom hが境界に近い候補は数値幅とhの比を検査し、不確かな判定を強制しない。

fresh whole-trajectory lawによる独立量子shotと、回路費用のpaired trajectory標本を分ける。費用探索と確認の標本、層化・rare event/tailの扱い、標本追加・停止・多重比較の規則を結果前に決める。n=2を母平均の完成条件にせず、instruction guardを分布全体のcompiled-cost上界へ流用しない。

## 高速化と並列化の準備

旧[費用port](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py)はcell/replicaとcontrol・probe・軸の回路検査を順次処理する。Numba4のsector作用だけでは回路検証・compileの主負担を並列化できない。[既存executor](../../src/trotterlib/parallel_validation_executor.py)のtask・seed・checkpoint・resource管理を再利用できるか確認し、新版で科学identityを保つ。

primary費用取得とcontrol感度の分離、cell/replica単位のprocess分離、不要な回路・参照の解放、identityを満たす固定構造のcacheを検討する。taskにseedを結合し、paired軸/control・compiler設定・数値精度を保存する。worker数とNumba/BLAS/OMP/Qiskit/Rayon threadsを調整し、同時workerの合計メモリを見積もる。

保存RSSはAS上限や利用可能RAMと同じではない。旧runのhigh-waterからworker数を単純倍増しない。cosine/sineの過去count一致だけで片方のcompileを代用せず、同値性を確認し新契約で明示する。GPUは対応backendと同値性が確認できる場合だけ選ぶ。必要なsynthetic検査と実行計画を用意し、測定していない速度改善率を報告しない。

新runnerのphase/total wall capはnullとし、旧runのcapとSTOPは変更しない。メモリ・出力・log・call/instructionの有限上限、progress、明示中止手段、失敗・欠測保存を維持する。実ホスト・CPU/GPU割当・worker数・合計予算はseal前に固定する。

## 次のCodex作業と完成条件

1. 共通target identity、有限候補生成規則、指標、shot・数値幅・費用統計・未取得処理を含む結果前契約を作る。
2. signal先行評価とprimary費用取得の新版runner/schemaを準備し、高速化・並列化と必要なsynthetic検査を行う。旧凍結sourceを上書きしない。
3. source/plan/input/environment/resource boundsを固定し、必要資料だけをcommit/pushしてremoteから照合する。
4. 具体的な追加計算の対象と回数を示す。新しい明示認可の後に実行し、結果公開・監査の後はmandatory STOPする。

今回の成果は1に進むための採用方針と作業範囲であり、1〜3の実装・固定完了を意味しない。新science runner、sealed manifest、実行grantは未作成。

## GPTへ戻す時点と研究の着地点

次の主要レビューは、H6の精度一致資源mapと主要比較の確認結果が揃い、H8で何を独立検証するかを決める時点に置く。通常の実装修正やsynthetic追加だけで全面的な科学レビューを繰り返さない。

数値矛盾、target/phase/estimatorの意味論変更、不公平な除外、必要baselineの削除、情報漏洩、研究範囲の変更は早期にGPTへ戻す。H6はdevelopment、H8は参照性能を開かずに残し、exposure台帳と条件・予測・モデルのfreezeを後で確認する。旧H4 rank12とH6 tol-only rank19は共通政策のsize scalingとして無条件に並べない。旧FEWの事後fitを移送成功としない。

PRが勝つまで条件を増やし続けない。公平な有限比較と不確かさの評価で利益が残る／失われる条件を説明することを成果とし、同じ限定点の工程完了しか増えない場合は規模拡張を止める。

## 本段階の保存状態

[採用manifest](../../artifacts/resource_applicability/track_a_h6_post_review_adoption/2026-10-11/adoption_manifest_v1.json)と[保全監査](../../artifacts/resource_applicability/track_a_h6_post_review_adoption/2026-10-11/preservation_audit_v1.json)を参照する。本段階は文書とJSON／bytesの静的照合のみで、新科学計算・signal・sampling・回路build/compileは0。資料はローカル未commitであり、公開済みの一次結果とは区別する。
