# 2026-10-22：決定論PFの共通探索域を準備

利用者の指示はm5決定論PFの疎な探索による比較の偏りを検査するため、既存S2/S4と同じprefix/q/mode格子を設計すること。実行承認の範囲は照合・設計・実装準備・人工test・契約までで、実分子signal/回路生成/compile/費用再ランキングは含まない。

現在の入力系列に限定したprefix和集合を採り、H4 DF10 p1..9、H6 DF19 p5..15、q1/2/4/8、Full/B0/B0I、S2/Yoshida4/m5で504候補に固定した。旧H4 DF12とH4既存q16を別表へ残した。m5の古いDF12 planをDF10へ移さず、H6のS4 p5/q2 pilotも再利用照合した。

354候補は保存signalまたは固定endpointがあり、新150候補だけのsignalを追加枠とした。内訳75基礎+75overlay、独立forward75。共通allowanceで最大2520 paired endpointの資格/shotを評価し、適格かつ費用未解決に限る最大373候補746軸のcompile poolを固定した。旧直接費用217候補のidentity接続が全て成立すれば未取得側は最大156候補312軸。科学値・費用による候補選択や新順位の計算は行っていない。

保存済みendpointだけのH6 28候補では共通uの観測原metadataが欠けるため、勝手な再生成を契約へ含めず実行前STOP条件とした。旧native/profileの異なる費用も同じoptions hashだけで受け入れない。これは共通域の完全比較を成立させるための必要条件である。

準備SOURCE 1fe0e60d28e7fc09821ee3d5a014fd5f8c976c19。新23＋既存12/18の53人工test PASS。既存signal人工回帰の最初のimport失敗は、元providerを明示して解消し科学コードは変更していない。旧PR SOURCE・raw・契約・marker・STOPを保全し、今回の実分子call/有効grant/pushは0。専用branchへローカルcommitしmandatory STOP。

[条件・予算・比較設計・欠測・実行順序](../track_a_pf_common_domain_preparation_20261022.md)。次は利用者の追加call予算と共通政策に対する明示承認、および原metadata/identity接続の解決が必要。詳細科学レビュー・論文本文・productionには進んでいない。

## 追記：包括承認による取得・固定比較

利用者がH4/H6共通504・H2別84候補を包括承認。系別pack/source/grantでH2/H4の4 stageと終了監査を完了した。H6は17原metadata回収後も63 B0Iでnorm/closure未回収のため科学call0 STOP。H2全公式はPF低費用、H4は厳しい2精度でPR15.01/37.37%低費用、緩い3精度はPF低費用。旧m5疎格子への優位を全公式PF優位と同一視できないことを確認。旧SOURCE/PR/STOPを変更せず、新84 signal・169 compile候補338軸を保存。[結果と限界](../track_a_pf_common_domain_results_20261022.md)。local commit後mandatory STOP、push0。


## 追記：588候補の正式採用・研究資料統合

ユーザーが結果90f1cd55a0d963953696bd0a27d9d041ff939bb7（H2 84/H4 228/H6 276・未解決0）を有限共通域の正式成果として採用した。
理由は決定論PF側の探索機会を3公式で統一した固定比較が完了し、PR有利/PF有利の両方向を含め保存結果を研究証拠へ接続するため。
[採用報告](../track_a_h246_pf_common_adoption_20261022.md)と[v2補足](../track_a_h246_pf_common_protocol_v2_supplement_20261022.md)を追加。
linear1Å/STO-3G/tol-only1e-8 rank3/10/19、L_D0..2/1..9/5..15・Full3/10/19、q≤8・δ=.8/.4/.2/.1/T=.8/P=0。
全公式比較ではH2の5精度はPF低費用、H4/H6はε=.001/.0005のみPR低費用。H6の保存分母分解はm5 prefix拡張比1で、S2/Yoshida4を含む公式集合の変更が主な影響だった。
既存旧比較を誤りへ書換えず、分母・cohort・政策・取得系列の違いを記録。保存値転記・構造検査だけで新しい科学call/解析/順位計算0。
H6旧COSTの187完成価格受入と新22価格の正常取得を区別し、旧中断・未知exit/reap・STOP・grant・ledgerを保持する。
旧CEM14行/台帳316行のbyte prefixを新証拠版へ保全。主表/Figure2/独立確認不変、経験的allowance/選択効果/未観測rare event/母平均保証なし。
AD03正式採択・最終freeze・詳細科学レビュー・新科学実行は未開始。専用branchへcommitし、公開許可範囲だけpush、mandatory STOP。
