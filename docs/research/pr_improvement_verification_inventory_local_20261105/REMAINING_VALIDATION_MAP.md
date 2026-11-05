# 次の研究で新たに必要になる検証の地図

2026-11-05 JST。以下は検証の不足と再利用可能な証拠の対応であり、実験実行・研究方向の採択・優先順位付けではない。
全項目を実施すべきbacklogとも扱わない。[Gap](IMPROVEMENT_GAP_MATRIX.md)と[台帳](IMPROVEMENT_VERIFICATION_LEDGER.csv)の固定条件・原STOPを前提に読む。

## 既知と新規の境界

| 将来、主張したい内容 | 既に利用できる証拠 | その主張に向けて不足するもの | 作業の種類と担当（提案段階） |
|---|---|---|---|
| 辞書選択が同精度総費用を削減する | 同じ合成HでのF1/F2全登録correctness、F2の1-shot実compiled期待費用、保存law/normalization | 共通targetへのgamma・統計資格、認証N、N加重費用、選択則の未使用入力性能、取得費用 | まずユーザーが主張scopeを選ぶ。必要ならCodexが既存資格・SOURCEを静的照合。新N計算・保存再会計・新fixture実行は別の明示許可後 |
| Gaussian singletonの選択則が役立つ | PS48の同operator/law検証、局所改善と悪化の保存例、強い二recipe full-wrapper対照の既存整理 | 未使用contextの予測・regret・古典取得費と全law期待値 | 独立方向は一時終了済み。ユーザーが明示的に再開を決めた場合だけ対象を定める。今回selector設計・実装・全F2展開は行わない |
| 非prefix/groupingが既知選択を超える | PR-2/M1/M2/PM、P-C/P-D、B-F/B-M/R3、同情報対照の反証材料 | 具体的な独立生成差、同情報・同候補domain・同探索費の比較、PF/normalization/実装境界の接続 | 科学的な差の採否はユーザー。重要な詳細レビューを行うなら独立した開始許可が必要。新split生成・再ランキングは別の科学実行 |
| 新しい高次PF/BCH方式に独自の利益がある | 既知PF係数・有限時間校正・response、compact BCH同値、既存高次PR/長時間の正負結果 | 既知係数・compact再帰と異なる操作、同target精度・同強対照・全実装費の差 | 既知familyの重複確認はCodexの静的調査候補。係数探索・bias再評価・compileは未許可。次数だけで採否を決めない |
| RA/新samplingが分子全taskで有利 | G6形式構成、G7〜G10限定native/十分shot会計、finite-bit/live API、C1分子logical完成 | 物理DF binding、共通認証N、strict nativeの全support、deterministic halves/full-wrapper、生成・前処理費込み比較 | C1追加研究は一時凍結。再開の科学的必要性をユーザーが判断した場合だけ、必要な接続と実行scopeを別途指定。G6構成そのものを再実装しない |
| allocation・合成配置の局所利益が全wrapperへ移る | BS/SP、R1、D1〜D3、G9/G10、訂正Parent S4と強CTS対照 | 同じlaw・提案分布・全trial・準備/readout・誤差予算の下でのwhole-wrapper移送 | 保存値を引用するだけなら追加科学不要。新native取得・再配分計算・再会計はユーザーの許可後。旧toy探索を自動延長しない |
| 表現変更が最終energy資源を下げる | Pauli/DF既存比較、研究Cの候補別強D/下流・独立価格・固定word検証、旧norm単独最適化の反例 | 元Hの同task binding、上流model/DF誤差、全law実装差、候補・状態・取得費を合わせた比較 | 既存baseline/SOURCEの照合を再利用。現在のnominal支持を厳密CAや一般因果へ格上げしない。新最適化/CA評価は別科学実行 |
| roundごとの表現/split変更で同energyを安く推定する | fixed-splitのround別r/K・配分、V1〜R1、条件付きenergy接続、M1-Sの局所診断 | 異構成signalの共通energy/branch/state接続、fixed-split強対照にない節約と全費用 | 新しい数学的interfaceや方式の採否は未決定。今回設計・証明しない。新保証レビューと科学実行はそれぞれ別の明示許可 |
| graphに応じたPR表現/splitが有利 | topology-free builder、basis共有、control、Pauli/DF/P5の既存実装 | 同graph・layout/routing予算・precision・controlの比較、既知routing後にも残る差 | 将来対象graphと問いをユーザーが選んだ場合のみ。今回routing/compileを実行しない。単なるbackend接続と算法差を区別 |
| 現在の限定結果を正確に説明する | 固定報告・台帳・既存coverage/原status・本Gap | 新性能計算は不要。文言に必要な範囲のmetadata同定不足だけがあり得る | 今回の5成果物で棚卸しを完了。論文執筆・詳細レビューは今回に含まれず、Codexの自動継続なし |

表は主張ごとの条件分岐であり、ここで新しい比較契約・入力grid・新アルゴリズムを固定していない。
保存結果の再集計・比・SE・fit・順位・十分Nの算出も、新しい科学的数値解析に含める。

## 証拠不足と実行不足を分ける

| 状態 | 具体例 | 今回の結論と扱い |
|---|---|---|
| 未検証／未実行 | RQ-07の改善比較、round別表現/split新方式、未資格N比較 | 科学的成功・失敗のいずれでもない。提案だけで実行しない |
| 実行したが未完了 | 旧F1/F2、旧G9/G10、旧C1-5AのP5 | 新runの完了とは別のattempt履歴として保持。旧outputの補完やresumeをしない |
| 証拠未発見・同定不足 | 旧UWC原runとの対応、旧DF修正入力、一部SOURCE/cohort | 未実施の証明でない。必要なら対象限定のread-only回収を別に指示できる。代替再計算はしない |
| 技術的失敗 | metadata setter、compiled位相、Fraction/API、RSS契約、JSON key、P5 worker | 性能negativeではない。後続修正版の固定SOURCEと別実行を辿る。原因未解決は未解決のまま |
| 限定的な負結果 | P-A非退化対照、P-C広域追跡、P-D強baseline、PR-3、PS48一律適用、generic whole費用 | 既に得られた科学的・技術的情報。研究全般の不能定理にせず、同じ問いの単純な二重実行を避ける |
| 既存法との同値／独立差未確立 | B-M、Phase2C-R1、Gaussian積、cost-aware/相関の一部、R3 | 性能実験未実施でも検討済み。名称変更やcache禁止で独立差を作らない |
| 保留・正式一時終了 | C1一時凍結、Gaussian singleton独立方向の一時終了、他の旧STOP | 未解決欄があるだけで再開しない。次段階・別namespaceでSTOPを迂回しない |

## 今回残るUNKNOWN

1. **改善専用のSERVER 1/2追加報告**は未取得。取得済みの11/5資源差分は別scopeである。LOCALの公開証拠整理は暫定で、3環境の改善研究全量を統合したと認定しない。
2. **旧70索引の実行SOURCE単独SHA**は全行を今回再同定していない。実装参照・結果carrier・文書commitを保持し、独立に確定できない欄をUNKNOWNにした。SOURCEが存在しないという判断ではない。
3. **旧UWC・失効DF**の原input/raw/修正版identity。後続研究Cや別候補を旧runの再現成功に代入しない。
4. **独立性・raw全量**。自己検査、二精度参照、同backend/seed取得、別host保存、独立acceptance、独立科学確認は異なる。private seed/raw/QPY全量は今回監査していない。
5. **未保証の科学範囲**。全law・母平均・一般分子・接続性・最終費用・実backend等は、各固定claimが必要とする場合にだけ追加判断する。全部を現結果の必須不足にしない。
6. **後続公開の範囲**。11/5の選択した固定追補まで確認したが、全稼働branch・全新結果の完全censusではない。今回の文書公開後の状態を推測しない。

## 停止点と次の担当

Codexはこの台帳・Gap・再利用地図・残検証地図・報告の作成、文書検査、専用ブランチ公開とremote照合で停止する。
追加の科学的検証、設計、実装、重要レビュー、論文執筆は開始しない。

次の担当はユーザー。必要であれば、未取得の改善専用報告の提供／取得対象、または新研究で判断したい問いを別途指定する。
これは今回の終了に必要な承認依頼ではない。今回依頼された棚卸しの後に自動実行する作業はない。
