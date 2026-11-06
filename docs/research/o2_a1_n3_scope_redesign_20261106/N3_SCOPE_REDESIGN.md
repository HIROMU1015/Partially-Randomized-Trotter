# A1 N3の範囲見直し：成立確認・予備比較・保証付き本検証

2026-11-06 JST。ユーザーが示した方向は「まずA1対B1-OLの小規模予備検証」。本書はその設計成果であり、具体的な新科学条件の正式採択・実行許可ではない。現行jobを停止・変更・再起動していない。

## 固定対象と変更理由

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/全4電子sector、L_D3、m0/q1/δ=.02/r1/K2、coefficient_atol0、X/Y。
N1 actual finite-grid Q・selected full policy、N2保証ground、認証済みexact Fraction engineを維持する。対象Hamiltonian・target・law・basis・phase/order・ν・controlを変更しない。
N1 744e1a09661ebba6841fe5faf456f88c41c47780、N2 5e9bd08068738cdf1070bdc43bee8194b113ae33、engine fdf00cee3c87d804845497afc89ac6a71f3b3f59。

旧契約は1条件でも4方式×X/Y、train/pilot1536＋cal9216＋prod最大65536=最大76288 trialsを含む。最大値を実際の必要Nや完了費用とは扱わない。
保存されたA1 X train320件の古典性能ではexact engine平均約361秒、価格処理平均約22秒。これは既存の実測値の引用で、新しいvariance/N/価格予測ではない。
初回から全方式の精度保証・全費用比較を一括して要求する設計と、exact engineの実測負荷の組合せが重い。GPU/floatへの置換より先に、各段階で言える結論を限定する。

## 三段階

| 段階 | 問い・取得 | 言えること | 言えないこと |
|---|---|---|---|
| Step 1 | 3-control・5層条件付き学習、rank/range/係数、時間・メモリ | 当該designの識別性guard、固定係数とrange、技術的実行負荷 | 母分散削減、精度達成、CV改善、期待費用優位 |
| Step 2 | A1とB1-OLのみ、独立した少数評価、全正層、価格記録 | 当該pilotの分散・費用の診断傾向と未解決事項 | 厳密ε/α保証、期待費用改善の証明、他条件への一般化 |
| Step 3 | Step 2が有望で、別途採択されたfresh cal/prod | 新しい本検証契約の範囲でのみ保証・比較 | pilotを後から保証付きproductionへ昇格すること |

既存の新B campaignではX/Y各320学習が保存され、rank4/range guardを通っている。SOURCE ee49b53f…の両FROZENを参考資料として保全した。再fitせず、新pilotの学習/評価テーブルへ混ぜない。
旧中断d97eの397件も参照履歴のまま保全し、新しい学習・配分・評価へ入れない。今回の縮小案は既存結果を参照した後の提案であり、完全に結果未閲覧で登録された計画とは称さない。

## 推奨する予備scope

第一候補は**fresh train160＋B1-OL配分probe80＋両方式評価512＝合計752 shots以下**、一条件・2方式×X/Y・一pilot。
新trainは5層×16/軸、probeは5層×8/軸、評価は方式/軸ごと128。全5正層に評価floor4を残し、残りを質量・分散proxy・RZ価格で配分する。
旧学習を参考資料に限定してfresh trainを含めるのは、旧campaignのデータ・係数を暗黙に新学習へ移すことを避けるため。16/層はrank designを反復しつつ旧64/層を1/4へ減らす案であり、係数精度やrange合格確率を保証しない。
学習回数を削ることと3-controlを削除することを混同しない。失敗時に追加trainや再fitをしない。

B0/B1-O、各単位512 cal、Bernstein/Hoeffdingによる十分N計算、8192 cap付き本production、4方式8座標の同時保証を今回のpilotから外す。
旧固定train/pilot/cal10752 shotsに対し752は約93%少なく、旧最大76288に対し約99%少ない**新しいcall予算案**である。実行時間/RZ/科学的検出力の同率改善を意味しない。旧実験で既に支出した費用も削除されない。
旧exact実測から、新752 shotsも時間は概ね時間単位になり得る。完全並列の単純換算でも数十分で、stage依存・末尾待機・共有資源により長くなる。新しい実測benchmarkや完走時刻保証は取得していない。

## 強い対照を保つ

事前判定a53c §6の既存整理を継承する。固定3-controlは5層内で定数であり、層内CV補正を足すだけなら既知質量での合成時に相殺する。
A1の評価はunconditional Q、B1-OLは全5層の直接条件付き取得を維持する。A1をbalanced 5層平均へ変えない。
既存の単純control-pattern集計では(0,0,0)が同じ二つの層を区別できないため、既存n3.production_meanのcell ID検査・exact質量合成を再利用する。
small pilotのfloor/整数配分/係数ノイズ/学習費に由来する差を、理想的な最適層別法に対する一般的優位と呼ばない。

## 今回の境界

追加したのは小さな配分・集計helper、人工test、契約案と読み取り専用の状態証拠だけ。実lawへ新allocatorを適用していない。実分散・実quota・新RZ・新Nは評価していない。
現行jobの正常停止は別の承認事項で、[CURRENT_N3_STATUS.md](CURRENT_N3_STATUS.md)に手順を残す。本設計のmandatory STOPは設計作業の終了であり、現行科学jobの停止ではない。
