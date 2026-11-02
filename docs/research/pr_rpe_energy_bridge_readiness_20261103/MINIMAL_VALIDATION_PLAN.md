# 最小限の追加検証候補

2026-11-03 JST。LOCALによる技術調査後の提案。**実行契約・parameter freeze・科学レビュー・実行承認ではない。** 数値再解析、sampling、simulation、RNG replay、compileは今回実施していない。既存STOP、局所compiler精密化停止、有限coherent-signal研究の採択範囲を維持する。

共通の固定参照・SOURCE hashは [EVIDENCE_INDEX.json](EVIDENCE_INDEX.json)、論点IDは [ASSUMPTION_EVIDENCE_MATRIX.csv](ASSUMPTION_EVIDENCE_MATRIX.csv)、保存roundへの対応は [ROUND_EVIDENCE_MAP.csv](ROUND_EVIDENCE_MAP.csv) にある。以下の優先順位は最終energy主張への影響による提案であり、研究方針の正式採択ではない。

## V1：共通E₀・状態・半径の接続（最優先）

- **命題**：同じ元DF HamiltonianのE₀を対象に、PartialとFullの各roundが、必要なPF biasと信号半径の条件を満たす。exact H ground state、PF有効固有状態、有限RTE観測信号を区別する。
- **既存直接証拠**：RPE4の固定4round、SCHEDULEJSONの56点、ABLGRIDの108点、ABLJSONのFull18点半径。選択opt2 Partial18点の全q/r/Kは108点gridに含まれる。PF3/PF12には同snapshotのD6と支配固有位相の短q比較がある。
- **欠ける接続**：モデルの「exact effective PF eigenstate・単位半径」と、物理H基底状態上の保存信号との論理関係。近いoverlapや近い半径は厳密な単位下界ではない。Fullの保存半径だけでは同δ・長qのPF対exact位相誤差を確定できない。経験的D6係数は厳密energy保証へ変わらない。
- **既存資料だけでできること**：今回完了した固定metadata照合、SOURCEの定義・符号・scalar・q反復の整理、旧56点と後続108点の用途分離。未記録の推移的依存SOURCEや準備hashの由来に既存記録があれば、read-only照合で不明を減らせる。欠ける記録を推定して埋めない。
- **新解析／新計算**：保存複素信号から共通E₀に対する位相残差を再算出すること、最適化後の予算に合わせて半径・shot・費用を再評価することは**新しい科学的再解析**で、別承認が必要。未保存の状態／演算子対応や必要な物理信号を生成する場合は新科学計算となる。新定理が必要なら、Codexで採択せず、ユーザーが別途承認するGPT科学レビュー候補として分離する。
- **判断への影響**：接続できれば既存18roundモデルを共通energyへ解釈する根拠が強まる。失敗すれば入力状態・半径・biasの条件付きモデルとして残し、最終energy実証への昇格を見送る。有限coherent-signalの既存結果を取り消す判断とは別である。
- **元SOURCE／境界**：METHOD、PLAN、FINITESRC、ACCOUNT、PFDELTASRC、ABLATIONSRC、PF3/PF12、RPE4JSON・ABLGRIDのdirty provenance。既存contractのSOURCE/input/STOPを変更しない。
- **次担当／承認**：ユーザーが追究する主張範囲を選ぶ。Codexは別指示があれば残存metadata照合を行う。科学再解析・新科学計算・詳細レビューはそれぞれ該当する明示承認後だけ行う。

## V2：同じ最終配分に対するbranchと全round成功率

- **命題**：V1で定義を揃えた信号・誤差・半径と、保存opt2のN/α/βが、同じalias-free branch上で最終energy精度・成功率へ接続する。
- **既存直接証拠**：RPE4は解析信号、厳密二項確率、明示fresh-IID1572 trajectory、10万回の周辺分布branch検証を完了。WP04には**長18roundの旧配分における軸別厳密二項・union-bound診断もある**。未実施に戻さない。
- **欠ける接続**：最終coherent opt2の別N/α/β・scheduleに対するbranch／最終energy成功の保存結果は未発見。4round成功率、旧WP04確率、51task価格監査を最終モデルの成功率へ代入できない。外部alias-free windowとfresh-IID条件の適用が必要。
- **既存コード再利用**：RPE4の逐次nearest-branch処理、finite信号validator、WP04の統計診断、ACCOUNTの厳密branch条件とunion-bound構造を再利用候補とする。scientific importを伴う既存テストは今回実行しない。
- **新解析／新計算**：保存signalと最終N/αの組で確率やbranchを新評価するだけでも新科学的再解析。新しい測定sampling、長round trajectory監査は新科学計算。既存108点・56点の全信号再生成や51task再compileは自動的に必要としない。V1の結果を先に使い、保存証拠で埋まらない接続だけを対象にする。
- **判断への影響**：成功なら固定小系の同energy taskへの経験的end-to-end解釈が進む。失敗なら費用モデルが成功するenergy手順を表すとは言えない。経験的PF入力のままで数学的certificationと呼ばない。
- **元SOURCE／境界**：RPE4SRC、HORIZONSRC、SCHEDULESRC、ACCOUNT、ABLATIONSRC、RPE4JSON、ABLJSON、OPT2JSON。CA/10は当時の暫定入力で、新しい正式目標は固定しない。
- **次担当／承認**：ユーザーの別承認後にCodexが限定再解析／検証を行う候補。新parameter grid・反復数・seed・実行契約は本書で指定しない。

## V3：全費用比較を述べる場合だけ、価格移送と不確かさを確認

- **命題**：共通energy taskを満たすPartial／Fullの総費用比較に、長q移送・較正誤差・状態準備を含めても解釈可能な不確かさがある。
- **既存直接証拠**：M06-Fの51/51監査は完了。全rのcoherent opt2較正、q8 holdout、A0の限定δ.02/r32/K2 selected-RZ q16,32照合がある。初期precision不通過をfresh32で回復した記録も保持する。
- **欠ける接続**：q>32へは短q affineモデルを移送している。A0は全metric／全セルのq32保証ではない。最終scheduleのr1/q16,32には、このr32価格を直接対応付けられない。保存5%・25%は全長qの信頼上界ではない。状態準備は未測定で、shot数が違うため相殺しない。
- **既存資料だけでできること**：直接価格のtuple/domain、較正・holdoutの分離、SOURCE hash、選択済み費用と区間の転記は今回完了。保存約4.858%点差・全区間重複を維持する。
- **新解析／新計算**：既存価格を用いる新不確かさ伝播・費用再ランキングは新科学的再解析で別承認が必要。必要に応じた追加holdoutはcompile承認が必要。ただし全q直接compileを既定要件にしない。既存局所compiler精密化STOPを回避せず、まずV1/V2でenergyへの接続可能性を確認し、比較主張に不可欠な移送問題だけを別提案する。
- **判断への影響**：点差と区間が同時に支持する比較へ進めるかを決める。現存区間は既にP=0で重複するので、現段階の頑健な優位性は主張しない。
- **元SOURCE／境界**：OPT2SRC、OPT2EXTSRC、DECISIONSRC、LINEAGESRC、OPT2DOCのResearch routing、AUDIT、A0、INITPROV/EXTPROV。全metric保証やDF10/19・opt1への移送を新たな当然要件にしない。
- **次担当／承認**：ユーザーが最終energy資源比較を追究すると決めた場合、Codexが最小の未解決移送問題を提案する。再解析／compile／既存停止条件の変更は該当する別承認後だけ。

## 完了済みの技術照合とSTOP

共通energyの定義・符号、固定証拠の所在、選択36行、Partial18 tupleと保存108点の対応、Full18半径、既存長round統計の所在、q≤32価格の限定domain、coherent/mixedの区別、報告済みSOURCE hash照合は今回のread-only技術調査で整理した。これらの棚卸しを理由に同じ科学計算を再提案しない。

**最も判断を変え得る次候補はV1の共通E₀・入力状態・半径の接続である。** V2/V3の大きな実行を先に固定しない。新たな中心claimは未採択、科学実行・詳細レビューは未承認のまま、公開照合後mandatory STOP。
