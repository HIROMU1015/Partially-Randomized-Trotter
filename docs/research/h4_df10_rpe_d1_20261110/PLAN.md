# H4 DF10 D1：固定保存scalar評価の範囲

2026-11-10 JST。ユーザーは指定D1数値評価の一括実行を明示許可した。対象はe91021ae2の保存T=.8・PF 49709717a6c3d405c60d／PR 14bd81a375400c9b0305のみ。targetは保存DF-Hの固定spin sector、化学的精度は既存config.CA=1.59360010199040e−3 Ha、各方式αtotal=.05、P=0。旧SOURCE・結果・STOPと準備15人工テストは保持する。

- priorは保存係数のscalar計算で構成する。Hermitian one-body/GのGershgorin境界、固定4粒子のsigned square境界、保存sector内の固定occupation determinantの変分上界を用いる。係数はbinary64値を正確な有理数にして演算し、表示floatは外向きに丸める。新Fock Hamiltonian／state vector／eigensolveを作らない。保存E0をpriorへ入れない。
- βtotal=.4、βstat=.2を第一設定。h=.8、次に.1を既存principal decoderの `h max(|Emin|,|Emax|)+βtotal<π` で判定する。両者が不適格なら.1のdyadic縮小を代替とする。energy shiftは0。
- 既存horizon関数で最大round indexとround数を求め、全round・二軸一様αを設定する。全時間のN/signalは計算しない。
- 保存referenceとstored numerical state energyのphase/radius整合性、raw/corrected closure、total signed bias＋既存u、reference offset＋T×保存residualを数値pilotのdiskへ接続する。PF B0のdiscard biasを除外しない。ground identificationは未保証で、rigorous boundへ昇格しない。
- ideal reference-defined law/exact-native条件付きのNと、implementation/law距離が未取得のactual lane（N=null）を分ける。未知ground誤差もnull。保存価格はcandidate別cohortの単価を対応確認するだけで、N×Cや全Gを計算しない。
- H6 99ad77ef報告は静的な参考だけ。H6数値、signal生成、trajectory、build/compile、fit、候補追加、RPE simulation、D2/D3は0。

新コードはh4_df10_rpe_d1.pyと明示runner。既存h4_df10_rpe_connection、geometry/Hoeffding/horizonを変更しない。追加人工7件PASS。初回の追加接続testはtoy priceのcohort欠測で失敗し、fixtureを補って通過した。実H4の数値評価前の修正である。

科学実行SOURCEをこのPLANとコードのcommitに固定し、結果へSOURCE・使用input/file hash・処理範囲を記録する。新しいapproval-proof file、grant、ledgerは作らない。小さい保存scalar処理なので単一processを使い、人工CPU/RAM cap・共有環境の変更はしない。結果・報告をGitHub公開し、remote SHA/tree/本文照合後mandatory STOP。
