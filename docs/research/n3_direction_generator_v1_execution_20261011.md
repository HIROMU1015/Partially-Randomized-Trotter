# N3保証生成・限定移送：実行sourceの固定

利用者の新しい実行指示により、準備commit `ba4530678df9d665d2e6005fb08179b0f7a7921f` から専用branchを作った。準備契約と旧STOP記録のbyteは保持する。今回はN3の一回batchのみをこのbranchで扱い、N1・N2・旧4例・GPT付録は実行しない。

数学条件・12入力・比較法・上限は [準備契約](n3_generator_transfer_contract_v1_20261011.md) のまま。μ0=(κU−δD)/ν が許容する最小μなので、これでPSDにならないκには同じκの大きいμでも証拠はない。ただし数値golden-sectionと有限dyadic候補は完全性を保証しない。未取得証拠は UNRESOLVED_NOT_CERTIFIED とする。μ0のaffine余裕は厳密にzeroで、正の余裕とは呼ばない。

producerはSchur更新、独立保存checkerは元matrixからのLDL再帰とL D Lᵀの恒等式を使う。負pivotとzero pivotの非zero残りrowを拒否し、toleranceは使わない。すべての有理中間値を4096bit上限で監視する。元のκ・μ三条件も復元する。

trialは全determinantと接続したequal pair、符号・1/√2 normalizationを保存。r、旧五点、外向き候補6点の検証済み下界からellを得る。下界生成は入力ごと一回だけで、その費用は共通U/cの比較すべてに共通の会計として記録。旧scalarは元U/c定義のまま別fieldへ保持。旧c五点は副対照で、primaryの五点/直接生成との差が下界改善か探索改善かを分ける。

CPU32、GPU deviceなし。2 spawn worker、各BLAS/OMP/Numba/Rayon1 thread、AS4GiB、aggregate8GiB、wall/CPU capなし。sector operatorをbaseごとにcacheし、one-body tensor contractionでdiagnostic matrixを構成する。synthetic occupation-list参照で一致を確認。速度倍率は比較していないため主張しない。新synthetic tests15件PASS。初回fixtureのδ境界指定ミスと修正はtest_historyに保持。12移送入力の評価はsource固定後の一回のみ。

全12×6証拠ledgerを保存してhash sealを作ってから、full/active groundの数値診断24 solvesを行う。数値許容1e−10は証明ではない。停止時も一次記録を保存し、科学retry/resumeは0。結果取得後も中心テーマnull、次段認可false、mandatory STOPを維持する。

[module](../../src/trottertracks/representation_exploration/n3_direction_generator_v1.py)、[runner](../../scripts/run_n3_direction_generator_v1.py)、[tests](../../tests/test_n3_direction_generator_v1.py)、[独立checker](../../scripts/verify_n3_direction_generator_v1.py)、[実行前資料](../../artifacts/n3_direction_generator_v1/2026-10-11/preexecution/)を参照。
