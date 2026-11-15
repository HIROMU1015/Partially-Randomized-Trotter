# 決定論PF m1/m2の有限直接価格取得

利用者が承認したFull PF / B0I p7 × S2 / 標準Yoshida4 / M5 × m1/m2 × X/Y、24軸の不足分だけを取得する。m1/m2は同じ指示の範囲であり、途中の再承認は不要。新規compile上限24、sampling・科学retry・価格予測・モデル判定・fit・MSE/N/G=0。開始時の同条件direct coverageは72/156軸。旧m0・Stage1/2・m9・C凍結モデル・SOURCE・raw・STOPは保全する。

| round | formula | T | q | actual幅 |
|---|---|---:|---:|---:|
| m1 | S2 / Y4 / M5 | .2 | 1 | .2 |
| m2 | S2 | .4 | 2 | .2 |
| m2 | Y4 / M5 | .4 | 1 | .4 |

C alignment `0b4b2d941cd669aaa45231e1236bc2d3158a9d12:docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json` の6固定configurationを元Full/p7 plannerで再構成し、`configuration_id / physical_id / m / axis`と符号付きstructureを一意に照合する。S2の同じactual幅でもT / q / scalar位相は別cell。m0と同様、保存linear H4 / 1.00 Å / STO-3G / tol-only DF10 / 元H-state-snapshot、P0除外。Fullは一体項＋fragment0..9、p7は一体項＋fragment0..6を保存順に保持する。Full c_H、p7 c_H＋tail identityをtotal Tへ一回適用する。

## 重複照合

m0公開 `c95353a44b4ce99ed86e949e408d522d6ad7e7fe` と全round coverage `bcb67b2b2bb78776db1dd8d0625a3facb37cd298` を引き継ぐ。5 carrier（746 record）とFull catalog（136候補）を旧ID・別schedule名にかかわらずsystem/method/prefix/formula/T/q/actual幅/R/Kで照合した。Stage1/2・m9・m0・今回namespaceの公開/private RESULT、対象IDの既知private trial path、自ユーザーの関連running jobも差分確認する。旧PF commonのT=.8、旧direct表の他rank/prefixなどの除外根拠は既存監査を継承し、価格の再計数・再fitをしない。

今回の初回照合は**24軸とも不足、受入可能な保存候補0、未解決候補0、関連running job0**。[DUPLICATE_PREFLIGHT.json](DUPLICATE_PREFLIGHT.json)。関連remote3枝はm0完了時から変更なしを確認した。未列挙archive・別サーバーまで未実施と断定しない。実行入口で照合を更新し、重複候補は自動compileしない。

## 最小拡張と実行

既存 `scripts/run_pf_m0_prices.py` に `(1,2)` の有限selectorと別namespaceを追加した。既定 `(0,)` のID / 物理条件 / 出力は保ち、過去の実行SOURCE729fe5370と公開m0 rawを保全する。元m0 readerの固定checkout再現と、新しいcaller SOURCEを区別する。native / wrapper / compiler、元Full/p7 trial・7作用gate、fast context、process監視・段階計測・C guard経路は再利用する。新しい科学pipelineや候補探索・N評価は実装しない。

Qiskit1.3.0 / rz,sx,x,cx / opt1 / seed17、symmetric_directional、測定付きfull controlled Hadamard wrapper。native policy `H46_LONG_TIME_NATIVE_COMPOSE`、RTE `NOT_APPLICABLE_NO_RTE`。contextのscalar / prep hash / fragment順を保存Stage2の固定recordに完全一致で照合し、新しい空cacheでX/Y各軸のcompiled-vector gateまで取得する。元native <1e-9、compiled <1e-7、位相合わせなしを緩和しない。

人工testは新3件＋m0回帰4件PASS。24軸・scope違反・T/q/actual幅/axis/structure誤結合の拒否、同幅S2 m1/m2の別fingerprintと元7作用gate、段階計測、worker exit/signal、異常時の部分保存とretryなしを確認した。m1/m2用testの初回で、悪いstructureがValueErrorではなくTypeErrorで拒否される既存挙動を確認し、testの例外型だけを合わせた。科学guardの変更・緩和はない。

```bash
python -B tests/check_pf_m1_m2_prices.py
python -B tests/check_pf_m0_prices.py
python -B scripts/run_pf_m0_prices.py --short-rounds --metadata-output <new-file.json>
# 今回のcaller/tests/metadataをcommitで固定した後、一回のみ：
python -B scripts/run_pf_m0_prices.py --execute-approved-m1-m2
# raw公開commitへsidecarを束縛：
python -B scripts/run_pf_m0_prices.py --short-rounds --bind-route FULL_PF --publication <raw-commit>
python -B scripts/run_pf_m0_prices.py --short-rounds --bind-route TRUNCATED_PF --publication <raw-commit>
```

出力はFull / p7の既存studyにそれぞれ新設する `m1_m2_direct/run01`、privateも同suffix。総括はこのstudyの `run01`。12物理cellは利用可能CPU/RAM/diskと対象数に応じてprocess並列化する。routeごとのcontextをforkで共有し、各cell内でevolutionをX/Yに共有する。高速化は元trial内部で有効にし、両軸compileを省略しない。協調的stage log・親processの終了code/signal観測を使い、非同期stack読取・共有設定変更・他process操作をしない。

作用gate / C guardを満たさない軸はUNKNOWN、部分raw / cache / logを保全する。再compile、片軸補完、seed・round変更、gate緩和をしない。公開表はm0のAXIS_PRICES形式を再利用し、round別countsとcoverageの実増分を付ける。全24軸有効なら96/156、6候補のm0～m6に直接証拠が揃うが、旧m3のconsumer binding等の受入状態は別課題として残す。

結果・HANDOFFのcommit/push・remote SHA/tree/主要成果物照合後mandatory STOP。m6/m9の2%規則・モデル資格拡張・他round・MSE/N/G/winnerへ進まない。ロードマップ本体は編集せず、差分を統合担当へ渡す。
