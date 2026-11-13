# Full PF追加幅探索の判断と有限案

追加探索には価値がある。S2/Y4は既存の局所誤差予算境界を補完でき、M5は既存上端Δ=.8でも小さいshiftを保ったため、より粗い幅を調べる動機がある。ただし、今回新しいH4固有値・保存数値再解析・価格/N/G計算は行っていない。以下はユーザー承認前の有限案であり、最適幅や全RPE精度の認定ではない。

[科学実行案](SCIENCE_PLAN.md)／[C・B HANDOFF](HANDOFF.md)／[exact配置・SOURCE・ID](PLAN.json)／[全13roundの表](ROUND_SCHEDULE.csv)／[人工テスト](TESTS.txt)。

## 判断の根拠

既存結果の正本はcommit `3ae055fa7a696b13320e00ec79a5ba15c4ed182e`、`docs/research/h4_full_pf_heff_scan_results_20261114/`。保存Linear H4/1Å/STO-3G/tol-only DF10、sector(2,2)/36次元、Full PF/prefix10、同H/state/snapshot/scalar、h=.1/M12/P0である。既存の数値を引用すると、S2/.2はδE=+5.35412051879e−4 Ha、Y4/.4は−1.90607716779e−4、M5/.8は−7.28958845286e−7。S2/.4、Y4/.8は元の局所εtrot=.00079680005099520 Haを超えた。これは既存spectral screenの結果であり、今回境界をfit・補間・外挿していない。

H_eff固有値シフトは採用済みの決定論PF誤差評価の主経路である。個々のshiftが全RPE biasやRMSE上界となる資格は引き続き未取得。共通step、固有状態対応、全spectral成分と保存state支持、branch、early例外を保持して比較する。

## 推奨する初回5配置

幅は表示小数ではなく、保存exact hの有理数倍を正本とする。

| 公式 | 要求幅（正本） | 表示近似 | m12 qPF | m12 signed S2 piece数 | 同公式の既存最粗候補からのlate q減少 |
|---|---|---:|---:|---:|---:|
| S2 | 16h/7 | .2285714286 | 1792 | 1792 | 12.5%（旧.2） |
| S2 | 8h/3 | .2666666667 | 1536 | 1536 | 25%（旧.2） |
| 標準Y4 | 16h/3 | .5333333333 | 768 | 2304 | 25%（旧.4） |
| 標準Y4 | 32h/5 | .64 | 640 | 1920 | 37.5%（旧.4） |
| M5 best | 16h | 1.6 | 256 | 2816 | 50%（旧.8） |
| M5 best・保留 | 32h | 3.2 | 128 | 1408 | 75%（旧.8） |

これは式・整数scheduleから得る構造比較で、compiled価格やGの予測ではない。S2/Y4の2点ずつは境界近傍の比較と有意なq削減を両立させ、各公式で新actual幅を2種類に抑える。既存結果だけでは、いずれが予算内に残るかは分からない。2点とも同じ判定でも自動二分探索・追加幅へ進まない。

M5/16hは旧M5/.8に対してpiece数を半減できる一方、m12では旧S2/.2の2048より多い。M5/32hなら1408となり、公式間の構造順位を変える可能性があるため、候補として保持する。ただし分枝準備が未完のため初回科学batchに含めない。N・compilerの融合・角度依存・wrapper費用によってcompiled総費用順位は変わり得る。

S2/Y4/M5は既存1/3/11個のsigned S2 pieceを使用する。同一公式でqを減らしても、全回路のRZがq比例とは仮定しない。単一step・m12の価格だけで最終選択しない。

## 全roundのactual幅と再利用

既存B/Dのcanonical/physical APIで6配置×13round=78行を生成した。初回5配置の新actual幅は5種類で、保留M5/32hがさらに1種類。全6配置の15 actual cellのうち9は既存SOURCE参照である。同じexact actual幅について、backendへ渡すsigned timeのbinary64表現も一致することをmetadata検査した。

- S2/16h/7：m0=h、m1–3=2h、m≥4=16h/7。
- S2/8h/3：m0=h、m1–2=2h、m≥3=8h/3。
- Y4/16h/3：m0=h、m1=2h、m2–3=4h、m≥4=16h/3。
- Y4/32h/5：m0=h、m1=2h、m2–3=4h、m4=16h/3、m≥5=32h/5。
- M5/16h：m0–3=h,2h,4h,8h、m≥4=16h。
- 保留M5/32h：m0–3は同じ、m4=16h、m≥5=32h。

要求幅をactual幅の代わりに使わない。例えばexact 32h/5を表示float .64に置換すると、m5のceil計算でq=5から6へ変わる。C側もexact分数・物理signed time・全scheduleを参照する必要がある。

## M5の分枝

既存評価器はPRINCIPALと、幅が2π/Δより小さい単一ENERGY_WINDOWを扱える。M5/16hには**[-2.5,1.0] Ha、purpose=EFFECTIVE_SPECTRUM_WINDOW**を新たな承認候補として提案する。元H全固有系の保存端点（commit69ee0ff8、EIGENSYSTEM.json/reference_comparison/eigenvalues_haの0番と35番）は−2.1663874486347643、+0.7509635882732985 Haであり、その全帯域を外側から囲む単純な窓を選んだ。保存solver energy・priorで分枝を合わせる方法ではない。

この窓はΔ=16hの幅条件を満たすが、実PF固有系の対応・cut距離・再構成は未評価。次回は旧M5/.8の保存全スペクトルとの窓整合検査1件も明示的に承認範囲へ含める。窓を選んでもHamiltonian/scalar/energy_shift=0/priorは変えない。

Δ=32hでは同じ窓が幅条件を満たさず、単にprincipal logの最低値を比較して物理FAILとすることもできない。mode/groupにSOURCE付きで対応するcontinuation等の分枝規約が別途必要である。現APIの制約を物理的な不適格性と呼ばず、保留する。既存`pf_delta_validation`のtarget中心Schur位相方式はreferenceを分枝に使うため、この契約へそのまま移植しない。

## 実装・価格coverage・資格

[metadata CLI](../../../scripts/prepare_h4_full_pf_followup.py)は既存`prepare_h4_full_pf_heff_scan`、C86f37d25/D e7dca5f7/B437ba4acのcanonical生成・照合を再利用する。PF式・H_eff評価器・科学runnerは作り直していない。既存scannerはPRINCIPAL固定/旧15cell用なので、次回はbranchと新scopeを渡す最小caller差分が必要である。

価格はD e7dca5f7の正本catalogへのphysical一致とXY statusを転記しただけで、新受入・fit・外挿をしていない。初回5配置のm12は全て照合域外で、早期の一部だけEXACT_METADATA_MATCH。Cの最新公開context367dc98dでも今回必要な新Full PF長q価格は確定していない。exact recipe・axis・profileを渡し、C側のdomain判定が必要。

[新4人工/metadataテスト](../../../tests/check_h4_full_pf_followup_plan.py)はPASS。全round exact identity、初期幅、表示丸め、科学import/leafの非起動、人工2×2のwindow・reference非依存・不適格窓を確認した。実科学性能の証拠ではない。新H4計算/保存数値再解析/実価格/N/MC/Gは全て0、十分N/global RMSE/G/winnerはnull。

次は**ユーザー**が[初回5cellの科学範囲とM5窓](SCIENCE_PLAN.md)を承認し、承認後に**Codex A**が固定callerで実行する。Cの価格取得は別範囲。今回の公開・remote照合後mandatory STOP。
