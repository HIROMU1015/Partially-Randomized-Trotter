# A1 N3一campaign：プロセス消失による中断・mandatory STOP

2026-11-06 JST、SERVER1 `GPUA100-gleap-01`。**N3未完了。精度達成・CV改善は未判定。**
公開照合済み科学SOURCE `03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25` から一campaignを起動した。
継続確認時に親PID2698500と記録された77 worker全てが不在で、実行sessionも取得不能だった。
driver自身のSTOP・例外・finalizer出力は残っていない。**終了原因・終了コードはUNKNOWN**であり、OOM、数値誤差、range違反、利用者の操作など特定の原因へ帰属させない。
科学計算を再起動せず、既存のjournal集計だけを再利用して途中結果を保全し、STOPを追記した。

## 固定条件とSOURCE

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits・全4電子number sector、L_D3、m0/q1/δ=.02/r1/K2、coefficient_atol=0.0、X/Y。
N1 actual finite-grid Q/selected policy、N2保証ground、保存認証のexact Fraction engine、理想gate・独立uniform53-bitモデルに限定する。
元未近似Hamiltonian、実機ノイズ、全RPE、最終総costの結論へ拡張しない。

- 採択契約：`a167effd4a1192d9806ea0bde8e0f7af71a1ef55`、[実行指定](EXECUTION.md)。
- N1：`744e1a09661ebba6841fe5faf456f88c41c47780`、N2：`5e9bd08068738cdf1070bdc43bee8194b113ae33`、engine認証：`fdf00cee3c87d804845497afc89ac6a71f3b3f59`。
- 科学SOURCE：`03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25`、tree `c747ade6cc9863b8790398291458995cb0709656`。実入力前の公開本文/blob/tree照合を保存した。
- ε_X=ε_Y=1/8、同時α=1/20、既存g1/g2/g3、五層×64 conditional train/axis、cal512/cell、prod≤8192/method/axis、J8、D_CV≤4、seed20261106、path64word/trialを維持。旧1/32は不採択。
- b_engineは[SETUP.json](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/SETUP.json)に記載した保存済みの厳密有理数（表示≤7.8079008e-7）をそのまま再利用した。

## 取得結果

| 方式 | 保存train/pilot X / Y | calibration / production | 保存bit分の実現compiled RZ | ε達成 |
|---|---:|---|---:|---|
| B0 | 0 / 0（pilot不要） | 未実行 | 0（未着手） | 未判定 |
| B1-O | 0 / 0 | 未実行 | 0（未着手） | 未判定 |
| B1-OL | 0 / 0 | 未実行 | 0（未着手） | 未判定 |
| A1 | 320 / 77 | 未実行 | 1,935,700 | 未判定 |

A1 Xは各五層64件、合計320件を完了し、rank4、D_CV=`3588805953060867/1125899906842624`（約3.1875≤4）で既定guardを通過した。
凍結係数は`2533274790395915/4503599627370496`、`-5066549580791807/9007199254740992`、`-562949953421315/9007199254740992`。
これは条件付きtrainの一回のOLS結果であり、Q最適係数や改善の保証ではない。
Yは層0が64件、層1が13件の合計77件のみ。Yのfit/rank/range判定は未実行で、係数UNKNOWN。
g2/g3の希少性、control集合、予算は変更していない。

474 trialが実行前journal・path/word固定・worker開始まで進み、397 bitと397価格を保存した。
残る77 trialは最後の記録がexact engine内のselected buildで、bit・価格・未完了の古典費はUNKNOWN。
不足分を補充せず、未着手の層を除外せず、Yの部分データで再fitしていない。
calibration、必要Nの計算・凍結、production、最終X/Y mean・誤差保証は全方式で未取得。
**A1対強いB1-OLを含むCV改善比較はUNDETERMINED**。未着手方式の費用0を比較優位の根拠にしない。

## call・価格・古典費

[CALL_COUNTS](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/CALL_COUNTS.json)はdurableな関数入口記録であり、未完了callの成功を意味しない。

| 記録された処理 | 回数 |
|---|---:|
| campaign起動 / 保存bundle復号 | 1 / 1 |
| exact engine入口 / Born入口 | 474 / 397 |
| selected full build入口（測定・価格合計） / compile入口 | 871 / 206 |
| PCG64による53-bit word（pathとbit合計） | 1,077 |
| 完了OLS fit（X） / Y fit / cal / N計算 / prod | 1 / 0 / 0 / 0 / 0 |
| preparation / CDF再生成 / state solve / spectral / engine再認証 / retry | 全て0 |

397保存bitについてbit先行保存→価格取得、同一measurement/price record、重複trialなし、word範囲/cap、親worker価格を全件照合した。
Xは1,566,902 RZ、Yは368,798 RZ。全て同selected full no-prep、Qiskit1.3.0、(rz,sx,x,cx)、opt2、seed17。
cache hit191件にもfull shot価格を課金し、miss/compile206件を記録した。実機を走らせたgate数ではなく、取得trialに対応するcompiled価格の和である。
未完了77 trialのUNKNOWN価格を0で埋めていない。`realized_RZ_complete`は保存bitに限る価格完備性を指し、campaign完了を指さない。

[CLASSICAL_COSTS](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/CLASSICAL_COSTS.json)に保存済みtrialのstage別時間・generation・compile/cache・setup/fitを記録した。
保存397件のengine worker時間和は147,275.762秒、compile時間和250.059秒。price/cache/build時間には同一path lock待ちも含む。
これらは並列workerの時間和で、wall timeではない。未完了77件の時間、親の最終peak、campaign全体のwall/peakはUNKNOWN。
記録されたworker RSS highwaterの和35,275,980,800 bytesは共有pageを重複計上し、未記録末尾もあるため、実同時物理peakや完了時上界ではない。
期待費用・期待総費用改善の保証はUNKNOWNを維持する。

## 証拠・correctness・保全

- [SOURCE](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/SOURCE.json)、[初回公開照合](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/SOURCE_PUBLICATION_VERIFIED.json)、[PROFILE_LOCK](PROFILE_LOCK.json)。科学SOURCE・固定条件・入力identity・依存環境を追跡できる。
- [ACCOUNTING](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/ACCOUNTING.json)、[中断trial](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/INTERRUPTED_TRIALS.json)、[中断観測](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/INTERRUPTION_OBSERVATION.json)、[STOP](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/STOP.json)。
- [人工11件PASS](TESTS.json)、既存46件の人工/mock証拠を再利用。これはCVの科学的改善の証拠ではない。CIや独立再現は未認定。
- [保存記録の整合確認](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/CORRECTNESS.json)と[保全確認](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/PRESERVATION.json)。SOURCE542、固定input25、gate証明、元main HEAD/status・3087既存ファイルのbytesを照合した。旧SOURCE/結果/STOP/grant/ledger/marker・C1凍結は維持。
- [journals.tar.gz](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/journals.tar.gz)は親journal、77worker journal、内部word tapeの全79ファイルを元bytesのまま保全。[JOURNAL_ARCHIVE](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01/JOURNAL_ARCHIVE.json)に各raw SHA256・行数・archive SHAと全member復号一致を記録。科学SOURCEの変更はなく、検証後に今回のrawコピーだけを圧縮へ置換した。
- [journal専用後処理](reconcile_interrupted.py)は固定driverの`reconcile`だけを再利用し、実state/確率を読まず、科学call0で会計・UNKNOWN・後付けSTOPを保存した。STOPはdriver自身による通常終了記録とは区別する。

archiveの復号は独立directoryで`tar -xzf journals.tar.gz`。word tapeはaudit用で、CV学習・calibration・productionの入力にはしない。

## 次の判断とmandatory STOP

最初の限定campaignは中断し、未完了として保存した。source/profile/seed/controls/精度を変更せず、科学retry・不足shot補充・次方式への救済進行を行っていない。
次に科学実行を考える場合、まずprocess寿命・終了通知・中断保存の技術的原因を切り分け、未完了callとstreamの扱いを具体化する必要がある。
現承認は追加campaignや中断部分の再実行を許可しない。再開または再実行する有限scopeと固定SOURCE・stream/既存データの扱いを示した**別途明示許可**が必要。
今は結果のcommit/push・remote SHA/本文/blob/tree照合まで行い、その時点でmandatory STOPする。新しい科学レビュー・条件採択・N3追加計算へ自動進行しない。
