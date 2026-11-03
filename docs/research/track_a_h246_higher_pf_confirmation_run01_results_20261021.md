# 高次PF PR独立確認run01：H2/H4再現、H6異常終了

2026-10-21。**H2とH4は独立取得・固定評価・数値照合を完了。H6はnative生成中のSIGBUSでSTOPし、独立確認は未評価である。** 全stageのnative process・workerは回収済み。科学call消費済みH6のretry/resume、追加標本、補助比較、pushは実施せず、local commit後mandatory STOPする。

基準commit `c7c1d8229508253390212588ae2b25e504eb4915`、科学実行SOURCE **`8825391ff73b531be5834bf852b9bf2479d1799b`**。ユーザーは主要3比較と118 trajectory/236 compile軸、取得後の固定analysis scopeを明示承認した。SOURCE・候補・n・入力・shot・既存科学条件は変更していない。[準備契約](track_a_h246_higher_pf_confirmation_readiness_20261021.md)は事前資料として保全し、本報告は事後の実行記録として別に保存した。

## 独立確認の結果

共通条件：linear H-chain、隣接1 Å、STO-3G、tol-only 1e-8の保存DF rank3/10/19、prefix1/3/5、T=.8/P=0、K4、ABS_TIME_PER_OUTER=32。full cosine/sine Hadamard wrapper、symmetric-directional control、scalar once、保存Qiskit1.3.0/rz-sx-x-cx/opt1/seed17。通常signed・経験的allowance倍率1・ε=.0005。固定した保存shotで `G_i=N_cos C_cos,i+N_sin C_sin,i` を会計し、shotやqを選び直していない。

|系|固定S2 / 高次PF|確認n（左/右）|S2平均総RZ|高次PF平均総RZ|独立標本の削減率|探索時削減率|結果|
|---|---|---:|---:|---:|---:|---:|---|
|H2 DF3|S2/q4 / Yoshida4/q2|24 / 28|8.963456984×10¹¹|4.656496765×10¹¹|**48.05%**|47.12%|REPRODUCED|
|H4 DF10|S2/q8 / m5_best/q1|25 / 9|1.133011457×10¹³|5.271006938×10¹²|**53.48%**|53.30%|REPRODUCED|
|H6 DF19|S2/q8 / m5_best/q1|23 / 9予定|未取得|null|未評価|63.12%|NOT_EVALUATED_STOP|

H2/H4では事前固定した改善方向が再現した。H6の探索点改善を独立確認済みとはしない。H6のnullは不利・同点・費用0を意味せず、技術異常による欠測である。比較は固定した2候補間に限り、確認標本で最良候補やqを再選択していない。旧強化決定論的PFや全PF格子への優位性、AD03の正式主比較採用を認定しない。

候補別exact mean/variance・SE・両軸平均/分散/共分散は[4候補の統計表](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_run01/2026-10-21/candidate_statistics_4.csv)、trajectory別単価とresult SHAは[86本の表](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_run01/2026-10-21/trajectory_costs_86.csv)、全stageの来歴は[結果JSON](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_run01/2026-10-21/confirmation_results.json)を正とする。保存探索・pilot・旧確認の標本は一切混合していない。

## 費用差・統計的不確かさ

差の向きは事前固定の `S2 − 高次PF`。cosine/sineは同一trajectory内でpaired、候補間の差SEは別cohort間の独立性を仮定する。両軸を2nの独立標本に数えない。

|系|差の点推定|差の標本SE|主要3比較Bonferroni正規近似区間|近似方向|
|---|---:|---:|---|---|
|H2|4.306960219×10¹¹|1.118897213×10¹⁰|[4.039098486×10¹¹, 4.574821951×10¹¹]|POSITIVE|
|H4|6.059107628×10¹²|1.569328193×10¹¹|[5.683413629×10¹², 6.434801627×10¹²]|POSITIVE|
|H6|null|null|null|未評価|

family α=.05、両側3比較、z≈2.3939798を事前契約どおり使った。H6が欠測でもfamilyを2へ事後変更していない。区間は0を含まないが、正規近似・有限標本・保存bias/normalization/shot条件付きの診断であり、厳密な信頼区間、母平均順位、希少tail保証ではない。費用比の保証区間は作っていない。

H2のS2/Yoshida4で確認された層はS0=24/28、H4のS2/m5はS0=25/9。**完成86標本では全てS0で、S2_SINGLE・S2_MULTI・S4_PRESENTは0観測**である。未観測層の条件付き費用・母分散・tail上界はunknown/nullのまま。有限標本の平均がよく再現しても、未観測希少eventを含む真の期待費用差を保証しない。保存経験的allowanceや数値gateを厳密bias certificateへ格上げしていない。

## 取得・終了監査

|stage|予定trajectory / 軸|受理完了trajectory / 軸|worker|launcher wall|native exit / 全回収|
|---|---:|---:|---:|---:|---|
|H2_CONFIRMATION|52 / 104|52 / 104|5|83.07秒|0 / 全回収|
|H4_CONFIRMATION|34 / 68|34 / 68|14|248.89秒|0 / 全回収|
|H6_CONFIRMATION|32 / 64|0 / 0|25|269.75秒|1 / 全回収|

起動時のfresh CPU/RAM/disk/inode/I/O観測と各stageのprewarm後gateを通過した。H2/H4/H6=5/14/25 workerで同時取得し、固定q×pieceの長い回路を先に投入。共有context、入力SHA付きcache、Git blob cache、process内だけのnested thread抑制を使用した。無関係worktreeの変更やhost PSIを停止条件へ追加せず、共有設定・他user/jobへ変更やsignalを行っていない。

H2/H4の各8 native/normalization/occupation-oracle/Hadamard/compiled gateが全件PASS。compiled vector最大差はH2約1.182×10⁻¹²、H4約4.270×10⁻¹²で、固定CT01の1e-7未満を通過。他gateも固定1e-9未満、phase alignmentなし。H2は元のcompiler scalar guardのgate/角度/wire不変receiptを両軸104件で監査した。

H2/H4は予約call ledgerが固定task全件の期待集合と完全一致し、native exit0、全worker exit0/reap、全結果SHA、archive原byte復元を監査した。H6は完全call集合と正常完了gateを満たさず、STOPのまま保持した。[承認と固定scope](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_run01/2026-10-21/grant_and_scope.json)、[終了・保全](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_run01/2026-10-21/execution_and_preservation_audit.json)に記録した。

新QPY/NPZはlevel1 byte可逆保存。H2は原約51.74 MB→archive約3.09 MB、H4は原約483.89 MB→archive約28.89 MB、全原SHA復元PASS。ledger/metadata等も含むrunの実割当量はH2約49.71 MB、H4約238.51 MB。H6の失敗runは約304.24 MBで、全stageのbyte/native PASSを付けずunknownを保持する。

観測した自分のprocess RSSの合計peakはH2約2.41 GB、H4約7.61 GB、H6約33.17 GB。fork共有pageの二重計数を含み得る5秒観測であり、物理RAMの正確なpeak・上界ではない。scratch raw/partialは定期観測では0だったが、短命bufferを観測できなかっただけで必要scratch0を意味しない。

## H6のSTOPと技術調査

H6はnative_build中、2 workerが**SIGBUS（signal7、元wait status135）**で異常終了した。compile予約は0、受理pairも0。25 trajectoryのevent記録、227 piece law、sampling/piece合成予約各227、native evolution予約25、oracle/low作用予約各23を保存した。予約は非返金call会計であり、作用の完了を保証しない。未着手7 trajectoryも同じ消費済みstageのresume対象にはしない。

Python例外STOPが残らなかった2 taskはS2/q8のsample8と14で、最終progressはnative_buildだった。ただしPIDとtaskの完全対応receiptは保存されていないので、2 signal死との厳密な対応までは認定しない。compile/vector gateに達しておらず、今回のSTOPは旧m5の1e-9/1e-7閾値超過ではない。セッションcgroupのoom/oom_killは0、/dev/shm空きも約270 GBだった。容量不足を原因とする証拠はなく、**SIGBUSの具体的なnative原因は未確定**。元core保存上限は0で、fatal stackログもなかった。

最初のBrokenProcessPool後、残るworkerが終了せず、親のpool管理threadがwait、子23件が待機していた。fork前に設定したlauncher SIGTERM handlerが子へ継承され、SIGTERMを受けてもstop設定だけで戻る終了処理上の問題をコードと人工processで確認した。これは**最初のSIGBUSとは別のcleanup障害**である。

停止済みH6だけについて、既知PID・親PID・starttime・専用configを照合したpidfdで残る23子をSIGKILLし、元2 SIGBUSを含む25 worker全てを元parentにreapさせた。H2/H4や他user/jobへsignalは送っていない。元SOURCE・marker・ledger・STOPを書き換えず、launcherのexit1と失敗終了監査を保存した。

将来の新SOURCE向けに、自分のworkerだけSIGTERMをdefault terminationへ戻し、fatal stackを記録する小さなcomponent `h246_confirmation_worker_diagnostics.py` と人工2件を追加した。**今回の封印済みrunへは未接続であり、SIGBUSそのものの修正完了とは扱わない。** 新しいH6科学診断・retry・resumeは0。[失敗証拠の公開projection](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_run01/2026-10-21/H6_failure_diagnosis.json)。

## 数値照合・保全・公開範囲

固定評価は元SOURCEの中心化Fraction演算・paired共分散恒等式を使った。別の `finalize_h246_confirmation_results.py` は整数raw momentとDecimal80で平均・不偏分散・SE・共分散、固定差・改善率・Bonferroni近似区間を照合した。H2/H4のexact mean/varianceは一致、浮動値も明示した128×binary64 epsilonのroundoff範囲内。正規分位点は同じ宣言済み設計定数を使う。別AIレビューや外部再現を意味しない。

新規の数値照合人工6件＋cleanup人工2件が固定Python3.12.3でPASS。公開表の行数・null・SHA・リンク・保全・機密情報pattern等の最終構造76検査もPASS。事前の31人工/66回帰は準備結果を再利用し、実取得と同時に重い人工回路を繰り返していない。旧名指しcontrol18 fileと全stageのSOURCE/input/runtime封印SHAを最終照合、元raw/grant/ledger/marker/STOPへの書込0。全projects/旧archiveの広域再走査や他worktreeの凍結は行っていない。

科学SOURCE8825391ffは不変。独立照合/projectionのSOURCEはcommit `41ea15bc7`、将来用diagnostics componentのSOURCEは `61488edc2`。資料はこれらの後で保存する。公開資料には集計・無seed trajectory費用・call budget・SHA・失敗経緯を含め、private原本、QPY/NPZ、seed一覧、認証情報、不要な内部絶対pathは含めない。GitHub pushは今回0。

## 次の担当とmandatory STOP

**ユーザー**：H2/H4の有限標本再現結果と、H6未確認を含む現状を確認する。H6の追加native診断・新しい一回科学実行を行う場合は、旧STOPを保持した別SOURCE/診断設計/追加予算に対する新しい明示承認が必要である。今回の消費済みgrantを使ったretry/resumeはしない。

**Codex（別指示後）**：H6のSIGBUSの静的な切分けを続け、必要なら新しいfatal stack付き診断契約を準備する。未接続componentだけで根因解決を宣言しない。新しい実分子sampling/生成/compileはその別承認後に限る。

H4補助比較・追加標本・候補拡張・AD03採択・詳細科学レビュー・論文本文・公開pushへは移行しない。本指示で許された一回取得はH2/H4成功・H6STOPとして終了し、結果・監査のlocal commit後mandatory STOPする。
