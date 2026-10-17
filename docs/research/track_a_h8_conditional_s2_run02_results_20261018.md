# H8 PR s2条件付きpilot run02：16 trajectory・paired32軸の実測完了

2026-10-18 JST。利用者の「計算を開始して」を、worker修正後の新しい一回実行への明示承認として新grantへ結合した。run01の使用済みgrant・lock・STOP・未返却予約は保持した。新run02は正常終了し、追加sampling・自動retry・次stageへ進まずmandatory STOP。

## 固定条件・SOURCE

Linear H8、隣接1.00 Å、STO-3G、保存済みtol-only DF rank28、16 spin modes・sector4900、T=.8、deterministic prefix7、S2、r4、K4。q4はR16・delta=.2、q8はR32・delta=.1。元finite-RTE lawの「少なくとも1個のorder2が存在し、order4がない」s2層へ条件付け、各候補8 trajectoryを新seedで取得した。

- SOURCE：`03479a7546a994bf9d02561d1ab74cec67201b4f`
- 準備REVIEW：`86fdc3494debfc0e825ff86a768cc7090bf012b9`
- 固定science provider：`c31ef424175c02379eafebe8fc6c7548c1d9aabe`
- compiler hash：`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`
- environment/compiler profile：`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`
- plan digest：`38f26accb0af98470cd2935659cd3900f39609192b107ea88051e36e7a23cff3`
- config digest：`3c5d2183d9e113b58aa46820bc037509b4cfbb2f451776ffb657b6fa428dc12c`
- 使用済み新grant digest：`a32afc2623181255ac3eb69e49f8d339375ee411b51770aa8a990caf9ac63c63`

既存venv・provider・元入力・科学条件・compiler optionsは変更していない。元のlaw・規約・人工検査は[準備契約](track_a_h8_conditional_s2_pilot_readiness_20261017.md)、修正は[worker修正報告](track_a_h8_conditional_s2_worker_fix_20261018.md)、失敗run01は[旧STOP報告](track_a_h8_conditional_s2_pilot_STOP_20261018.md)を参照。

## 取得済み結果

費用はstate preparationを除く元native Hadamard whole-wrapperの実compiled指標。cosine/sineは同一seed・同一event列のpaired測定であり、軸ごとに再samplingしていない。以下は保存済みの条件付きn8記述統計で、qやrによる追加の費用倍率を掛けていない。

| 候補 | 条件付き本数 | cosine RZ平均 | sine RZ平均 | 各軸RZの標本SE | paired RZ合計の平均 | 各軸RZ最小〜最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 p7/q4 S2/r4/K4 | 8 | 127,062.125 | 127,062.125 | 952.457994 | 254,124.25 | 122,356〜130,727 |
| B2 p7/q8 S2/r4/K4 | 8 | 255,555.875 | 255,555.875 | 1,317.070172 | 511,111.75 | 249,740〜261,300 |

RZ/CX、各depth、size、全16本・32軸のfingerprint、seed/event digest・order/step seed、各軸wall/RSSを[結果bundle](../../artifacts/resource_applicability/track_a_h8_conditional_s2_run02/2026-10-18/README.md)に保存した。SEは標本SD/sqrt(8)、paired sumはtrajectory内の両軸を足してから算出したもので、独立2標本として扱っていない。

今回の16本はいずれもorder2 eventが1個で、order4は0個。複数order2の可能性は元の条件付きlawに残り、今回の実測がその費用まで網羅したという意味ではない。条件付き分布の実装同値性は固定SOURCEと既存人工検証に基づく。n8の経験分布だけで分布全体の一致を証明したとは扱わない。

先頭trajectoryのcontrol検査は両候補とも元two-branch fallbackを使用し、max errorはq4で3.069874437908981e-14、q8で1.8830724805567155e-14（契約1e-8未満）。4 logical probesずつを課金した。これは有限のnative数値検査で、厳密なbias保証や独立した新signal実験ではない。

## 実行・回収・資源

00:49:21 JSTにwitnessを起動し、01:01:15 JSTに最終gate `PASS_MANDATORY_STOP`。外側native waitのwallは714.527857743秒（約11分55秒）。q4の全8本保存・検証・reap後にq8へ進み、各wave最大8 trajectory worker、各worker内部thread1、cosine/sine間のevolution再利用を使用した。逐次版を同条件で測定していないため、実speedup倍率は未認定。

CPUはown workerだけ[1,2,3,4,5,6,7,12]へ固定。最大単一workerのwait4 peak RSSは1,099,399,168 bytes（約1.024 GiB）。これは同時実行8 processの合計RSSではない。witness/driverのru_maxrssも全processメモリの和とは扱わない。新たなAS/RSS/CPU/wall上限は設けず、OS既存制御は変更していない。fresh gateの2GiB実block割当・inode・実効headroom検査はPASS、numeric quotaはUNKNOWNであり無制限とは主張しない。

全16 worker、driver、witnessはnative exit0・wait/reapを保存し、/proc不在を確認した。first STOPなし、cleanup errorなし。最終poolの`observation_stale`フラグは診断情報として公開した。資源観測の完全な連続性をそれだけで主張せず、native wait4/RSS・回収と区別する。共有環境・他job・他worktree・venvへの変更は0、サブエージェント0。

## Ledger・保全・公開範囲

今回の実消費はtrajectory16、whole-wrapper compile32、occurrence96（外側qの単位）、logical control8。新SCF/DF/state生成、signal、normalization、primitive/reference matvec、決定論compile、fit、費用再会計は0。

旧H8 resource-map lineageに関するcarry・未返却予約込みの会計はcompile1642/上限1722、trajectory803/843、occurrence3195/4219、control376/376。列挙済み過去carryを含めた別の集計はcompile3931・trajectory1915であり、プロジェクト全体の未知のactualまで既知になったとは扱わない。run01の予約compile16/trajectory8/occurrence32/control4と、さらに前の予約を返却・resetしていない。旧run01の51 fileと既知6証拠のbytes/SHA、旧SOURCE checkout不変を確認した。

run02のone-shot試行は1回、条件付きpilotのrun01失敗を含むlaunch来歴は2回。自動retryは0。有効grant・lock・原NPZ・raw trajectory・runtime/checkpoint/cacheを公開せず、軽量の実測・sampling receipt・hash/native証拠・来歴だけを保存した。後処理は保存済み結果の照合と記述統計の再確認で、新たな科学dispatchは0。

## 解釈の限界・次の担当

今回のn8は無条件PR lawのn8ではない。元の確認n32へ加えてn40にせず、元PR平均・bias・normalization・shot・既存費用ランキング・削減率を変更していない。s4費用はnullのまま。条件付きs2平均からPR母平均、厳密な母平均上界、winner優位性、統計的信頼区間を認定しない。有限n8のばらつき、未観測s4、多数order2の費用、費用の裾と選択後効果は未解決。

次の担当：ユーザー・GPTが、この新しいs2費用情報と残る不確かさから次の研究判断を行う。追加sampling/s4 compile/無条件PR費用再会計/他条件拡張は今回の承認外。結果保存・公開試行後、mandatory STOP。
