# Track D B4-2：元H向け13-round RPE十分shot数

2026-11-06 JST。**ideal finite-RTE IID・exact nativeモデルで全13 roundの十分条件が成立し、X/Y合計のHoeffding十分shot数は14,738。元H energy誤差幅は0.0012783173232380342 Haで目標0.00159 Ha以内、全26軸のfailure配分和は0.01。** 保存B1の数値operator誤差＋engineering guardと元H厳密ground仮定に基づく条件付き結果であり、actual実装保証や実測成功率ではない。1回の実行を完了し、mandatory STOP。[固定PLAN](PLAN.md)

## 1. 固定条件とSOURCE対応

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。初期状態は**元Hamiltonian Hの厳密基底状態**を仮定し、準備費を除外する。保存own-DFの信号・Nを元Hの結果として採用していない。

| 今回正式に採用した評価条件 | 値 |
|---|---|
| energy精度 εE | .00159 Ha |
| failure上限 αtotal | .01 |
| βtotal／βstat | π/6／π/12 rad |
| αaxis | .01/26、全13 round・X/Y共通 |
| schedule | m0..12、q=2^m=1..4096、t=.1q |
| energy prior | [−10,10] Ha |
| モデル | ideal finite-RTE IID、exact native、fresh state／trajectory per shot／occurrence |
| DF energy reserve／Weyl再加算 | **なし**。B1の元H operator誤差に一回含まれる |

| 役割 | 固定identityと確認 |
|---|---|
| B4-1実装基点 | `31cd67c91abda3c9c5d4b643c434c040b8ceaeca`。元Hadapter・accuracy・Hoeffding・horizon・decoderの関連SOURCEを同一bytesで再利用 |
| B4-2実行SOURCE（local） | `8e472b5001278e51b9919a8302d0a4bd3fcf9209`。新runner、PLAN、人工test7件PASSをcommit後に一回実行。科学SOURCE／入力を前後照合 |
| [B4-2公開SOURCE][PUBSOURCE] | `6da5afba79895c6b3c092f41e332d444dedc8ed3`。GitHub接続で保存した同一SOURCE tree `767d68f537d0c99f76b83fab398be2c900c55fe5`。実行local commitとはcommit metadataが異なる |
| B1科学SOURCE／保存結果 | `8c6eeb02e89f6cf14870789719493e75482f62f8`／`d3913866e6ade4ab97ac02645fed91b2d5fcfa75`。B1 result・SOURCE・STOPと輸送定義を照合 |
| CDF8/r32科学SOURCE | `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`。元Hとcandidate Hを別々に扱う生成経路を確認 |
| 保存carrier | `3649b75af191916daf44abb91e14d34c43b83e8f`。科学SOURCEそのものと区別 |
| 元h/g/scalar packet SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| CDF J/U endpoint SHA256 | `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| 保存physical H SHA256 | `92a8d27bdebb53937f6361c106a13ef90b9110930d4bf4ddb577796cfb208739`、sector36のH／basis。NPZのhash・headerを確認し、配列数値解析はしていない |

[SOURCE.json][SOURCE]に実行SOURCE、関連コードhash、17件の固定参照commit／path／bytes／hash、runtimeを保存した。元H／CDF／sector／encoding／scalar・identity／law・normalizationの対応をB1 SOURCE、CDF固定契約と生成SOURCEから確認した。old physical JSONの共通stateラベルだけで同一性を認定していない。過去の契約・SOURCE・input・結果・grant／ledger／marker／STOPは変更していない。

旧B1・B3・B4-1のSTOPはそれぞれの終了履歴であり、今回の新しい明示科学実行許可と区別する。新grant・承認証明は作っていない。上表でlocal実行SOURCEと、GitHub接続で同じSOURCE treeを公開したcommitを対応させた。SHAが異なっても実行時SOURCE.jsonの原記録は書き換えていない。

## 2. 再利用した評価と13-round結果

既存 `OriginalHRoundEvidence.from_b1_row → assess_round → assess_energy` を使い、accuracyの共通幾何、既存 `_hoeffding_shots`、horizonを呼んだ。[runner][RUNNER]の追加は固定input／SOURCE照合、明示したtarget・premise、結果保存だけ。B4-1 library、B1輸送、shot式、decoderは変更していない。

B1のεmはPF＋normalization-corrected RTE＋元Hとの差のDuhamel輸送を同じoperator normで含む。今回のraw誤差・半径・phaseは

\[
d_m=A_m\epsilon_m,\quad \rho_{m,lb}=A_m-d_m,\quad
\beta_{det,m}=\arcsin(d_m/A_m),\quad
\epsilon_{coord,m}=\rho_{m,lb}\sin(\pi/12)/\sqrt2.
\]

idealモデルの定義としてimplementation／lawの追加raw誤差を0とした。actualの未知誤差0という認定ではない。正半径と `beta_det+beta_stat<=beta_total` を通ったrowだけ、既存Hoeffding helperで各軸の十分Nを算定した。情報論的な最小Nではない。

下表は保存[result.json][RESULT]の表示桁を丸めたもの。全rowのA・operator ε・raw誤差・半径・phase／stat allowance・coordinate・axis別α／N・statusは同JSONにfull binary64桁で保存している。βstatは全rowでπ/12、phase余裕は `βtotal−βdet−βstat`。

| m | q | t | A | radius下界 | βdet upper / rad | phase余裕 / rad | coordinate許容幅 X=Y | N_X | N_Y |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.1 | .999960931398 | .999882796671 | .000078137780 | .261721250019 | .182991252194 | 512 | 512 |
| 1 | 2 | 0.2 | .999921864321 | .999766003794 | .000155872707 | .261643515092 | .182969877614 | 512 | 512 |
| 2 | 4 | 0.4 | .999843734748 | .999535219094 | .000308563877 | .261490823922 | .182927641083 | 512 | 512 |
| 3 | 8 | 0.8 | .999687493915 | .999094707957 | .000592971300 | .261206416499 | .182847021949 | 512 | 512 |
| 4 | 16 | 1.6 | .999375085490 | .998347758370 | .001027969694 | .260771418105 | .182710320687 | 513 | 513 |
| 5 | 32 | 3.2 | .998750561498 | .997372230520 | .001380055710 | .260419332089 | .182531786700 | 514 | 514 |
| 6 | 64 | 6.4 | .997502684092 | .995444855741 | .002062981728 | .259736406071 | .182179052634 | 516 | 516 |
| 7 | 128 | 12.8 | .995011604771 | .991116564757 | .003914577421 | .257884810378 | .181386920406 | 521 | 521 |
| 8 | 256 | 25.6 | .990048093629 | .982570498434 | .007552831359 | .254246556440 | .179822881718 | 530 | 530 |
| 9 | 512 | 51.2 | .980195227698 | .965447249247 | .015046528012 | .246752859787 | .176689109619 | 549 | 549 |
| 10 | 1024 | 102.4 | .960782684402 | .931877160000 | .030089933130 | .231709454669 | .170545356883 | 589 | 589 |
| 11 | 2048 | 204.8 | .923103366647 | .867541179875 | .060227048343 | .201572339456 | .158771055332 | 679 | 679 |
| 12 | 4096 | 409.6 | .852119825516 | .749331105632 | .120921543224 | .140877844576 | .137137110254 | 910 | 910 |

全13 rowは `CONDITIONAL_ROUND_PASS`。最小半径はm12の `.7493311056315545`、最小phase余裕はm12の `.14087784457561348 rad`。合計は

\[
N_{\mathrm{total}}=\sum_{m=0}^{12}(N_{m,X}+N_{m,Y})=14738.
\]

各軸合計は7369。mのcoherent反復数qやr32を量子shot数へ重ねて掛けていない。これは実行した量子shots数ではなく十分shot数の計画値。[axis_shots.csv][AXIS]に26行のm/q/t/axis/N/α/資格を保存した。

## 3. 全round・energy・failure・decoder条件

| 条件 | 保存判定・意味 |
|---|---|
| 正半径・phase | 13/13 PASS。B1 geometryとのbinary64技術照合abs2e−15を通過。物理的phase予算やfailure上限に許容超過は追加していない |
| 26軸failure union | `fsum(2*alpha_axis)`=.01、上限.01以内。round間独立はunion bound自体に不要。各軸内の独立±1測定はideal IIDモデルの前提 |
| horizon | 必要最小M=12、qmax=4096。直前q=2048の幅は `.0025566346464760683 Ha` で目標を超える。連続dyadic m0..12を使用 |
| 元H energy幅 | `βtotal/(h*qmax)=.0012783173232380342 Ha <= .00159 Ha`、`CONDITIONAL_CA_PASS`。DF差・reserveを再加算しない |
| 符号・分枝 | φ=−hE、E=−φ/h。scalar／identityを二度補正しない。既存連続dyadic最近branch decoderの条件を使用 |
| 実際のdecoder成功率 | **未評価**。測定データ、Monte Carlo、量子shot simulationは使っていない |

宣言した誤差・状態・law／独立性・alias前提が成立する場合に、全coordinate事象とRPE phase条件を通じてfailure≤.01の十分条件を与える。これは99%の実測成功率でも、実装全lawに対する新しい数学的certificateでもない。保存operator norm＋guardは数値pilotで、丸め誤差の厳密包囲ではない。

## 4. Physical H・prior／aliasの資格

CDF科学SOURCEの`execute`は固定元H inputと`physical_reference.H`を読み、`ground(physical['H'],shared,'physical')`を呼び、candidate Hのgroundとは分離している。元H packet／physical H／CDF hash、sector36、元frame・Qiskit bit-reverse対応を固定契約とB1 SOURCEに照合した。保存physical ground JSONの`state='own_DF_ground_state_oracle'`は共通helperの出力ラベルで、これを元Hの同一性根拠にはしていない。

同じ保存physical Hの数値最低固有対記録はenergy=`−2.1663874486347625 Ha`、residual=`1.7642254846760187e−15`、normalization error=`7.771561172376096e−16`。旧gateはwindow[-10,10]内包含とresidual／normalizationを確認済み。今回その入力・生成SOURCE対応を再照合し、採択prior／h／βから

\[
\pi-\{.1\max(|-10|,|10|)+\pi/6\}=1.6179938779914944>0
\]

を評価した。結果は `CONDITIONAL_NUMERICAL_PRIOR_PASS`。priorへの包含根拠は保存binary64数値groundであり、新しい厳密spectral enclosureではない。元Hの厳密ground oracleは評価上の仮定。今回新ground solve・保存ground vectorの展開／state-actionは0である。

## 5. Ideal／actualとStage Cへの受渡し

| 区分 | 今回の到達点 |
|---|---|
| ideal lane | ideal finite-RTE IID・exact native、元H厳密ground、B1数値operator誤差＋guardを前提としたradius／phase／Hoeffding十分Nとenergy／failure整合性を確認 |
| actual lane | implemented finite-bit law、shot／occurrence IID bridge、synthesis／compiled誤差、Reference χ／adjoint／leakage、全law・全round correctness coverageはnull／UNQUALIFIED。actual N・total shotsもnull |
| 同じNの方式間共用 | 未認定。Reference／強いsymmetricの同whole ordered path・sampling law・normalization・fresh測定分布、全使用domainの実装誤差予算が必要 |
| 価格・全RPE費用 | **未評価**。有限fixture postcheck残差を全law boundへ移送せず、N×期待価格の乗算もしていない |

[Stage C handoff][HANDOFF]は元H／CDF／sector／law／SOURCE・target・axis別N・Aの定義・測定規約・資格をまとめる。raw平均のnormalizationは保存CDF8の `A=(1+tau^2)^(-r*2^m/2)`、tau=λ_R h/r、r32/K0。各rowにAそのものを保存する。raw resultの`Stage_C_handoff.normalization`はlaw descriptorであり、数値normalization定義はhandoff文書・JSONを使う。

将来は同じscope・law・round／axisのfull-wrapper期待価格へ `G_g=sum N_m,b E[C_g,m,b]` で接続する。RZ／CX／depthは資源別、peak qubitsは加算せず別幅として扱う。long-q coverage、compiler／合成精度、価格標本単位・shared calibrationの共分散も必要。C4のbranch・SOURCE・環境・STOPには介入していない。

## 6. 技術検証・資源・保全

[人工テスト7件][TESTS]は全てPASS。新runnerのinput hash拒否、人工operator13行からの接続、非実行CLI、およびB4-1既存の単位／DF一回／ideal-actual／horizon・failureを確認した。人工PASSを科学結果の代用にせず、その後固定SOURCEで分子保存入力の評価を1回だけ実行した。

[resources.json][RESOURCES]のwallは2.598480953秒、process peak RSSは560372 KiB、affinity64 CPU、MemAvailable開始510639152 KiB／終了510561824 KiB。13行の短いscalar評価のため単一processで十分であり、workerやCPU/RAMへ人工capを設けていない。数値package importを含むRSSで、並列speedupやaggregate peakの主張はしない。Python3.11.0rc1、numpy1.26.4、scipy1.14.1、openfermion1.6.1、openfermionpyscf0.5、qiskit1.3.0を既存project専用venvから読取専用利用。過去のnorm計算runtimeと同一であるとはしていない。

旧SOURCE・保存input・raw結果・grant／ledger／marker／STOPを保持し、新結果は専用namespaceに分離した。再計算していないPF/RTE norm・保存physical groundは旧証拠の数値資格を継承する。新sampling、量子shots、Monte Carlo、ground solve、DF refit、B2、回路build／compile、全回路費用、詳細科学レビュー、C4変更はいずれも0。retry/resume0。[STOP.json][STOP]

実行したコマンドは以下の一回のみ（`TRACK_D_PYTHON`は使用した既存project専用環境）。再実行指示ではない。既存run01があればrunnerは起動を拒否する。

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$TRACK_D_PYTHON" -B scripts/run_track_d_b4_2.py --execute
```

## 7. 完了境界・次の担当

**科学実行は完了しmandatory STOP。** 許可された記録・公開・SHA／tree／bytes照合だけを行い、B2、新sampling、actual保証、回路compile、総回路費用、詳細科学レビューへ進まない。

**次担当：ユーザー／GPT。** この条件付きNをStage C4以降へ接続する際のclaim水準と、全law／long-q価格・実装誤差資格の不足に対する次の最小scopeを判断する。**Codexは別の明示許可後**、その範囲の技術接続・検証を行う。B1十分条件は今回も成立し、B2を自動必須にする根拠は生じていない。

[SOURCE]: ../../../artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01/SOURCE.json
[RESULT]: ../../../artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01/result.json
[AXIS]: ../../../artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01/axis_shots.csv
[HANDOFF]: ../../../artifacts/track_d_b4_2_rpe_shots/2026-11-06/Stage_C_handoff.json
[TESTS]: ../../../artifacts/track_d_b4_2_rpe_shots/2026-11-06/technical/synthetic_junit.xml
[STOP]: ../../../artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01/STOP.json
[RESOURCES]: ../../../artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01/resources.json
[RUNNER]: ../../../scripts/run_track_d_b4_2.py
[PUBSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/6da5afba79895c6b3c092f41e332d444dedc8ed3
