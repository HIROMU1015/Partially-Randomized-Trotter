# Track D Stage B4-1：元H向けRPE統計の最小接続実装

2026-11-06 JST。**元Hamiltonianの厳密基底状態を明示する統計adapterを実装し、新規人工52件・既存回帰21件がPASSした。** own-DF専用guardを保持し、raw誤差→radius／phase／Hoeffding Nと全round energy／failure会計を既存処理へ接続した。これは実装・人工検証の結果であり、実H4のN、phase／radius／energy、成功率、全RPE費用の評価ではない。B4-1の公開照合後mandatory STOP。

## 1. 固定SOURCEと実施範囲

専用branch／worktreeは `track-d-b4-1-rpe-statistics-implementation-20261106`。基点は[B3公開commit][B3] `89a13fe6841c1017f0feba724ed71dfc79f9e81b`。B3 §2・§3・§6と今回のB4-1指示を基準にした。共通／repository AGENTS.md、PROJECT_MAP、研究総説、正式共通指示2026-10-25、関連SOURCE・既存STOPを確認した。正式共通指示はlocal原本を参照し、存在しないrepository公開リンクは付けない。

研究前提はlinear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、元H厳密groundを仮定、準備費除外を維持する。今回はこれらの分子入力で計算していない。保存13-round、.00159 Ha、failure .01、π/6／π/12等を新しいtargetに自動採用しない。

| 固定資料・既存資産 | 今回の参照・利用 |
|---|---|
| B3 `89a13fe6841c1017f0feba724ed71dfc79f9e81b` | 実装前SOURCEと最小接続方針、誤差単位、資格・実行境界 |
| B1/C1結果 `d3913866e6ade4ab97ac02645fed91b2d5fcfa75` | 実行SOURCE `8c6eeb02e89f6cf14870789719493e75482f62f8`、B1輸送と保存schema、STOP。実H4保存rowから新規Nを算定していない |
| B/C静的準備・Stage A・C2 | `5f94972adf5a10e96e9400d3919c00d14e7a718a`／`3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0`／`c8fe5091c42ec786688b6931105ad64d10c87d18`。全law・長q・Nの未資格を維持 |
| `c1_stage3_accuracy.py` | own-DF API・state guardを保持。計算本体だけを共通内部helperへ抽出 |
| `track_d_b1_c1.transport_round` | 人工scalarからB1形式rowを作るテストで利用。輸送の別実装なし |
| `rpe_resource_accounting._hoeffding_shots` | 既存ceil・range処理を直接利用。unit-radius候補APIを元Hへ偽装しない |
| `required_rpe_round_horizon` | energy精度から必要horizonを求める既存関数を利用。旧own-DF reserveを加えない |
| `reconstruct_rpe_phase` | 提供済み人工X/Y平均からの連続dyadic最近branch復元を利用。sampling・Monte Carloは呼ばない |
| 既存accuracy／accounting／decoderテスト | scalar・既成位相列だけを扱うnodeを選択。同じtest file内のsampling・回路生成を伴う別nodeは実行しない |

[SOURCE.json][SOURCE]は基点と変更前blob、新revisionのコード・テストhash、実際に再利用した依存の版、Python／package版を記録する。これは実装来歴であり承認証明ではない。人工テスト後・公開前の内容を記録したもので、科学実行前に封印したSOURCEやclosure全体の再監査とは主張しない。実装前のファイルは基点Git objectに保持されている。

C3のbranch／worktree／SOURCE／入力／結果／STOP／processは変更・操作していない。旧契約・grant／ledger／marker／STOPも改変・再消費していない。既存project専用Python環境を読取専用で使い、bytecode出力を無効化し、test cacheは専用worktree内に限定した。共有環境・OS・scheduler・他ユーザーのファイル／processは変更していない。AI sub-agentは使用していない。

## 2. 変更理由とown-DF互換性

[新module][CODE]は実験track用の薄いadapterで、分子input loader・科学実行CLIを追加しない。importはstdlibと既存accuracyだけ。評価関数の明示呼出時には既存loaderから数値helperを得るため、その既存依存packageは必要だが、分子constructor・solver・sampler・compilerは本adapterから呼ばない。

[既存accuracy][ACCURACY]では次の本体だけを内部helperへ抽出した。

1. `_assess_raw_signal`：raw complex-signal誤差、正半径、arcsin phase、phase予算、共通coordinate幅、既存Hoeffding N。
2. `_assess_energy_interval`：全horizonの連続性、round status・target一致、全axis failure和、最終phase→energy幅。own-DFは従来どおりWeyl shiftとreserveを渡し、元Hは輸送済みDFを再加算しない。

状態・law・aliasの資格確認はそれぞれのwrapperが担う。共通helperだけで資格を付与しない。`AccuracyTarget`、`RoundEvidence`、旧`assess_round`／`assess_energy`の公開signature・state guard・科学的意味は維持した。元H state名を旧経路へ渡すと拒否される。既存`aggregate_totals`、Hoeffding helper、decoder、horizon、B1輸送は変更していない。

元Hではstateモデル・DF計上・targetの意味が旧APIと異なるため、専用target／binding／wrapperを追加する必要があった。shot式・decoder・分子計算・価格推定の重複実装は追加していない。軸別Nや非一様配分は今回不要なため、既存の共通座標幅・共通αaxisを維持した。

## 3. Interface仕様と資格

### 入力とentry

| 型・関数 | 明示する入力と処理 |
|---|---|
| `SignalBinding` | `original_h,cdf,sector,law,source,implementation`。空でないidentityを要求し、expected bindingと各evidenceを照合。sourceはoperator誤差等の出典を特定する |
| `OriginalHTarget` | energy精度[Ha]、h、βtotal／βstat、αaxis／αtotal、最大round M、energy prior windowを全て明示。科学値の既定値なし。Mが既存horizon関数の必要M未満なら拒否 |
| `OriginalHRoundEvidence` | m、A、元H exact evolutionまでのnormalization-corrected operator誤差（DF込み）、implementation／lawのraw complex-signal誤差、binding、law model、alias／IID条件とprovenance、総合的誤差資格 |
| `from_b1_row` | m/q/timeを確認し、B1 rowのAと`original_H_corrected_operator_error`を選ぶ。旧N・radius・phase配分は採用しない。callerが元packetをbindingし、B1資格はnumerical pilotに固定 |
| `assess_round` | state／単位・入力・資格を確認後、共通幾何へ接続。各軸あたり共通の十分shot数`shots`を返す。最小Nの主張なし |
| `assess_energy` | 同じtarget／state／SOURCE／方式、m0..M、lane統一、failure和を確認。DFの追加energy加算なし |
| `decode_energy` | qualified planとm順の提供済みX/Y平均を既存decoderへ渡す。入力生成・shot simulation・成功率評価なし |

binding・provenanceは科学入力の取り違えを防ぐための情報であり、入力文字列だけからboundやIIDを証明・認証する仕組みではない。callerは後続実行時に対象SOURCE／入力、根拠の適用範囲を確認する。新grant、approval authority、承認証明台帳はない。

### 単位・誤差・N

stateは `exact_original_H_ground_state_oracle`、operator誤差の意味は `original_H_corrected_operator_norm_DF_included_once`。own-DF／HFや別の単位ラベル・非対応lawを拒否する。数値の出典自体の正当化は外部証拠に依存し、ラベルを付けるだけで正しいboundになるわけではない。

\[
d_m=A_m\epsilon^{\mathrm{original\ H}}_m+e^{\mathrm{raw}}_{impl,m}+e^{\mathrm{raw}}_{law,m},\quad
\rho_{m,lb}=A_m-d_m,\quad \beta_{det,m}=\arcsin(d_m/A_m).
\]

正半径と `beta_det+beta_stat<=beta_total<π/3` が成立するときだけ、既存 `epsilon_coordinate=rho_lb*sin(beta_stat)/sqrt(2)` と `_hoeffding_shots` を呼ぶ。raw追加項へAを重ねて掛けない。DF差はoperator輸送内で一度計上済みで、元H energy幅は `beta_total/(h*2^M)`。旧DF reserveやWeyl差を重ねない。

各rowの`shots`／`alpha_axis`をX/Y両軸に使用し、全roundのfailureは既存 `fsum(2*alpha_axis)` で判定する。αtotalを超えた値を許容差でPASSにしない。非一様round／axis配分・weighted／postselection・非連続q・shiftしたenergy branchは未対応として後続scopeへ残す。

| Lane・状態 | 出力・資格 |
|---|---|
| ideal IID／exact native | `ideal_finite_RTE_law`、両追加誤差を明示的に0、model前提のprovenanceを要求。`CONDITIONAL_IDEAL_MODEL`。actual実装の誤差0の証拠ではない |
| implemented | `actual_implemented_law`、raw implementation／law誤差とprovenance、alias／IID前提を要求。`CONDITIONAL_IMPLEMENTED_ERROR_PREMISES`。外部premiseが妥当な場合の条件付き結果 |
| 誤差null／provenance不足／alias・IID未資格 | `UNRESOLVED_SIGNAL_OR_SAMPLING_PREMISES`、Nはnull。未知誤差を0へ変えず、energyも未資格 |
| radius≤0／phase超過 | `INFEASIBLE_RADIUS`／`INFEASIBLE_PHASE`、Nを算定しない。不十分条件を物理的不可能性の証明にしない |
| shot計算underflow／overflow | `UNRESOLVED_NUMERICAL_SHOT_RANGE`。clipしない。異常な型・欠損・非有限値・非対応条件は例外として拒否 |

出力はtarget／binding／入力誤差・provenanceを残し、`unconditional_success_claim=False`、`success_probability_evaluated=False`。B1 adapterはbinary64 norm＋engineering guardをrigorous enclosureへ昇格しない。直接evidenceの`evidence_kind`はoperator・追加誤差を含む総合資格であり、全根拠がrigorousでない場合は`rigorous_bound`を選ばない。

### Decoder・alias・energy区間

符号はU(t)=exp(−itH)、φ=−hE、E=−φ/h。H→core→YのみSdg→H、bit0→+1／1→−1のX/Y平均を前提に、`atan2(Y,X)`をm順で既存`reconstruct_rpe_phase`へ渡す。E0を正解branch選択に与えない。

今回のprior windowは `h*max(|lo|,|hi|)+beta_total<π` により誤差余裕を含め主値分枝内へ収まる範囲。逆順・非有限・越境をconstructorで拒否する。zero coordinates、数値branch tieは未解決。最終energy誤差区間がpriorと交わらなければ`ENERGY_BRANCH_OUTSIDE_PRIOR`、交われば交差区間を返す。区間包含は宣言した全系統・座標誤差事象が成立する場合の条件付きで、推定1件から成功率を測定したとはしない。

## 4. 人工検証と再現範囲

| 検証 | 実施内容・結果 |
|---|---|
| [新規人工fixture][TEST] | 52 PASS。state、6種binding、単位、人工B1輸送との接続、DF一回、未知誤差、ideal／actual、radius／phase／failure／horizon、異常入力、helper N一致、decoder符号・wrap・tie・prior、既存費用interface |
| 既存回帰 | 21 PASS。旧accuracyの13関数node（parametrize込み16件）、B1 scalar2件、既存decoder3件。own-DF guard／Weyl／failure／scopeの従来動作を確認 |
| 意図しない処理の防止 | passive import、および人工評価中にground solver／matrix power／expm／sampler／QuantumCircuit constructorを禁止するテストを含む。分子保存fileを開かない |

人工targetは `.125 Ha,h=1,beta_total=.4,beta_stat=.2,M=2,window=(-1,1)` 等のtoy値。B1輸送testも人工scalarだけで、実H4の保存数値からのNではない。人工費用の加算は既存interfaceの検査であり、科学総資源評価ではない。

[新規XML][SYNXML]／[回帰XML][REGXML]／[回帰出力][REGLOG]／[結果概要][RESULT]に保存した。公開XMLから不要なhostnameだけを除去し、判定・件数・時刻・時間は保持。Python／package版はSOURCEへ記録する。今回の主エージェントによるlocal人工検証であり、immutable CIや独立レビューではない。検証済みコードを変更しない再開時は、保存結果・SOURCE同一性を確認し、同じテストの不要な再実行を省いた。

再実行は明示人工entryだけを使う。`TRACK_D_PYTHON`は必要依存を持つ専用環境とする。

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$TRACK_D_PYTHON" -B -m pytest \
  -p no:cacheprovider -q tests/check_track_d_original_h_accuracy.py
```

既存回帰の明示node一覧はSOURCEに記録した。通常の全test discovery、sampling・回路生成を含む別test、旧科学runnerは起動しない。

## 5. C3接続interfaceと未解決事項

| 将来の受取 | 接続条件 |
|---|---|
| 元H qualified N／方式別誤差 | `SignalBinding.implementation`でReference／symmetricを区別。全whole-pathの測定対応、同law／normalization／attenuation、fresh state／trajectory、actual adjoint／χ、両実装の誤差予算を確認するまで同じNを認定しない |
| round／axis full-wrapper価格 | 方式・law・compiler・合成精度・scope、whole-wrapper depthかscheduled serial sumか、valid q／long-q coverage |
| 期待値とfinite fixture | Stage A/C2/C3の固定fixture価格を全law期待値へ格上げせず、欠落を0にしない |
| RZ／CX／depth | 共通N形式は既存`aggregate_totals`へ同target／同scopeで接続可能。`expected_scope`文字列を変えるだけで資格を得たとはしない。実13-round費用は未集計 |
| qubits | 加算METRICSへ入れずpeak widthとして別保持。N・qを掛けない |
| 古典価格不確かさ | n、paired差、shared calibration covariance、support／SE／区間資格を別に扱う。価格nは量子shot倍率でない |

C3結果・実装は今回取得・変更していない。現APIが受け付けるprovenanceは、全law boundやIIDの新証拠そのものではない。非一様配分、shifted branch、実backend誤差、実H4の数値資格は後続で必要な範囲だけ確認する。

## 6. B4-2の科学入力・許可とSTOP

**次担当はユーザー／GPT**：energy精度・成功確率・β／α配分・round範囲・alias prior・ideal／actualのclaim水準を決定する。旧の13-round等を自動採用しない。

**ユーザーの明示科学実行許可後のCodex作業案**は、新SOURCE revision、元H／CDF／sector／lawとB1 inputのbindingを確認し、採択targetのradius／phase再評価、Hoeffding十分N、全round／failure／energy条件を評価すること。SOURCE・保存input・round・再解析項目・精度・出力／STOPを許可範囲に含める。今回の実装・人工検証の許可はB4-2科学実行許可ではない。

actual保証には全使用domainのimplementation raw誤差、implemented-law biasとjoint-law／IID bridge、alias根拠の資格が必要。未知ならactualを未資格に保ち、ideal条件付きと分ける。B1十分条件で足りればB2は不要。B2、sampling、ground solve、DF refit、build／compile、成功率simulation、総RPE費用、C3介入、詳細科学レビューは別scopeで自動開始しない。

後続STOP条件はSOURCE／input／binding不一致、state／単位不整合、alias／IID未資格、radius／phase／failure不通過、horizon不足、overflow／underflow、全law誤差や価格coverage不足、未許可scopeが必要な場合。partialを保全し、未知値0・閾値緩和・目標変更でPASSを作らない。旧one-shot／retry・resume／mandatory STOPを維持する。

今回の[STOP記録][STOP]は技術完了を記録する。実装・tests・文書・技術成果物を専用branchへ公開し、remote SHA／tree／公開bytesを照合して終了する。Git認証が不足する場合は共有認証を変えず、許可されたGitHub接続で公開した場合の元commitと公開SHAの違い・同内容照合を最終チャットで報告する。

**mandatory STOP。実H4のN、B2、成功率、総資源、C3介入、詳細科学レビューへ進まない。**

[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89a13fe6841c1017f0feba724ed71dfc79f9e81b/docs/research/track_d_b3_rpe_statistics_preparation_20261106/REPORT.md
[CODE]: ../../../src/trottertracks/track_d_original_h_accuracy.py
[ACCURACY]: ../../../src/trottertracks/c1_stage3_accuracy.py
[TEST]: ../../../tests/check_track_d_original_h_accuracy.py
[SOURCE]: ../../../artifacts/track_d_b4_1_rpe_statistics/2026-11-06/SOURCE.json
[RESULT]: ../../../artifacts/track_d_b4_1_rpe_statistics/2026-11-06/RESULT.json
[STOP]: ../../../artifacts/track_d_b4_1_rpe_statistics/2026-11-06/STOP.json
[SYNXML]: ../../../artifacts/track_d_b4_1_rpe_statistics/2026-11-06/technical/synthetic_junit.xml
[REGXML]: ../../../artifacts/track_d_b4_1_rpe_statistics/2026-11-06/technical/regression_junit.xml
[REGLOG]: ../../../artifacts/track_d_b4_1_rpe_statistics/2026-11-06/technical/regression.txt
