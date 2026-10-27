# 研究C M3：Proposed PR s1/r64・A途中の技術STOP

2026-10-28 JST。**19/最大471 logical unitsを実行し、18 COMPLETE＋1 schema correctness FAILでSTOP。PF/RTE round評価は0/13、M3-B/Cは未実行である。** これはs1/r64の科学的INFEASIBLEという結果ではない。登録契約の「科学開始後の修正・retry/resume禁止」に従い、旧run01・SOURCE・契約を変更せず、追加科学評価をしていない。

## 固定SOURCEと範囲

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 起点：`a977adda5a9fa5a882ef8b79acb8c68cff8f0bb1`。
- Branch：`codex/research-c-M3-s1-r64-20261028`。
- 科学実行SOURCE：`9752f28cf5863e5caab815004a21d65b53c122d4`（実装固定`eb7631f4a`＋人工log公開）。
- [固定契約](research_c_pr_aware_df_M3_s1_r64_contract_20261028.json) SHA256：`faec54edc960414df750ffaa9478985102fafae565b5b84208db4fc9b0cfeb88`。
- [SOURCE81 manifest](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/technical/SOURCE.json) SHA256：`3fe877a1a8913ba98d5d75d2488dc2b3abcd4ca6b477f748a4104653fbda48be`。
- [技術準備・再利用・call内訳](research_c_pr_aware_df_M3_s1_r64_preparation_20261028.md)／[人工・回帰log](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/technical/tests_final.log)。
- [run01部分原本8件](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run01/)／[実行summary](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/execution_summary.json)／[postflight・raw hash](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/execution_postflight.json)。

保存直鎖H4・隣接1.00Å・STO-3G、通常number CDF rank8 Proposed、Nα=Nβ2/sector36、system256/last-ancilla full512、元g補正κ・核scalar一回・元frame0..7/prefix LD2・tail identity抽出を維持。計画はpartial ordered symmetric S2、δ=.1/s1/h1/10/r64/K0、m0..12/n=2^m。CA=.00159/DF reserve CA/8、βtotalπ/6・βstatπ/12、failure.01/axis.01/26、own-DF ground oracle・状態準備費用除外。

保存h/g/c input SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、Proposed J/U SHA256 `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`、元P-H4 provenance `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。新fit・optimizer・rank探索・ground eigensolve・DF誤差再計算0。

ユーザーはM3-A→条件付きB/C→公開まで一括許可した。人工検証・契約・SOURCEを公開固定後、追加承認なしでfresh run01を一度実行した。承認証明・原文保存・新grantは作成していない。科学開始後のSOURCE修正・再起動・別runへの移送は0。

## STOPの原因と到達範囲

最初のFAILは`tail_gate`、label `216 signed atoms/identity/mass/probabilities`、例外`ValueError: ATOM_ID_SIGN_BASIS_ORDER`。

[M3照合実装](../../src/trottertracks/c_pr_aware_df_m3.py)は次の直接比較を行っていた。

```python
[atom_record(x) for x in arm.plan.atoms] != old['atoms_catalog']
```

`MixedAtom.origins`と`word`はtupleであり、既存`atom_record`はそのまま返す。一方、保存JSONを`io.read/json.loads`で読み込むと対応箇所はlistになる。したがって、内容が同一でもtuple/list差だけでこの比較は不一致になる。人工fixtureは`atom_record`のPython辞書を直接保持し、JSON保存・再読込を経由していなかったため、この境界の不備を見逃した。**追加した照合とfixtureの実装上の不備である。**

これは静的SOURCE・保存schema・既存tracebackで特定した技術原因で、新しいH4 decode/再構築や追加の科学比較は行っていない。他のcatalog内容差まで全件排除したという主張はしない。今後はsequenceを明示的に正規化して内容を比較し、fixtureもJSON round-tripさせる最小修正が必要だが、**今回の実行後にはその修正を実装していない**。

保存SOURCE/input/evidenceの照合、Proposed J/Uのロード・shape/finite/J対称性/U直交性、元κ/scalar/LD2モデル、保存sector reference、全9 general-K対保存CAR因子・Hermiticity・sum、保存groundの残差/normalization/aliasまでは通過した。Ground gate：[原本](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run01/A/ground_gate.json)。このPASSは新PF/RTEの適格性やfull512 nativeの正しさではない。

Tail gate内では既存のsigned atom/CAR tail比較、λ・identity・216 atom数・probability列の照合が先に評価されるが、gate全体はFAILであり、PASS済みgateとして集計しない。代数τ/A gate、PF/corrected kernel、13 round評価には到達していない。

## 実行状態・call

| stage | 上限 | 実績 | 状態 |
|---|---:|---:|---|
| M3-A | 61 | 19 attempts、18 COMPLETE＋1 FAIL | 部分技術STOP |
| M3-B | 325 | 0 | 依存gate未通過のためNOT_RUN |
| M3-C | 84 | 0 | 依存gate未通過のためNOT_RUN |
| program summary | 1 | 0 | runner内は例外終了、外側の実行報告を保存 |
| **全体** | **471** | **19** | retry0/resume0、mandatory STOP |

19件はevidence/input/fit/model/sector-reference/context各1＝6、factor9＝9、ground-load/ground-gate/law-prepare各1＝3、tail-gate1。Journalにはattempt19/COMPLETE18/FAIL1、inflight0があり、[STOP](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run01/STOP.json)はfailed=true。数値operator比較のFAILは観測されていないが、correctness FAIL総数はschema照合の**1件**である。

新kernel build/compile/QPY、trajectory sampling、RPE shots/総費用評価はすべて0。Scientific INFEASIBLEや候補UNRESOLVEDを評価した段階にはなく、状態は**NOT_EVALUATED**。全law synthesis・implemented-law・費用support・actual CA等は引き続きnull。未実行の値を0で埋めない。

## Q1〜Q8

1. **τ/attenuationの旧s4/r16との整合**：設計上は同じ保存λとsr64から数学的τ・K0 Aが一致する。人工検証はPASSしたが、実H4の代数/数値整合gateは未実行で、実測PASSとは報告しない。
2. **全13round PF/corrected-RTEの変化**：未評価。旧B1値を新候補へ代用していない。
3. **Feasibility**：判定不能。登録十分条件の不適格ではなく、先行する技術gateのSTOP。
4. **64-event kernel RZ/CX/depth**：B未実行、標本平均・分散・SE・観測範囲なし。
5. **条件付きshots/総資源**：C未実行、比較値なし。
6. **同J/U決定論との資源差**：新しい比は未取得。保存M2の旧s4/r16対決定論約5.31倍という文脈は変わらない。
7. **Head反復削減仮説**：支持/不支持のいずれも判定できない。誤差や費用を観測していないため、macrostep変更の結論を出さない。
8. **次の一検証**：現時点では別のアルゴリズム変更を選ぶ根拠が不足する。最小の次作業はJSON境界の照合/fixtureを直し、同じs1/r64候補を新SOURCE・新契約・fresh outputで完遂すること。その結果でPF/RTEとhead価格・shotsのどこが制約かを判断する。旧run01のretry/resumeではない別明示許可が必要。

## 人工・資源・保全

人工・回帰は92 unique PASS（新13＋既存79、79.53秒）。最後にscenario一致のpassive検査を追加し、影響する5件を再実行してPASS。これらは既存B1/B2/C1/M2非回帰、人工64-event full512、32 mock dispatch/QPY/call計上、C1仮定/serial、partial STOP等を検証したが、上記JSON型境界を網羅していなかった。人工PASSを実H4適格性や科学的GOへ昇格させない。

[実行前](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/execution_preflight.json)：SOURCE81/input39、契約、依存版、fresh、保護原本125がPASS。開始時affinity32、MemAvailable約56.5GB、memory pressure avg10/60/300=0。既存plannerはB用に32 workers/BLAS1を予定したが、**Bの並列poolは起動していない**。実際はAの前処理中に停止した。CPU/RAM/worker/wallの人工cap、共有環境・他jobへの変更は0。

Runner wall **1.522346556186676秒**、process CPU差 **12.268730586999999秒**、parent peak RSS **371020KiB**。BLAS内部並列を含むprocess全体のCPU差であり、32 native workersを使用した結果ではない。Task-local venv/cacheを使い、既存依存はread-only。Science logのtracebackをそのまま保全した。

[Postflight](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/execution_postflight.json)でSOURCE81・保存input39・保護原本125がすべてhash不変。新run01 raw8件のhash・call journal・STOPも保存。旧A/B1/B2/C1/M1/M2、F2-K1/F3-K1とH2/H4/H6旧SOURCE/raw/grant/ledger/marker/STOPは変更していない。新科学evaluatorの追加呼出し0、旧科学callの残枠転用0。

結果・原本・logをcommit/pushし、remote SHA/tree、SOURCE/input/raw/call/STOPを照合後mandatory STOPする。**次はユーザー／研究C**が、照合修正と同条件のfresh科学実行を許可するか判断する。**Codex**はその具体的指示後にJSON round-trip人工試験を追加して修正SOURCEを固定し、新しい限定契約に基づいて対応する。今回の科学SOURCEや条件を変更せず、別s/r/K/γ、追加arm/H6/sampling/保証/詳細レビューへ進まない。
