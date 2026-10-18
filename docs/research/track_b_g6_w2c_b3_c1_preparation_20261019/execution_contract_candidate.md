# B3/C1最小実行契約候補（未採択・未認可）

[machine候補](execution_contract_candidate.json) / [入口](README.md) / [handoff](handoff.md)。`adopted=false`、`contract_grants_execution=false`。この文書・guardはauthorization/grant/marker/claimを生成しない。**候補を編集しても、補助SOURCEの`reject_real_execution`は常に実実行を拒否する。** 実runnerは正式条件固定後に接続し、SOURCEを別途固定する必要がある。

## 1. 固定対象・再利用

G7 `P5_general_order`、p=(1/5,3/10,1/2)、x=5/7、m=5、Ordinary/Partial Return＋tail/Closed P3＋tail/Full。入力/law/root/ceil/η/ρ/H160/K256、ε_axis=1/200、primitive_error=1/1000000、α_axis=1/320と元16-axis familywise≤1/20、strict phaseを旧契約から継承する。実CPU confidenceではない。

既存N・digital受理率・原係数・有限RZは[引用表](existing_evidence_reuse.json)の元SHA/path/JSON pointerから再利用する。Nや受理率、reference momentを計算し直さない。既存の科学claim/STOP・消費済みmarkerは変更しない。G9 P5Closedとは比較しない。

## 2. B3候補call範囲

- frozen G7/F3の固定SOURCE＋新補助adapterのみ。科学対象・sampling lawを変更しない。
- setup/事前検査・記録読み込み、各方式のR/C/A・canonical各mode、angle/guard/requery/serialization/outputの観測を含む。
- 第一候補は、層Aと層Bそれぞれで4方式・X/Yの**保存N分の完了した固定workload**を一度観測する。各arm/layerのsetup1回、X/Y共有はstate不変を確認して初めて適用する。seedは既存G7診断seedを候補として引き継ぐが、全Nへの延長は新実行であり採択・承認を要する。
- CPU/profileの前後で科学数値やbudgetを再計算しない。必要な初回correctnessはidentity・event/zero/weight/phase・law・角度の固定gateへ限定し、独立B1/B2計算を自動追加しない。
- 実sampling/profile call数は選んだscheduleにおける保存Nとcallback回数で固定する。現候補の未決scheduleを埋める前にrun数を消費しない。

**未採択事項**：固定workloadの記述的実費か、fresh-IID母期待費用か。後者なら費用測定のsample数・confidence/停止規則が必要。角度τ/K_phi/J、どの層を測るか、shared/cold cache、保存retention・observerの粒度も正式固定が必要。CPU順序や一回の観測から母期待値優位を認定しない。

## 3. C1候補call範囲

優先は旧RZ cacheの同一identity再利用（**新RZ synthesis 0**を候補とする）。物理CQはQ/encoding/sector/basisを先に固定する必要があり、現候補ではnull。gate set/compiler/version/seed、ancilla、strict phase/error、common optimization/cacheを全方式に同条件で採用する。

第一の費用モデル候補はliteral primitive価格pool。label×control役割×forward/actual-adjointの必要keyを一回ずつ取得し、全方式が共有する。actual adjointはforward artifactとの厳密な関係・位相を保つ。既存G7 CQ error allowance=0と互換でない実装はSTOPし、勝手にbudgetを変更しない。

word/angle別のwhole-event compileやdepthが必要なら、それは追加scopeとし、対象event/key・科学call数・新B2縮約の有無を固定して別途採択する。全eventを勝手に列挙・compileしない。未知prep/readout/backendはnullとしてC2を除外した限定結論にとどめる。

## 4. 算術・統計・結果

N/π/price区間はexact rational/integerと固定来歴を保持。CPUはprocess time ns、wallはmonotonic ns、memoryはper-process観測/allocator/bufferを別欄。F2 bit上界はsymbolicでありCPU→bitの換算をしない。値のupper同士の比較、未知=0、区間重複=tieを禁止する。

`schemas/`はexact seal形式・stage record・provider価格・実行候補のshapeを記述する。source/input/policy/law/price/cache/compiler digest、call数、prefix、typed failure、output bytes/hash、STOPを記録する。新補助SOURCEが検査する範囲と外部numeric checkerが必要な範囲を分ける。scientific GO/NO-GOは結果schemaに自動判定させない。

## 5. 保存・資源・one-shot

science実行時は新しい専用結果領域と別途承認されたauthorizationを固定し、初回correctnessを含めone-shot1回、retry/resume0、完了/失敗の双方でmandatory STOP。旧ledger/marker/claimへ書き込まない。失敗prefixと消費資源・来歴を保全し、failureをzero扱いしない。再試行は新しい明示承認・独立記録が必要。

新規候補にCPU/RAM/worker/wallの人工上限は設定しない。空き資源・IO競合・aggregate memoryを確認し、独立profileのprocess分離/並列化を選ぶ。同一sampling/bit順序・precision・比較条件を変えない。共有環境への設定変更・他process操作を禁止する。旧固定契約のresource capsはそのまま保持する。

保存上限の**技術候補**はresult64 GiB、log256 MiB。予約ではなく、streamed stage summaries・hash・必要prefixを保全するための上限。実際の空diskとretentionを実行前に確認する。巨大証拠は安全な原本＋manifest/SHA/軽量要約で公開し、通常GitHubの制約を超えるファイルをpushしない。これらも現候補のままでは実行許可にならない。

## 6. 採択・承認が必要な項目

1. **ユーザー**：B3のcost estimand・層/schedule・angle policy・保存方式を採択する。
2. **ユーザー**：C1を行うなら物理Q/encoding/provider/compiler/誤差互換性と価格モデルを採択する。条件が足りなければC1を保留し、B3だけを独立に扱える。
3. **Codex**：採択済み条件を実driverへ接続し、SOURCE/guard/call予算を固定する（今回real driverを実行していない）。既存科学SOURCEは上書きしない。
4. **ユーザー**：具体的なB3 sampling/profilingと必要C1 compile/price取得、必要なら追加B2縮約を**別途明示承認**する。
5. **Codex**：初回correctnessから承認範囲だけone-shot実施し、成功/失敗を保全・公開してmandatory STOPする。

実行後に答えられるのは、採択したscopeにおけるCL/Q条件と各費用座標の順位である。C2未取得なら総End-to-End資源優位は判定しない。**現在はここでmandatory STOP。**
