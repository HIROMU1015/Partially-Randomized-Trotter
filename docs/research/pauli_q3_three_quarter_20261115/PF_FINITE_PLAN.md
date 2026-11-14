# Full PF／Truncated p7 shortlist：最初の有限Q3案

**計画のみ。今回のPF science callは0。以下のvector・6評価と数値追跡範囲は次回の明示承認対象。** Z1＋T1/Q3・solver proxy・全事象平方loss・τ=εE²というモデル評価方針はユーザー採択済みで、再採択を要求しない。

## 保存入力と6構成

完成13round signalとSOURCEは[PF_PACKET_ACCEPTANCE](PF_PACKET_ACCEPTANCE.json)からAのPACKET_INDEXへ戻る。各行の元configuration、full total-view、各round physical ID、元producerのlocatorを保持した。Cの候補表を作り直していない。

| 同公式のpair | Full PF 元 → total先頭 | Truncated B0I p7 元 → total先頭 | 新Q3案 |
|---|---|---|---:|
| S2・16h/7 | 11cf8ca6 → af18d37d | a6e55777 → e496358b | 各1、計2 |
| 標準Yoshida4・16h/3 | a799697f → 833ca2c4 | defc5e88 → 95b2c417 | 各1、計2 |
| M5・16h | 70f7b317 → a4a5e71a | 424eecbf → d1000f28 | 各1、計2 |

全てh=0.1/M12/同full DF-H target・state・sector・P0。PFのfinite-K/R/κは非該当で、PRのlawを継ぎ足さない。各pairは元formulaのsigned schedule・requested/actual幅・q/Tを保持し、Fullとp7のretained Hamiltonian/scalar差は物理差として保持する。B0Iはfull PFではない。full scalar c／p7のc＋tail identityを一度だけ含む元packetの規則を変更しない。

## whole-vector案と根拠

既存C `c628e9e8:src/trottertracks/paper_nominal_cost.py`は、Bの`pr_rpe_rmse.nominal_axis_shots`を再利用し、`ceil(B_m²[11+4(M−m)])`をnominalに用いる。Aのdeterministic PF packetはB=1/logB=0であり、PRのnormalization inflationを持たない。この**既存のdeterministic用の整数scheduleを、Q3で初めて検査するUSER_SUPPLIED controlとして提案**する。

```
N_X = N_Y = [59,55,51,47,43,39,35,31,27,23,19,15,11]
```

これはm=0..12に対する`11+4(12−m)`という設計式の展開である。実signalから新しいNを求めたり、nominal_packetを実行して十分性を評価した結果ではない。既存paperモデルとの追跡性があり、同pairで同じvectorを使うことで、最初の比較を異なるshot policyの影響と混同せずに行える。**Pauli controlや3/4配分の成功を移送した案ではなく、PFの誤差を補償できるという予測でもない。** paper型モデルは旧契約を参照しており、total decoder・Q3で成立する保証はない。その適合性を新しい6診断で調べる。

allocation ID案：各labelに`_DET_PAPER_SHAPED_CONTROL_01`を付与。元physical/canonicalを変えず、別のallocation SOURCEにvectorを固定する。初回は各候補1本だけ。目標は「この既存schedule形状の1本が、各保存rawに条件付くQ3閾値を満たすか」であり、N最小化・6方式rankingではない。

全shortlistを含む最小比較は**3 pair ×2＝6 call**。1 pairだけの技術pilotなら2 callだが、他の公式に結果を移送できず、6候補の比較完了とは表示しない。結果に依存して別vectorを自動投入しない。片側UNKNOWNなら未開始callを止め、開始済み独立jobの証拠は保存する。有限vector集合の拡張・縮小探索は次の別承認単位。

## SOURCE・有限scope・呼出し

1. A REPORT1adb7e65、数値fb9a1878、完成carrier06a15ee4。保存30＋新48というproducer履歴を保持し、元packet・canonicalを再生成しない。旧段階packetを使用しない。
2. B9c0e12d87以上の同一binding実装＋total kernel710aa69b＋accelerator9e294778、C c628e9e8のAPIを再利用。実行前に新caller/INPUTSをcommitする。今回のPauli専用runnerをラベルだけ変えて使わず、Aの6 locatorを明示的に消費する薄いcallerだけを追加する。
3. `T.create(root, original, original_source=...)`→`T.bind_signal(..., original_packet_source=...)`→`T.evaluate(..., allocation, evaluation_target, statistical_contract, options)`。Aのintake PASSは再利用し、SOURCE/viewが変化した部分だけを再検査。評価targetを明示し、packetの未採択proxyフィールドを書き換えない。
4. 保存packetの実rawを変えず、`EXACT_AFFINE_OF_BINARY64_RAW`・shots/axes/round間IIDを条件とする。PR用finite-Kを要求せず、deterministic evolutionに対する二項測定lawとして使用。bias²/Var/first-unsafe/zero HOLD/tie argmin/残余lossを全て保存。δEを加算しない。
5. 有限scopeの**提案**：ρ=1、round4225、verification55043173、frontier8192、transition141559036。上記設計vectorのcount-only上限はround3600、隣接組38,582,272、DP遷移117,985,808なので既存scopeに収まる。これは確率・MSE計算ではない。実count閉包の成立は未検査であり、成立を仮定しない。範囲やρを失敗後に無断変更しない。
6. 既存のsource/非有限/質量/閉包/上下界/残余/保存gateを維持。上界≤τで`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`、下界>τで`MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`、跨ぐと`INCONCLUSIVE`、数値・契約不成立は`UNKNOWN`。単なる上界超過は実精度未達の証明ではない。

6候補のMSEを評価しても、物理ground資格・target包含・実sampler移送・数値certificate・全shot空間最適N・適格E[C]/Gは未解決。Full/truncatedのH_eff固有値誤差と全RPE MSEは異なる量で、H_effのPASSを代用しない。直接riskに切断biasも入り得るため、保存biasを機械的に二重加算しない。

## 必要な次回承認と停止

ユーザーが採択済みモデル契約の下で、**この6構成×提案vector各1本＝6 call**、入力SOURCEと上記有限scope、UNKNOWN時の扱いを明示許可した場合だけ実施する。新しい承認証明書/grant等は不要。別vectorを希望する場合は実行前に一度だけvectorとcount-only範囲を確定する。

完了6件又は重大gate失敗でSTOP。結果が成功しても、評価済みvector、有限候補内比較、物理的十分N、全空間最適Nを区別する。適格compiled価格がない限りG/winnerを算出しない。**今回この計画は実行していない。**
